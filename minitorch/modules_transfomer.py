import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_projection   : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head
        self.use_fused_kernel = use_fused_kernel

        ### BEGIN YOUR SOLUTION
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)

        self.dropout = Dropout(p_dropout)
        ### END YOUR SOLUTION

    def create_causal_mask(self, seq_len):
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (broadcasted to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, K^T, V for self-attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        x = x.view(batch_size*seq_len, n_embd)
        ### BEGIN YOUR SOLUTION
        q = self.q_projection(x).view(batch_size, seq_len, self.n_head, n_embd//self.n_head).permute(0,2,1,3)
        kT = self.k_projection(x).view(batch_size, seq_len, self.n_head, n_embd//self.n_head).permute(0,2,3,1)
        v = self.v_projection(x).view(batch_size, seq_len, self.n_head, n_embd//self.n_head).permute(0,2,1,3)
        ### END YOUR SOLUTION
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """Compute scaled dot-product attention and return the MultiHeadAttention output.
           softmax((q @ kT)/sqrt(attn_hidden_dim)) @ V.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        ### BEGIN YOUR SOLUTION
        attention_scores = (q @ kT) / np.sqrt(self.attn_hidden_dim)
        if self.causal:
            mask = self.create_causal_mask(queries_len)
            attention_scores += mask

        if self.use_fused_kernel:
            # 
            attention_weights = attention_scores.f.attn_softmax_fw(attention_scores, None)
        else:
            attention_weights = softmax(attention_scores, dim=3).contiguous()

        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ v
        output = output.permute(0,2,1,3).contiguous().view(batch_size, queries_len, num_head*q_dim)
        ### END YOUR SOLUTION
        return output

    def forward(self, x):
        """Compute MultiHeadAttention with optional causal masking.

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        q, kT, v = self.project_to_query_key_value(x)
        output = self.self_attention(q, kT, v)
        output = self.out_projection(output)
        ### END YOUR SOLUTION
        return output


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.

        Args:
            n_embd     : Input dimension for first linear layer and output dimension for second linear layer.
            middle_dim : Intermediate dimension.
            p_dropout  : Dropout probability.
            bias       : Whether to include bias.
        """
        ### BEGIN YOUR SOLUTION 
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END YOUR SOLUTION

    def forward(self, x):
        """FeedForward network with GELU activation and dropout.

        Args:
            x : Tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        x = GELU(self.linear_in(x.view(batch_size*seq_len, n_embd)))
        x = self.linear_out(x)
        x = self.dropout(x).view(batch_size, seq_len, n_embd)
        ### END YOUR SOLUTION
        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd   : Embedding and hidden state dimension.
            n_head   : Number of attention heads.
            p_dropout: Dropout ratio.
            ln_eps   : Epsilon for numerical stability in LayerNorm.
            bias     : Whether to include bias.
        Attributes:
            ln_1 : LayerNorm before MultiHeadAttention.
            ln_2 : LayerNorm before FeedForward.
            attention : MultiHeadAttention layer.
            ff  : FeedForward layer.
        """
        ### BEGIN YOUR SOLUTION
        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ff = FeedForward(n_embd, middle_dim=256, p_dropout=p_dropout, bias=bias, backend=backend)
        self.use_fused_kernel = use_fused_kernel
        ### END YOUR SOLUTION

    def forward(self, x):
        """Forward pass for a Transformer Layer.

        Args: 
            x : Tensor of shape (batch_size, seq_len, n_embd)
        
        Returns: 
            output: Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        # Step 1: LayerNorm before MultiHeadAttention
        if self.use_fused_kernel:
            norm_x = x.f.layernorm_fw(x, self.ln_1.gamma, self.ln_1.beta)
        else:
            norm_x = self.ln_1(x.view(batch_size*seq_len, n_embd))
            norm_x = norm_x.view(batch_size, seq_len, n_embd)
        attn_output = self.attention(norm_x)
        x = x + attn_output

        # Step 2: LayerNorm before FeedForward
        if self.use_fused_kernel:
            norm_x = x.f.layernorm_fw(x, self.ln_2.gamma, self.ln_2.beta)
        else:
            norm_x = self.ln_2(x.view(batch_size*seq_len, n_embd))
            norm_x = norm_x.view(batch_size, seq_len, n_embd)
        ff_output = self.ff(norm_x)
        output = x + ff_output
        ### END YOUR SOLUTION
        return output


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        """A full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab    : Vocabulary size.
            n_embd     : Embedding dimension.
            n_head     : Number of attention heads.
            n_positions: Maximum sequence length.
            p_dropout  : Dropout ratio.
            ln_eps     : Epsilon for LayerNorm.
            bias       : Whether to include bias.
        
        Attributes:
            token_embeddings    : Token embedding layer.
            position_embeddings : Positional embedding layer.
            t_layer_1, t_layer_2, t_layer_3, t_layer_4 : 4 Transformer Layers.
            dropout     : Dropout layer.
            ln          : Final LayerNorm.
            lm_head     : Linear layer projecting to vocabulary logits.
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        ### BEGIN YOUR SOLUTION
        self.token_embeddings    = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout             = Dropout(p_dropout)
        self.ln                  = LayerNorm1d(n_embd, ln_eps, backend)
        self.lm_head             = Linear(n_embd, n_vocab, bias, backend)
        self.use_fused_kernel    = use_fused_kernel
        ### END YOUR SOLUTION
    
    def forward(self, idx):
        """Forward pass for DecoderLM.

        Args:
            idx: Tensor of shape (batch_size, seq_len)
        
        Returns:
            logits: Tensor of shape (batch_size, seq_len, n_vocab)
        """
        batch_size, seq_len = idx.shape
        ### BEGIN SOLUTION
        token_embedding = self.token_embeddings(idx)
        position_ids = tensor_from_numpy(np.arange(seq_len).reshape(1, seq_len), backend=self.backend)
        pos_emb = self.position_embeddings(position_ids)
        x = token_embedding + pos_emb
        x = self.dropout(x)
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        if self.use_fused_kernel:
            norm_x = x.f.layernorm_fw(x, self.ln.gamma, self.ln.beta)
        else:
            norm_x = self.ln(x)
        logits = self.lm_head(norm_x)
        ### END SOLUTION
        return logits
