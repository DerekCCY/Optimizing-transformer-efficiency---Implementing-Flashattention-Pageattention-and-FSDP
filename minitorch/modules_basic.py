"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np
import math

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor
import minitorch

from typing import Any, Dict, Optional, Sequence, Tuple
    

from minitorch import Tensor, Parameter, one_hot, tensor_from_numpy, rand

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings  # Vocabulary size
        self.embedding_dim = embedding_dim  # Embedding dimension
        
        self.weights = Parameter(rand((num_embeddings, embedding_dim), backend=backend, requires_grad=True))

    def forward(self, x: Tensor):
        """Maps word indices to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        one_hot_x = one_hot(x, self.num_embeddings)  # (batch_size, seq_len, num_embeddings)

        if not isinstance(one_hot_x, Tensor):
            one_hot_x = tensor_from_numpy(one_hot_x, backend=self.backend)
            
        output =  (one_hot_x.view(bs * seq_len, self.num_embeddings) @ self.weights.value).view(bs, seq_len, self.embedding_dim) 

        return output

    
class Dropout(Module):
    def __init__(self, p_dropout: float = 0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        # If not training, return x unchanged (fix for inference mode)
        if not self.training or self.p_dropout == 0:
            return x
        
        # Generate dropout mask
        mask_np = np.random.binomial(1, 1 - self.p_dropout, size=x.shape)
        mask = Parameter(tensor_from_numpy(mask_np, backend=x.backend))

        # Apply dropout and scale accordingly
        output = (x * mask.value) / (1 - self.p_dropout)

        return output


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-sqrt(1/in_size), sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-sqrt(1/in_size), sqrt(1/in_size)).
        """
        bound = math.sqrt(1.0/in_size)
        self.weights = Parameter(rand((in_size, out_size), backend=backend, requires_grad=True)* (2 * bound) - bound)
        if bias:
            self.bias = Parameter(rand((out_size,), backend = backend, requires_grad=True) * (2 * bound) - bound)
        self.out_size = out_size

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        original_shape = x.shape  # Save for reshaping later
        in_size = original_shape[-1]  # Last dimension is always in_size
        out_size = self.out_size

        # Flatten only if x has more than 2 dimensions (i.e., when used in transformers)
        if len(original_shape) > 2:
            batch_dims = original_shape[:-1]  # Everything except last dim
            flattened_dim = int(np.prod(batch_dims)) 
            x = x.view(flattened_dim, in_size)  
        else:
            batch_dims = original_shape[:-1]  # Should be (batch_size,)

        # Apply Linear transformation
        output = x @ self.weights.value.view(in_size, out_size)

        if self.bias is not None:
            output += self.bias.value

        # Restore the original batch dimensions
        return output.view(*batch_dims, out_size)


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(ones((dim,), backend))
        self.bias = Parameter(zeros((dim,), backend))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        #batch, dim = x.shape
        ### BEGIN YOUR SOLUTION
        mean = x.mean(1) # Compute mean along feature axis
        var = x.var(1)  # Compute variance along feature axis
        
        x_norm = (x - mean) / ((var + self.eps).__pow__(0.5))  # Normalize input
        ### END YOUR SOLUTION
        return x_norm * self.weights.value + self.bias.value