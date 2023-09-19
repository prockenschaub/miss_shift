# Based on https://github.com/marineLM/NeuMiss_sota

import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, Sequential, BatchNorm1d


class Mask(nn.Module):
    """A mask non-linearity.
    
    Args: 
        input: the input from which to create the mask
    """
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor, invert=False) -> Tensor:
        """Mask the input

        Args:
            input: input to mask
            invert: mode of masking. If False, mask the NON-missing values. If True, mask the missing
                values. Defaults to False.

        Returns:
            the masked input
        """
        if invert: 
            return ~self.mask * input
        return self.mask*input


class SkipConnection(nn.Module):
    """A skip connection operation.
    
    Args: 
        value: the value to add in the skipping
    """
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        """Add the value coming through the skip connection to the input 

        Args:
            input: current input

        Returns:
            the input plus the skip connection 
        """
        return input + self.value


class NeuMICEBlock(nn.Module):
    """A single NeuMICE block that is applied to the data multiple times

    Args:
        n_features : dimension of inputs and outputs of the NeuMICE block.
        depth : number of layers (iterations) in the NeuMICE block.
        dtype : Pytorch dtype for the parameters. Default: torch.float.
    """
    def __init__(self, n_features: int, depth: int, dtype = torch.float):
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.init = Linear(n_features, n_features, bias=True, dtype=dtype)
        self.linear = Linear(n_features, n_features, bias=True, dtype=dtype)
        self.norm = BatchNorm1d(n_features, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Take partially-observed data and embed it into a fully observed vector

        Args:
            x: input of shape (n, d) with missing values

        Returns:
            filled in (n, d) Tensor without missingness
        """
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)          # Initialize mask non-linearity
        x = torch.nan_to_num(x) # Fill missing values with 0     
        skip = SkipConnection(x)# Initialize skip connection with this value

        s0 = self.init(x)       # Choose initial "imputations"
        layer = [self.norm, self.linear, mask, skip]  # One NeuMICE iteration
        layers = Sequential(*(layer*self.depth))      # Several NeuMICE iterations = 1 block

        return layers(s0)

    def reset_parameters(self) -> None:
        """Initialies parameters
        """
        nn.init.xavier_uniform_(self.init.weight, gain=0.5)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)
