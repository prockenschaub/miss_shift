import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, Sequential


class Mask(nn.Module):
    """A mask non-linearity.
    
    Args: 
        input: the input from which to create the mask
    """
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        """Set all values to zero that were missing in the original input

        Args:
            input: input to mask

        Returns:
            the masked input
        """
        return ~self.mask*input


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


class NeuMissBlock(nn.Module):
    """The NeuMiss block from "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux.
    
    Args:
        n_features : dimension of inputs and outputs of the NeuMiss block.
        depth : number of layers (Neumann iterations) in the NeuMiss block.
        dtype : Pytorch dtype for the parameters. Default: torch.float.
    """

    def __init__(self, n_features: int, depth: int,
                 dtype=torch.float) -> None:
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Take partially-observed data and embed it into a fully observed vector

        Args:
            x: input of shape (n, d) with missing values

        Returns:
            filled in (n, d) Tensor without missingness
        """
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        """Initialies parameters
        """
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)
    