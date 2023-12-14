"""
Written by Nghia Be.

This file contains the Add & Norm layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""
# Import necessary modules and packages.
from __future__ import annotations
import torch
import torch.nn as nn


# Define the class Normalizer.
class Normalizer(nn.Module):
    """
    The Add & Norm layer of the Transformer Model.
    - ELI5: Add & Norm = Join a residual connection and then normalize the result.

    Attributes:
        - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
        - `alpha` (`nn.Parameter`): The multiplicative parameter of the layer normalization.
        - `bias` (`nn.Parameter`): The additive parameter of the layer normalization.
        - `epsilon` (`float`): The epsilon value of the layer normalization. Defaults to 1e-6.

    Methods:
        - `__init__`: Initialize the Add & Norm layer.
        - `forward(x: torch.Tensor) -> torch.Tensor`: Join the residual connection and then normalize the result.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-6):
        """
        Initialize the Add & Norm layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
            - `epsilon` (`float`): The epsilon value of the layer normalization. Defaults to 1e-6.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the dimensionality of the model.
        self.d_model: int = d_model

        ## Define two learnable parameters to calibrate the normalization.
        self.alpha: nn.Parameter = nn.Parameter(torch.ones(self.d_model))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.d_model))

        ## Define the epsilon value of the layer normalization.
        self.epsilon: float = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Join the residual connection and then normalize the result.

        Args:
            - `x` (`torch.Tensor`): The input tensor.

        Returns:
            - `output` (`torch.Tensor`): The output tensor.
        """
        # Normalize the result.
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        std: torch.Tensor = x.std(dim=-1, keepdim=True)
        output: torch.Tensor = (
            self.alpha * (x - mean) / (std + self.epsilon) + self.bias
        )

        # Return the output.
        return output
