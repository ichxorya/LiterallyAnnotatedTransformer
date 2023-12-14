"""
Written by Nghia Be.

This file contains the Postion-wise Feed-forward layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""

# Import necessary modules and packages.
from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the class FeedForward.
class FeedForward(nn.Module):
    """
    The Position-wise Feed-forward layer of the Transformer Model.
    - ELI5: Position-wise = Apply the same transformation to each position (word) in the sequence.
    - Used to add non-linearity to the model.

    Attributes:
        - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
        - `d_ff` (`int`): The dimensionality of the feed forward layer. Defaults to 2048.
        - `dropout` (`torch.nn.Dropout`): The Dropout layer. Defaults to 0.1.
        - `w_1` (`nn.Linear`): The first Linear layer.
        - `w_2` (`nn.Linear`): The second Linear layer.

    Methods:
        - `__init__`: Initialize the Position-wise Feed-forward layer.
        - `forward(x: torch.Tensor) -> torch.Tensor`: Apply the position-wise feed-forward transformation to the input sequence of vectors.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        internal_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the Position-wise Feed-forward layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
            - `d_ff` (`int`): The dimensionality of the feed forward layer. Defaults to 2048.
            - `internal_activation` (`Callable[[torch.Tensor], torch.Tensor]`): The internal activation function. Defaults to ReLU.
            - `dropout_rate` (`float`): The dropout rate of the Dropout layer.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the dimensionality of the model.
        self.d_model: int = d_model

        ## Define the dimensionality of the feed forward layer.
        self.d_ff: int = d_ff

        ## Define the Dropout layer.
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)

        ## Define the first Linear layer.
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)

        ## Define the second Linear layer.
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)

        ## Define the internal activation function.
        self.internal_activation: Callable[
            [torch.Tensor], torch.Tensor
        ] = internal_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward transformation to the input sequence of vectors.
        - This method applies a linear transformation to the input sequence of vectors.
        - It then applies the internal activation function to the intermediate sequence.
        - And finally applies dropout to the output.

        Args:
            - `x` (`torch.Tensor`): The input sequence of vectors.

        Returns:
            - `x_o` (`torch.Tensor`): The output sequence of vectors.
        """
        # Feed the input sequence of vectors to the first linear layer and apply the internal activation function.
        x_o: torch.Tensor = self.internal_activation(self.w_1(x))

        # Apply dropout to the intermediate sequence then feed it to the second linear layer.
        x_o = self.w_2(self.dropout(x_o))

        # Return the output.
        return x_o
