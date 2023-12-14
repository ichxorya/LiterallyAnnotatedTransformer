"""
Written by Nghia Be.

This file contains the Positional Encoder layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""

# Import necessary modules and packages.
from __future__ import annotations
import logging
import math
import torch
import torch.nn as nn
from typing_extensions import Callable


# Define the class PositionalEncoder.
class PositionalEncoder(nn.Module):
    """
    The Positional Encoder layer of the Transformer Model.
    - ELI5: Positional Encoding = Keep track of the order of words in a sentence.
    - Used to add positional information to the input sequence of tokens.

    Attributes:
        - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
        - `max_seq_length` (`int`): The maximum sequence length.
        - `dropout` (`torch.nn.Dropout`): The Dropout layer.
        - `positional_encoding` (`torch.Tensor`): The positional encoding. Registered as a buffer (non-parameter).

    Methods:
        - `__init__`: Initialize the Positional Encoder layer.
        - `forward(x: torch.Tensor) -> torch.Tensor`: Apply the positional encoding to the input sequence of tokens.
        - `splice_by_size(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor`: Splice the source tensor by the second dimension of the target tensor.
    """

    def __init__(self, d_model: int, max_seq_length: int, dropout_rate: float):
        """
        Initialize the Positional Encoder layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
            - `max_seq_length` (`int`): The maximum sequence length.
            - `dropout_rate` (`float`): The dropout rate of the Dropout layer.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the dimensionality of the model.
        self.d_model: int = d_model

        ## Define the maximum sequence length.
        self.max_seq_length: int = max_seq_length

        ## Define the Dropout layer.
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)

        ## Define the positional encoding.
        ### Initialize the positional encoding as a tensor of zeros.
        self.positional_encoding: torch.Tensor = torch.zeros(
            (max_seq_length, d_model)  # Shape: (max_seq_length, d_model)
        )

        ### For each position from 0 to max_seq_length,
        for pos in range(0, max_seq_length):
            ### And for each 2 dimensions from 0 to d_model (a.k.a each 2 tokens in the sequence):
            for dim in range(0, d_model, 2):
                ### Compute the positional encoding.
                self.positional_encoding[pos, dim] = math.sin(
                    pos / (10000 ** (2 * dim / d_model))
                )
                self.positional_encoding[pos, dim + 1] = math.cos(
                    pos / (10000 ** ((2 * dim + 1) / d_model))
                )

        ### Register the positional encoding as a buffer (non-parameter).
        self.register_buffer("positional_encoding", self.positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the positional encoding to the input sequence of tokens.

        - This method trims the input sequence to the maximum sequence length if it is longer than the maximum sequence length.
        - Then it scales the input sequence by the square root of the dimensionality of the model.
        - Then it adds the positional encoding to the input sequence.
        - And finally applies dropout to the output.

        Args:
            - `x` (`torch.Tensor`): The input sequence of tokens.

        Returns:
            - `x_o` (`torch.Tensor`): The input sequence of tokens with positional encoding.
        """
        # If the input sequence is longer than the maximum sequence length:
        if x.shape[1] > self.max_seq_length:
            # Trim the input sequence to the maximum sequence length.
            logging.warn(
                """
                The input sequence is longer than the maximum sequence length.

                Build a model with a larger `max_seq_length`  if you want to keep the input;
                or ignore if you want the input to be trimmed.
                """
            )
            x = x[:, : self.max_seq_length]

        # Scale the input sequence by the square root of the dimensionality of the model.
        x_o: torch.Tensor = torch.mul(x, math.sqrt(self.d_model))

        # Splice the positional encoding by the size of the input sequence.
        positional_encoding: torch.Tensor = self.splice_by_size(
            self.positional_encoding, x_o
        )
        self.positional_encoding = positional_encoding.requires_grad_(False)

        # Add the positional encoding to the input sequence.
        x_o = x_o + self.positional_encoding

        # Apply the Dropout layer.
        x_o = self.dropout(x_o)

        # Return the output.
        return x_o

    @torch.jit.script
    def splice_by_size(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Splice the source tensor by the second dimension of the target tensor.

        Args:
            - `source` (`torch.Tensor`): The source tensor.
            - `target` (`torch.Tensor`): The target tensor.

        Returns:
            - `spliced_source` (`torch.Tensor`): The spliced source tensor.
        """
        # Compute the length of the target.
        length: int = target.size(1)

        # Return the spliced source.
        spliced_source: torch.Tensor = source[:, :length]
        return spliced_source
