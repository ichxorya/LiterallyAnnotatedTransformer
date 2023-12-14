"""
Written by Nghia Be.

This file contains the Encoder layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""

# Import necessary modules and packages.
from __future__ import annotations
from . import Normalizer, FeedForward, MultiHeadAttention
from typing_extensions import Tuple
import torch
import torch.nn as nn


# Define the class Encoder.
class Encoder(nn.Module):
    """
    The Encoder layer of the Transformer model.

    Attributes:
        - `self_attention` (`MultiHeadAttention`): The self-attention layer.
        - `feed_forward` (`FeedForward`): The feed-forward layer.
        - `norm_1` (`Normalizer`): The first normalizer layer.
        - `norm_2` (`Normalizer`): The second normalizer layer.
        - `dropout_1` (`nn.Dropout`): The first dropout layer.
        - `dropout_2` (`nn.Dropout`): The second dropout layer.

    Methods:
        - `__init__`: Initialize the Encoder layer.
        - `forward(x: torch.Tensor, source_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Run the Encoder layer.
    """

    def __init__(self, d_model: int, h: int, dropout_rate: float = 0.1):
        """
        Initialize the Encoder layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (the size of the embedding vector).
            - `h` (`int`): The number of attention heads.
            - `dropout_rate` (`float`): The dropout rate.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the multi-head attention layer.
        self.self_attention: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout_rate
        )

        ## Define the feed-forward layer.
        self.feed_forward: FeedForward = FeedForward(d_model, dropout_rate=dropout_rate)

        ## Define the normalizer layers.
        self.norm_1: Normalizer = Normalizer(d_model)
        self.norm_2: Normalizer = Normalizer(d_model)

        ## Define the dropout layers.
        self.dropout_1: nn.Dropout = nn.Dropout(dropout_rate)
        self.dropout_2: nn.Dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, source_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Encoder layer.

        Args:
            - `x` (`torch.Tensor`): The input sequence of tokens.
            - `source_mask` (`torch.Tensor`): The padding mask.

        Return:
            - `Tuple[torch.Tensor, torch.Tensor]`:
                - `x_o` (`torch.Tensor`): The output tensor.
                - `self_attention_s` (`torch.Tensor`): The self-attention scores.

        Notes:
            - `x` should have the shape [batch_size, src_len, d_model].
            - `source_mask` should have the shape [batch_size, 1, src_len].
            - `x_o` has the same shape as `input_tensor`.
        """
        # Normalize the input tensor.
        normalized_input: torch.Tensor = self.norm_1(x)

        # Apply self-attention to the normalized input.
        self_attention_o, self_attention_s = self.self_attention(
            q=normalized_input, k=normalized_input, v=normalized_input, mask=source_mask
        )

        # Add the self-attention output to the input tensor after applying dropout
        x_o: torch.Tensor = x + self.dropout_1(self_attention_o)

        # Normalize the updated input tensor
        normalized_input_2: torch.Tensor = self.norm_2(x_o)

        # Add the output of the feed-forward network to the input tensor after applying dropout
        x_o = x + self.dropout_2(self.feed_forward(normalized_input_2))

        # Return the final output tensor and the self-attention scores
        return (x_o, self_attention_s)
