"""
Written by Nghia Be.

This file contains the Decoder layer used in the Transformer model.

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


# Define the class Decoder.
class Decoder(nn.Module):
    """
    The Decoder layer of the Transformer model.

    Attributes:

    Methods:
    """

    def __init__(self, d_model: int, h: int, dropout_rate: float = 0.1):
        """
        Initialize the Decoder layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (the size of the embedding vector).
            - `h` (`int`): The number of attention heads.
            - `dropout_rate` (`float`): The dropout rate.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the normalizer layers.
        self.norm_1: Normalizer = Normalizer(d_model)
        self.norm_2: Normalizer = Normalizer(d_model)
        self.norm_3: Normalizer = Normalizer(d_model)

        ## Define the dropout layers.
        self.dropout_1: nn.Dropout = nn.Dropout(dropout_rate)
        self.dropout_2: nn.Dropout = nn.Dropout(dropout_rate)
        self.dropout_3: nn.Dropout = nn.Dropout(dropout_rate)

        ## Define the feed-forward layer.
        self.feed_forward: FeedForward = FeedForward(d_model, dropout_rate=dropout_rate)

        ## Define the multi-head attention layers.
        ### Self-attention layer.
        self.self_attention: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout_rate
        )
        ### Encoder-decoder attention layer.
        self.encoder_decoder_attention: MultiHeadAttention = MultiHeadAttention(
            h, d_model, dropout_rate
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_o: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the Decoder layer.

        Args:
            - `x` (`torch.Tensor`): The input sequence of tokens.
            - `encoder_o` (`torch.Tensor`): The output of the Encoder layer, used for encoder-decoder attention.
            - `src_mask` (`torch.Tensor`): The padding mask for the encoder's output.
            - `tgt_mask` (`torch.Tensor`): The look-ahead (no-peeking) mask for the decoder.

        Return:
            - `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`:
                - `x_o` (`torch.Tensor`): The output tensor.
                - `self_attention_scores` (`torch.Tensor`): The self-attention scores.
                - `encoder_decoder_attention_scores` (`torch.Tensor`): The encoder-decoder attention scores.

        Notes:
            - `x` should have the shape [batch_size, tgt_len, d_model].
            - `encoder_o` should have the shape [batch_size, src_len, d_model].
            - `src_mask` should have the shape [batch_size, 1, src_len].
            - `tgt_mask` should have the shape [batch_size, tgt_len, tgt_len].
            - `x_o` has the same shape as `x`.
            - `self_attention_s` has the shape [batch_size, h, tgt_len, tgt_len].
            - `encoder_decoder_attention_s` has the shape [batch_size, h, tgt_len, src_len].
        """
        # Normalize the input tensor.
        normalized_input: torch.Tensor = self.norm_1(x)

        # Apply self-attention to the normalized input.
        self_attention_o, self_attention_s = self.self_attention(
            q=normalized_input, k=normalized_input, v=normalized_input, mask=tgt_mask
        )

        # Add the self-attention output to the input tensor after applying dropout.
        x_o: torch.Tensor = x + self.dropout_1(self_attention_o)

        # Normalize the updated input tensor.
        normalized_input = self.norm_2(x_o)

        # Apply encoder-decoder attention to the normalized input.
        (
            encoder_decoder_attention_o,
            encoder_decoder_attention_s,
        ) = self.encoder_decoder_attention(
            q=normalized_input, k=encoder_o, v=encoder_o, mask=src_mask
        )

        # Add the encoder-decoder attention output to the input tensor after applying dropout.
        x_o = x_o + self.dropout_2(encoder_decoder_attention_o)

        # Normalize the updated input tensor.
        normalized_input = self.norm_3(x_o)

        # Add the output of the feed-forward network to the input tensor after applying dropout.
        x_o = x_o + self.dropout_3(self.feed_forward(normalized_input))

        # Return the final output tensor, the self-attention scores, and the encoder-decoder attention scores.
        return (x_o, self_attention_s, encoder_decoder_attention_s)
