"""
Written by Nghia Be.

This file contains the Transformer class, which is the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""
# Import necessary modules and packages.
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.model_specific.transformer import (
    Encoder,
    Decoder,
)


# Define the Transformer class.
class Transformer(nn.Module):
    """
    This class implements the Transformer model.
    ELI5: Transformer ~ Encoder stack + Decoder stack.

    Attributes:
        - `encoder` (`modules.model_specific.transformer.Encoder`): The encoder stack.
        - `decoder` (`modules.model_specific.transformer.Decoder`): The decoder stack.
        - `out` (`nn.Linear`): The output layer.

    Methods:
        - `__init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, N: int, h: int,
        dropout_rate: float = 0.1, encoder_max_length: int = 100, decoder_max_length: int = 100)`: Initialize the Transformer model.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        N: int,
        h: int,
        dropout_rate: float = 0.1,
        encoder_max_length: int = 100,
        decoder_max_length: int = 100,
    ):
        """
        Initialize the Transformer model.

        Args:
            - `src_vocab_size` (`int`): The size of the source vocabulary.
            - `tgt_vocab_size` (`int`): The size of the target vocabulary.
            - `d_model` (`int`): The dimensionality of the model (the size of the embedding vector).
            - `N` (`int`): The number of encoder/decoder layers.
            - `h` (`int`): The number of attention heads.
            - `dropout_rate` (`float`): The dropout rate.
            - `encoder_max_length` (`int`): The maximum length of the input sequence.
            - `decoder_max_length` (`int`): The maximum length of the output sequence.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the encoder stack.
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            N,
            h,
            dropout_rate,
            max_seq_length=encoder_max_length,
        )

        ## Define the decoder stack.
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            N,
            h,
            dropout_rate,
            max_seq_length=decoder_max_length,
        )

        ## Define the output layer.
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Run the Transformer model.

        Args:
            - `src` (`torch.Tensor`): The input sequence of tokens.
            - `tgt` (`torch.Tensor`): The target sequence of tokens.

        Returns:
            - `torch.Tensor`: The output sequence of tokens.
        """
        # Run the encoder stack.
        encoder_o: torch.Tensor = self.encoder(src)

        # Run the decoder stack.
        decoder_o: torch.Tensor = self.decoder(tgt, encoder_o)

        # Run the output layer.
        output: torch.Tensor = self.out(decoder_o)

        # Apply softmax to the output.
        output = F.softmax(output, dim=-1)

        # Return the output.
        return output
