"""
Written by Nghia Be.

This file contains the Embedder layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""
# Import necessary modules and packages.
from __future__ import annotations
import math
import torch
import torch.nn as nn


# Define the class Embedder.
class Embedder(nn.Module):
    """
    The Embedder layer of the Transformer Model.
    - ELI5: Word embedding = Vectorization of words (words-as-vectors)
    - Used to embed the input sequence of tokens into a sequence of vectors.

    Attributes:
        - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
        - `lut` (`torch.nn.Embedding`): The embbeder's lookup table (a.k.a the word embeddings).

    Methods:
        - `__init__`: Initialize the Embedder layer.
        - `forward(x: torch.Tensor) -> torch.Tensor`: Embed the input sequence of tokens into a sequence of vectors.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the Embedder layer.

        Args:
            - `d_model` (`int`): The dimensionality of the model (the size of the embedding vector).
            - `vocab_size` (`int`): The size of the vocabulary.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Define the dimensionality of the model.
        self.d_model: int = d_model

        ## Define the embedding lookup table.
        self.lut: nn.Embedding = nn.Embedding(
            num_embeddings=vocab_size,  # Size of the dictionary of embeddings.
            embedding_dim=d_model,  # Size of each embedding vector.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed the input sequence of tokens into a sequence of vectors.

        - This method embeds the input sequence of tokens into a sequence of vectors
        using the lookup table defined in the constructor.
        - Then it scales the embeddings by the square root of the model's dimensionality.

        Args:
            - `x` (`torch.Tensor`): The input sequence of tokens.

        Returns:
            - `x_o` (`torch.Tensor`): The embedded sequence of vectors.
        """
        # Pass the input sequence through the embedding lookup table.
        x_o: torch.Tensor = self.lut(x)

        # Scale the embeddings by the square root of the model's dimensionality.
        ## The authors of the paper didn't explain why they did this.
        x_o = torch.mul(x_o, math.sqrt(self.d_model))

        # Return the output.
        return x_o
