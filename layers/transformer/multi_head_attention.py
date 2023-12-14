"""
Written by Nghia Be.

This file contains the Multi-Head Attention layer used in the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""

# Import necessary modules and packages.
from __future__ import annotations
import math
from typing_extensions import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the class MultiHeadAttention.
class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention layer of the Transformer Model.
    - ELI5: Multi-Head Attention = Attention with multiple heads.
            Attention = A mechanism that allows the model to focus on the relevant parts of the input sequence
                        when generating the output sequence.
    - Used to calculate the attention output and attention scores.

    Attributes:
        - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
        - `h` (`int`): The number of heads.
        - `d_k` (`int`): The dimensionality of each head.
        - `dropout` (`nn.Dropout`): The Dropout layer.
        - `w_q` (`nn.Linear`): The query as a Linear layer.
        - `w_k` (`nn.Linear`): The key as a Linear layer.
        - `w_v` (`nn.Linear`): The value as a Linear layer.
        - `w_o` (`nn.Linear`): The output as a Linear layer.

    Methods:
        - `__init__(self, h: int, d_model: int, dropout_rate: float = 0.1)`: Initialize the Multi-Head Attention layer.
        - `forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None)
        -> Tuple[torch.Tensor, torch.Tensor]`: Calculate the attention output and attention scores, using the Scaled Dot-product Attention with multiple heads.
        - `attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None)
        -> Tuple[torch.Tensor, torch.Tensor]`: Calculate the attention output and attention scores, using the Scaled Dot-product Attention.
    """

    def __init__(self, h: int, d_model: int, dropout_rate: float = 0.1):
        """
        Initialize the Multi-Head Attention layer.

        Args:
            - `h` (`int`): The number of heads.
            - `d_model` (`int`): The dimensionality of the model (a.k.a the size of the word embedding).
            - `dropout_rate` (`float`): The dropout rate of the Dropout layer.

        Raises:
            - `AssertionError`: If `d_model` is not divisible by `h`.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Check if `d_model` is divisible by `h`.
        assert d_model % h == 0, "`d_model` must be divisible by `h`."

        ## Define the dimensionality of the model.
        self.d_model: int = d_model

        ## Define the number of heads.
        self.h: int = h

        ## Define the dimensionality of each head.
        self.d_k: int = d_model // h

        ## Define the Dropout layer.
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_rate)

        ## Define the query, key, and value as Linear layers.
        self.w_q: nn.Linear = nn.Linear(d_model, d_model)
        self.w_k: nn.Linear = nn.Linear(d_model, d_model)
        self.w_v: nn.Linear = nn.Linear(d_model, d_model)

        ## Define the output Linear layer.
        self.w_o: nn.Linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the attention output and attention scores, using the Scaled Dot-product Attention with multiple heads.

        Args:
            - `q` (`torch.Tensor`): The query.
            - `k` (`torch.Tensor`): The key.
            - `v` (`torch.Tensor`): The value.
            - `mask` (`Optional[torch.Tensor]`): The mask to apply to the attention scores. Defaults to `None`.

        Returns:
            - `Tuple[torch.Tensor, torch.Tensor]`:
                - `attention_o` (`torch.Tensor`): The output of the attention process.
                - `attention_s` (`torch.Tensor`): The attention scores.

        Notes:
            - `q`, `k`, and `v` should all have the same shape: [batch_size, sequence_length, d_model].
               The `sequence_length` only differs in the decoder attention,
               where `q` has the shape [batch_size, tgt_len, d_model] and `k` and `v` have the shape [batch_size, src_len, d_model].
            - `mask` should have the shape: [batch_size, 1, sequence_length] or [batch_size, sequence_length, sequence_length].
        """
        # Calculate the attention output and attention scores.
        ## Get the batch size.
        batch_size = q.shape[0]

        ## Apply the Linear layers to the query, key, and value.
        q = self.w_q(q)
        k = self.w_q(k)
        v = self.w_v(v)

        ## Split the query, key, and value into multiple heads.
        q = q.view(batch_size, -1, self.h, self.d_k)
        k = k.view(batch_size, -1, self.h, self.d_k)
        v = v.view(batch_size, -1, self.h, self.d_k)

        ## Transpose the dimensions to apply the attention.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        ## Apply the attention.
        attention_o, attention_s = self.attention(q, k, v, mask, self.dropout)
        attention_o = (
            attention_o.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        attention_o = self.w_o(attention_o)

        # Return the attention output and attention scores.
        return (attention_o, attention_s)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the attention output and attention scores, using the Scaled Dot-product Attention.

        Args:
            - `q` (`torch.Tensor`): The query.
            - `k` (`torch.Tensor`): The key.
            - `v` (`torch.Tensor`): The value.
            - `mask` (`Optional[torch.Tensor]`): The mask to apply to the attention scores. Defaults to `None`.
            - `dropout` (`Optional[nn.Dropout]`): The Dropout layer. Defaults to `None`.

        Returns:
            - `Tuple[torch.Tensor, torch.Tensor]`:
                - `attention_o` (`torch.Tensor`): The output of the attention process.
                - `attention_s` (`torch.Tensor`): The attention scores.

        Notes:
            - `q`, `k`, and `v` should all have the same shape: [batch_size, sequence_length, d_model].
               The `sequence_length` only differs in the decoder attention,
               where `q` has the shape [batch_size, tgt_len, d_model] and `k` and `v` have the shape [batch_size, src_len, d_model].

            - `mask` should have the shape: [batch_size, 1, sequence_length] or [batch_size, sequence_length, sequence_length].
               The last two dimensions must match or are broadcastable.
        """
        # Calculate the attention scores.
        ## This is the attention scores: (Q * K^T) / sqrt(d_k).
        attention_s: torch.Tensor = torch.matmul(q, k.transpose(-2, -1))
        attention_s = torch.mul(attention_s, 1 / math.sqrt(self.d_k))

        ## Apply the mask if it is provided.
        if mask is not None:
            # Add a dimension for the heads.
            mask = mask.unsqueeze(1)
            # Apply the mask to the attention scores.
            attention_s = attention_s.masked_fill(mask == 0, -float("inf"))

        ## Apply softmax to the attention scores to get the attention probabilities.
        attention_s = F.softmax(attention_s, dim=-1)

        ## Apply dropout if it is provided.
        if dropout is not None:
            attention_s = dropout(attention_s)

        # Calculate the attention output.
        attention_o: torch.Tensor = torch.matmul(attention_s, v)

        # Return the attention output and attention scores.
        return (attention_o, attention_s)
