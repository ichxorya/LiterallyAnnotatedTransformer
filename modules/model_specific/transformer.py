"""
Written by Nghia Be.

This file contains the Encoder and Decoder classes used in the Transformer model.

Notes: The Encoder and Decoder defined here are not layers, but stacks of layers.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""
# Import necessary modules and packages.
from __future__ import annotations
import layers
import torch
import torch.nn as nn
import copy
from typing_extensions import List, Tuple


def get_clones(module: nn.Module, N: int, keep_module: bool = True) -> nn.ModuleList:
    """
    Create N copies of the given module.

    Args:
    - `module (nn.Module)`: The module to be copied.
    - `N (int)`: Number of copies to be created.
    - `keep_module (bool)`: Whether to keep the original module or not.

    Raises:
    - `ValueError`: If N < 1.

    Returns:
    - `nn.ModuleList`: A list of N copies of the given module.
    """
    # Check if N is valid.
    if N < 1:
        raise ValueError("N must be greater than or equal to 1.")

    # If keep_module is True and N >= 1:
    ## Create N - 1 new copies + the original module (N copies in total).
    if keep_module and N >= 1:
        return nn.ModuleList([module] + [copy.deepcopy(module) for i in range(N - 1)])
    ## Otherwise, create N new copies.
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    """
    This class implements the Encoder stack.
    ELI5: Encoder stack ~ N encoder layers + Input embedding + Positional encoding.

    Attributes:
    - `embed (layers.Embedder)`: The embedding layer.
    - `pe (layers.PositionalEncoder)`: The positional encoding layer.
    - `encoder_layers (nn.ModuleList)`: The list of encoder layers.
    - `norm (layers.Normalizer)`: The normalizer layer.
    - `max_seq_length (int)`: The maximum sequence length.

    Methods:
    - `__init__(vocab_size: int, d_model: int, N: int, h: int, dropout_rate: float, max_seq_length: int = 200)`: Initialize the Encoder stack.
    - `forward(src: torch.Tensor, src_mask: torch.Tensor, output_attention_s: bool = False, seq_length_check: bool = False) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]`: Take a batch of indexed tokens, embed them, add positional encoding, then pass them through N encoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        N: int,
        h: int,
        dropout_rate: float,
        max_seq_length: int = 200,
    ):
        """
        Initialize the Encoder stack.

        Args:
        - `vocab_size (int)`: The size of the vocabulary.
        - `d_model (int)`: The dimensionality of the model (the size of the embedding vector).
        - `N (int)`: The number of encoder layers.
        - `h (int)`: The number of attention heads.
        - `dropout_rate (float)`: The dropout rate.
        - `max_seq_length (int)`: The maximum sequence length.

        Raises:
        - `ValueError`: If N < 1.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Check if N is valid.
        if N < 1:
            raise ValueError("N must be greater than or equal to 1.")

        ## Define the number of encoder layers.
        self.N = N

        ## Define the embedding layer.
        self.embed = layers.Embedder(vocab_size, d_model)

        ## Define the positional encoding layer.
        self.pe = layers.PositionalEncoder(d_model, max_seq_length, dropout_rate)

        ## Define the encoder layers.
        self.encoder_layers = get_clones(layers.Encoder(d_model, h, dropout_rate), N)

        ## Define the normalizer layer.
        self.norm = layers.Normalizer(d_model)

        ## Define the max sequence length.
        self.max_seq_length = max_seq_length

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        output_attention_s: bool = False,
        seq_length_check: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Take a batch of indexed tokens, embed them, add positional encoding, then pass them through N encoder layers.

        Args:
            -`src (torch.Tensor)`: The batch of indexed tokens (in integer form).
            -`src_mask (torch.Tensor)`: The padding mask.
            -`output_attention_s (bool)`: Whether to output the attention scores or not. Defaults to False.
            -`seq_length_check (bool)`: Whether to check the sequence length or not. Defaults to False.
        Returns:
            - `attention_o` (`torch.Tensor`): The output of the Encoder stack (the attention output).

            or

            - `(attention_o, attention_s_list)` (`Tuple[torch.Tensor, List[torch.Tensor]]`): The output of the Encoder stack and the list of attention scores.

        Notes:
            - `src` has the shape [batch_size, src_len].
            - `src_mask` has the shape [batch_size, 1, src_len].
            - `attention_o` has the shape [batch_size, src_len, d_model].
            - `attention_s_list` has the shape [N], each element has the shape [batch_size, heads, src_len, src_len].
        """
        # Check if seq_length_check is True and the sequence length is greater than the maximum sequence length.
        ## If so, truncate the sequence.
        if seq_length_check and src.shape[1] > self.max_seq_length:
            src = src[:, : self.max_seq_length]
            src_mask = src_mask[:, :, : self.max_seq_length]

        # Embed the source sequence and add positional encoding.
        x: torch.Tensor = self.embed(src)
        x = self.pe(x)

        # Initialize the list of attention scores.
        attention_s_list: List[torch.Tensor] = [torch.Tensor(None)] * self.N

        # Pass the embedded sequence through N encoder layers.
        for i in range(self.N):
            x, attention_s = self.encoder_layers[i](x, src_mask)
            attention_s_list[i] = attention_s

        # Normalize the output of the Encoder stack.
        attention_o: torch.Tensor = self.norm(x)

        # If output_attention_s is True, return the list of attention scores, too.
        if output_attention_s:
            return (attention_o, attention_s_list)
        else:
            return attention_o


class Decoder(nn.Module):
    """
    This class implements the Decoder stack.

    ELI5: Decoder stack ~ N decoder layers + Input embedding + Positional encoding.

    Attributes:
    - `embed (layers.Embedder)`: The embedding layer.
    - `pe (layers.PositionalEncoder)`: The positional encoding layer.
    - `decoder_layers (nn.ModuleList)`: The list of decoder layers.
    - `norm (layers.Normalizer)`: The normalizer layer.
    - `max_seq_length (int)`: The maximum sequence length.

    Methods:
    - `__init__(vocab_size: int, d_model: int, N: int, h: int, dropout_rate: float, max_seq_length: int = 200)`: Initialize the Decoder stack.
    - `forward(tgt: torch.Tensor, encoder_o: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor, output_attention: bool = False) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]`: Take a batch of indexed tokens and the encoding outputs, embed them, add positional encoding, then pass them through N decoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        N: int,
        h: int,
        dropout_rate: float,
        max_seq_length: int = 200,
    ):
        """
        Initialize the Decoder stack.

        Args:
        - `vocab_size (int)`: The size of the vocabulary.
        - `d_model (int)`: The dimensionality of the model (the size of the embedding vector).
        - `N (int)`: The number of decoder layers.
        - `h (int)`: The number of attention heads.
        - `dropout_rate (float)`: The dropout rate.
        - `max_seq_length (int)`: The maximum sequence length. Defaults to 200.

        Raises:
        - `ValueError`: If N < 1.
        """
        # Call the constructor of the parent class.
        super().__init__()

        # Define the attributes of the class.
        ## Check if N is valid.
        if N < 1:
            raise ValueError("N must be greater than or equal to 1.")

        ## Define the number of decoder layers.
        self.N = N

        ## Define the embedding layer.
        self.embed = layers.Embedder(vocab_size, d_model)

        ## Define the positional encoding layer.
        self.pe = layers.PositionalEncoder(d_model, max_seq_length, dropout_rate)

        ## Define the decoder layers.
        self.layers = get_clones(layers.Decoder(d_model, h, dropout_rate), N)

        ## Define the normalizer layer.
        self.norm = layers.Normalizer(d_model)

        ## Define the max sequence length.
        self.max_seq_length = max_seq_length

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_o: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        output_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Take a batch of indexed tokens and the encoding outputs, embed them, add positional encoding, then pass them through N decoder layers.

        Args:
            - `tgt (torch.Tensor)`: The batch of indexed tokens (in integer form).
            - `encoder_o (torch.Tensor)`: The output of the Encoder stack.
            - `src_mask (torch.Tensor)`: The padding mask.
            - `tgt_mask (torch.Tensor)`: The no-peeking mask.
            - `output_attention (bool)`: Whether to output the attention scores or not. Defaults to False.
        Returns:
            - `attention_o` (`torch.Tensor`): The output of the Decoder stack (the attention output).

            or

            - `(attention_o, attention_s_list)` (`Tuple[torch.Tensor, List[torch.Tensor]]`): The output of the Decoder stack and the list of attention scores.
            the decoded values [batch_size, tgt_len, d_model]

        Notes:
            - `tgt` has the shape [batch_size, tgt_len].
            - `encoder_o` has the shape [batch_size, src_len, d_model].
            - `src_mask` has the shape [batch_size, 1, src_len].
            - `tgt_mask` has the shape [batch_size, tgt_len, tgt_len].
            - `attention_o` has the shape [batch_size, tgt_len, d_model].
            - `attention_s_list` has the shape [N], each element has the shape [batch_size, heads, tgt_len, tgt_len/src_len].
        """
        # Embed the target sequence and add positional encoding.
        x: torch.Tensor = self.embed(tgt)
        x = self.pe(x)

        # Initialize the list of attention scores.
        attention_s_list: List[torch.Tensor] = [torch.Tensor(None)] * self.N

        # Pass the embedded sequence through N decoder layers.
        for i in range(self.N):
            x, attention_s = self.layers[i](x, encoder_o, src_mask, tgt_mask)
            attention_s_list[i] = attention_s

        # Normalize the output of the Decoder stack.
        attention_o: torch.Tensor = self.norm(x)

        # If output_attention is True, return the list of attention scores, too.
        if output_attention:
            return (attention_o, attention_s_list)
        else:
            return attention_o
