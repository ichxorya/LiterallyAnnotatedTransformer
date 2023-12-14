"""
Written by Nghia Be.

This file contains the DirtyTransformer class, which is the Transformer model.

"Dirty" means that this class is a quick-and-dirty implementation of the Transformer model.

Checklist:
- Documentation: Fully documented.
- Type-checking: Passed (Mypy strict mode).
"""
# Import necessary modules and packages.
from __future__ import annotations
from . import layers as layer
import torch
import torch.nn as nn


# Define the DirtyTransformer class.
class DirtyTransformer(nn.Module):
    """
    This class implements the Transformer model.
    ELI5: Transformer ~ Encoder stack + Decoder stack.

    Subclasses:
    - `Encoder(nn.Module)`: Encoder stack.
    - `Decoder(nn.Module)`: Decoder stack.

    Attributes:
    """

    class Encoder(nn.Module):
        """"""
