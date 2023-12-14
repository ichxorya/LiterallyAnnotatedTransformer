from layers import (
    Embedder,
    PositionalEncoder,
    FeedForward,
    Normalizer,
    MultiHeadAttention,
)
import torch
import torch.nn as nn

# Test
ttt_0 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"ttt_0: {ttt_0}")

embed = Embedder(12, 20)
print(f"embed: {embed}")

ttt_1 = embed(ttt_0)
print(f"ttt_1: {ttt_1}")

pos_enc = PositionalEncoder(12, 12, 0.1)
print(f"pos_enc: {pos_enc}")

ttt_2 = pos_enc(ttt_1)
print(f"ttt_2: {ttt_2}")

mha = MultiHeadAttention(6, 12, 0.1)
print(mha)

ttt_3 = mha(ttt_2, ttt_2, ttt_2)
print(f"ttt_3: {ttt_3}")
