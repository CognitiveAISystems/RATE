from typing import NamedTuple
import torch

class MemoryState(NamedTuple):
    vec: torch.Tensor          # (B, M, D)
    pos: torch.LongTensor      # (B, M)