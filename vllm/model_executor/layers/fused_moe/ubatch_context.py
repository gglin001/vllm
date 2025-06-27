from enum import Enum, auto
from dataclasses import dataclass  # noqa: E402

import torch

# @dataclass
# class UBContext:
#     pass


class UBContext(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


# TODO: support decoding  stages
class UBStage(Enum):
    nop = auto()
    attn = auto()
    dispatch = auto()  # prepare
    dispatch_a = auto()
    dispatch_b = auto()
    mlp = auto()  # fused_experts
    combine = auto()  # finalize
    combine_a = auto()
    combine_b = auto()
    shared = auto()
