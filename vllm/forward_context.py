# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, TypeAlias

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

UbatchSlice: TypeAlias = tuple[slice, slice]
UBatchSlices: TypeAlias = list[UbatchSlice]

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor
    cu_tokens_across_dp_cpu: torch.Tensor
    num_tokens_tensor: torch.Tensor
    support_cg: bool = False
    support_ubatch: bool = False

    @staticmethod
    def num_tokens_across_dp(num_tokens: int, dp_size: int,
                             dp_rank: int) -> torch.Tensor:
        """
        Gather the num_tokens across all DP ranks and return results in a
        CPU tensor of size dp_size.
        """
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        return num_tokens_tensor

    @staticmethod
    def make(
            parallel_config: ParallelConfig,
            attn_metadata: Any,
            num_tokens: int,
            num_tokens_across_dp: Optional[torch.Tensor] = None
    ) -> "DPMetadata":

        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = attn_metadata.num_prefill_tokens + \
                attn_metadata.num_decode_tokens
        else:
            # for v1 attention backends or no attn_metadata
            batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert (num_tokens_across_dp is None
                or num_tokens_across_dp[dp_rank] == batchsize)
        if num_tokens_across_dp is None:
            num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
                batchsize, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_across_dp, dim=0)
        return DPMetadata(max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu)

    @staticmethod
    def make_ubatch(
        parallel_config: ParallelConfig,
        attn_metadata: Any,
        num_tokens: int,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        support_cg: bool = False,
        support_ubatch: bool = False,
    ) -> "DPMetadata":
        assert parallel_config.data_parallel_size > 1
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        assert num_tokens_across_dp is None
        num_tokens_across_dp = [0] * (dp_size + 2)
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_across_dp[-2] = int(support_cg)
        num_tokens_across_dp[-1] = int(support_ubatch)
        num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                         device="cpu",
                                         dtype=torch.int32)
        from vllm.distributed.parallel_state import get_dp_group
        dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        max_tokens_across_dp_cpu = torch.max(num_tokens_tensor[:-2])
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor[:-2], dim=0)
        support_cg_all = num_tokens_tensor[-2].sum().item() == dp_size
        support_ubatch_all = num_tokens_tensor[-1].sum().item() == dp_size
        return DPMetadata(max_tokens_across_dp_cpu, cu_tokens_across_dp_cpu,
                          num_tokens_tensor, support_cg_all,
                          support_ubatch_all)


@dataclass
class UBMetadata:
    ubatch_slices: Optional[UBatchSlices] = None
    ubatch_index: int = -1

    @classmethod
    @contextmanager
    def set_ubatch_index(cls, ubatch_index: int):
        forward_context = get_forward_context()
        assert forward_context.ub_metadata is not None
        prev_ubatch_index = forward_context.ub_metadata.ubatch_index
        forward_context.ub_metadata.ubatch_index = ubatch_index
        yield
        forward_context.ub_metadata.ubatch_index = prev_ubatch_index

    def to_tensor(self, device) -> torch.Tensor:
        if self.ubatch_slices is None:
            return None
        assert len(self.ubatch_slices) == 2
        # TODO: .cuda()
        meta_tensor = torch.zeros(3, 4, dtype=torch.int32, device=device)
        # ubatch_slice_0
        meta_tensor[0][0] = self.ubatch_slices[0][0].start
        meta_tensor[0][1] = self.ubatch_slices[0][0].stop
        meta_tensor[0][2] = self.ubatch_slices[0][1].start
        meta_tensor[0][3] = self.ubatch_slices[0][1].stop
        # ubatch_slice_1
        meta_tensor[1][0] = self.ubatch_slices[1][0].start
        meta_tensor[1][1] = self.ubatch_slices[1][0].stop
        meta_tensor[1][2] = self.ubatch_slices[1][1].start
        meta_tensor[1][3] = self.ubatch_slices[1][1].stop
        # ubatch_index
        meta_tensor[2][0] = self.ubatch_index
        return meta_tensor


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    """
    Type AttentionMetadata for v0, 
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each 
    attention layer to its attention metadata
    set dynamically for each forward pass
    """
    attn_metadata: Union["AttentionMetadata", dict[str, "AttentionMetadata"]]
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None
    skip_cuda_graphs: bool = False
    # ubatch metadata
    ub_metadata: Optional[UBMetadata] = None


_forward_context: Optional[ForwardContext] = None


@torch.compiler.disable
def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


# @torch.library.custom_op("vllm::get_forward_context", mutates_args=("x", ))
# def yield_and_switch_from_compute_to_comm(x: torch.Tensor) -> ForwardContext:
#     return get_forward_context()


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: Optional[int] = None,
    num_tokens_across_dp: Optional[torch.Tensor] = None,
    skip_cuda_graphs: bool = False,
    ub_metadata: Optional[UBMetadata] = None,
    dp_metadata: Optional[DPMetadata] = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    assert dp_metadata is not None
    """
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
            attn_metadata is not None or num_tokens is not None):
        dp_metadata = DPMetadata.make(vllm_config.parallel_config,
                                      attn_metadata, num_tokens or 0,
                                      num_tokens_across_dp)
    """

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        skip_cuda_graphs=skip_cuda_graphs,
        ub_metadata=ub_metadata,
    )

    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            else:
                # for v1 attention backends
                batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform
            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)

        _forward_context = prev_context
