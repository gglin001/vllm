# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils import cdiv, round_up
from vllm.model_executor.layers.fused_moe.ubatch_context import UBContext, UBStage


def pplx_hidden_dim_scale_bytes(
    max_num_tokens: int,
    hidden_dim: int,
    in_dtype: torch.dtype,
    quant_dtype: Optional[torch.dtype],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]],
):
    # All pplx byte sizes must be 16-byte aligned.
    align = 16

    # For blocked per token: set to
    #   ceil_div(hidden_dim, block_size) * sizeof(float32)
    # For per-token: set to 4 * sizeof(float32) (x4 for alignment)
    if quant_dtype is not None:
        assert quant_dtype.itemsize == 1
        hidden_dim_bytes = hidden_dim * quant_dtype.itemsize
        elem_size = torch.float32.itemsize

        if per_act_token_quant:
            # per-token
            assert block_shape is None
            hidden_scale_bytes = elem_size
        elif block_shape is not None:
            # per-group
            block_size = block_shape[1]
            num_blocks = cdiv(hidden_dim, block_size)
            hidden_scale_bytes = num_blocks * elem_size
        else:
            # per-tensor
            hidden_scale_bytes = elem_size
    else:
        hidden_dim_bytes = hidden_dim * in_dtype.itemsize
        hidden_scale_bytes = 0

    return (
        round_up(hidden_dim_bytes, align),
        round_up(hidden_scale_bytes, align),
    )


# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        a2a: pplx.AllToAll,
        max_num_tokens: int,
        world_size: int,
        rank: int,
        dp_size: int,
    ):
        super().__init__()
        assert max_num_tokens > 0
        self.a2a = a2a
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

        self.ubatch_ctxs = [UBContext() for _ in range(2 + 1)]

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.uint32

    def prepare_a(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ) -> UBContext:
        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert topk_ids.size(0) == num_tokens
        # assert expert_map is None, "NYI"

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        repeat_cols = 4
        repeat_rows = 1 if quant_config.per_act_token_quant else a1.size(0)
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1, (None if quant_config.per_act_token_quant else a1_scale),
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape)

        if a1q_scale is not None:
            if a1q_scale.numel() == 1:
                orig_a_scale_block_shape = 1
            else:
                orig_a_scale_block_shape = a1q_scale.shape[-1]
            a1q_scale = a1q_scale.repeat(repeat_rows, repeat_cols)

        # rem_experts need to be 0 for pplx to work properly.
        rem_experts = num_experts % self.world_size
        assert rem_experts == 0
        num_local_experts = ((num_experts // self.world_size) +
                             (1 if self.rank < rem_experts else 0))

        expert_num_tokens = torch.empty(
            num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        num_dp = self.world_size // self.dp_size
        expert_x = torch.empty(
            (num_local_experts, self.max_num_tokens * num_dp, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            block_size = (quant_config.block_shape[1]
                          if quant_config.block_shape is not None else 1)
            expert_x_scale = torch.empty(
                (num_local_experts, expert_x.size(1),
                 round_up(
                     (expert_x.size(2) + block_size - 1) // block_size, 4)),
                dtype=torch.float32,
                device=device,
            )

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
            #
            do_send=True,
            do_recv=False,
            #
        )

        ubatch_ctx = self.ubatch_ctxs[ubatch_slice]
        ubatch_ctx.expert_num_tokens = expert_num_tokens
        ubatch_ctx.expert_x = expert_x
        ubatch_ctx.expert_x_scale = expert_x_scale
        ubatch_ctx.a1q = a1q
        ubatch_ctx.a1q_scale = a1q_scale
        ubatch_ctx.rank_topk_ids = rank_topk_ids
        ubatch_ctx.bound_m = bound_m
        return ubatch_ctx

    def prepare_b(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        ubatch_ctx = self.ubatch_ctxs[ubatch_slice]
        expert_num_tokens = ubatch_ctx.expert_num_tokens
        expert_x = ubatch_ctx.expert_x
        expert_x_scale = ubatch_ctx.expert_x_scale

        self.a2a.dispatch(
            out_expert_num_tokens=ubatch_ctx.expert_num_tokens,
            out_expert_x=ubatch_ctx.expert_x,
            out_expert_x_scale=ubatch_ctx.expert_x_scale,
            dp_x=ubatch_ctx.a1q,
            dp_x_scale=ubatch_ctx.a1q_scale,
            indices=ubatch_ctx.rank_topk_ids,
            bound_m=ubatch_ctx.bound_m,
            #
            do_send=False,
            do_recv=True,
            #
        )
        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, :orig_a_scale_block_shape]

        return expert_x, expert_x_scale, expert_num_tokens, None, None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ):
        _ = self.prepare_a(
            a1,
            a1_scale,
            a2_scale,
            rank_topk_weights,
            rank_topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            #
            ubatch_stage=UBStage.dispatch_a.value,
            ubatch_slice=ubatch_slice,
            #
        )
        return self.prepare_b(
            a1,
            a1_scale,
            a2_scale,
            rank_topk_weights,
            rank_topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            #
            ubatch_stage=UBStage.dispatch_b.value,
            ubatch_slice=ubatch_slice,
            #
        )

    def finalize_a(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ) -> None:
        num_tokens = output.size(0)  # M
        # This argument is optional
        # There's not much point setting this unless it is != topk_ids.size(0)
        bound_m: Optional[torch.Tensor] = None

        assert topk_ids.size(0) == num_tokens, (
            f"{topk_ids.size(0)} == {num_tokens}")
        assert output.size(0) <= self.max_num_tokens, (
            f"{output.size(0)} <= {self.max_num_tokens}")
        assert output.size(1) == fused_expert_output.size(-1)

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        self.a2a.combine(
            out_tokens=output,
            indices=topk_ids,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
            #
            do_send=True,
            do_recv=False,
            #
        )

        ubatch_ctx = self.ubatch_ctxs[ubatch_slice]
        ubatch_ctx.bound_m = bound_m
        ubatch_ctx.topk_weights = topk_weights

        return ubatch_ctx

    def finalize_b(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ) -> None:

        ubatch_ctx = self.ubatch_ctxs[ubatch_slice]
        bound_m = ubatch_ctx.bound_m
        topk_weights = ubatch_ctx.topk_weights

        self.a2a.combine(
            out_tokens=output,
            indices=topk_ids,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
            #
            do_send=False,
            do_recv=True,
            #
        )

        return output

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        #
        ubatch_stage: int = UBStage.nop.value,
        ubatch_slice: int = -1,
        #
    ) -> None:
        _ = self.finalize_a(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            #
            ubatch_stage=UBStage.combine_a.value,
            ubatch_slice=ubatch_slice,
            #
        )
        return self.finalize_b(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            #
            ubatch_stage=UBStage.combine_b.value,
            ubatch_slice=ubatch_slice,
            #
        )
