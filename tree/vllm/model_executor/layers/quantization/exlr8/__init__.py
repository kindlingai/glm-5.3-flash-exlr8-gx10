# SPDX-License-Identifier: Apache-2.0
"""exlr8 v2 releases (kindlingai/exlr8 FORMAT.md): routed and shared experts on the exlr8 MoE kernel.

Serve with `--quantization exlr8 --load-format exlr8`. VLLM_EXLR8_WIDTHS picks the expert widths: k2, k3 or k4 for
every unit (default k4), or the path of a v2 manifest. Dense matrices are decoded to BF16 by the exlr8 load format.

The shared expert runs inside the MoE kernel as one more route per token, so a model built with this config has no
separate shared-expert MLP (see Glm5NextMoE).
"""

import fcntl
import os
import re

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from . import release

logger = init_logger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "exlr8_moe.cu")
_KS_MAX = 4                     # K splits of a decode item, at most
# In the fused SP path, run the shared expert as two FP8 GEMMs on the gathered E4M3 rows instead of as a ninth route.
_SHARED_DENSE = os.environ.get("VLLM_EXLR8_SHARED_DENSE", "1") == "1"
_exts: dict[tuple, object] = {}
_scratch: dict[tuple, dict] = {}


class Exlr8Config(QuantizationConfig):
    """Expert widths and the layers exlr8 serves. Everything else in the model stays unquantized."""

    def __init__(self, widths: str | None = None) -> None:
        super().__init__()
        self.widths = widths or os.environ.get("VLLM_EXLR8_WIDTHS", "k4")

    def get_name(self) -> str:
        return "exlr8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "Exlr8Config":
        return cls(config.get("widths"))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        if isinstance(layer, RoutedExperts):
            return Exlr8MoEMethod(self, layer.moe_config, prefix)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


def _jit_load(name: str, src: str, flags: list[str]):
    """torch's cpp_extension.load() under an flock, which the kernel drops when its holder dies, so a build killed
    midway cannot leave torch's own lock file to hang later loads (the same guard as megamoe_vllm)."""
    from torch.utils.cpp_extension import _get_build_directory, load

    build = _get_build_directory(name, False)
    with open(os.path.join(build, "lock.flock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            os.remove(os.path.join(build, "lock"))
        except FileNotFoundError:
            pass
        return load(name, [src], extra_cuda_cflags=flags, extra_cflags=["-O3", "-DEXLR8_TORCH"])


def _ext(H: int, E: int, upr: int):
    key = (H, E, upr)
    if key not in _exts:
        flags = ["-O3", "-DEXLR8_TORCH", "-gencode=arch=compute_121a,code=sm_121a", f"-DEXLR8_H={H}",
                 f"-DEXLR8_E={E}", f"-DEXLR8_UPR={upr}"] + os.environ.get("VLLM_EXLR8_FLAGS", "").split()
        # The source and flags are in the name: torch reused a stale build of a changed kernel under one name.
        import hashlib

        h = hashlib.sha256(open(_SRC, "rb").read() + " ".join(flags).encode()).hexdigest()[:10]
        _exts[key] = _jit_load(f"exlr8_moe_{H}_{E}_{upr}_{h}", _SRC, flags)
    return _exts[key]


def _scratch_for(ext, device, H: int, I: int, mcap: int) -> tuple:
    """The key of the scratch shared by every MoE layer of a process (they run one after another). Methods keep the
    key, not the tensors, so the weight snapshot, which saves every tensor reachable from the model, skips them."""
    key = (device, H, I, mcap)
    s = _scratch.get(key)
    if s is None:
        ni, nf, ny, nh8 = ext.scratch_sizes(mcap, _KS_MAX)
        s = dict(si=torch.zeros(ni + 64, dtype=torch.int32, device=device),
                 sf=torch.zeros(nf, dtype=torch.float32, device=device),
                 xr=torch.empty(mcap, H, dtype=torch.float16, device=device),
                 xr8=torch.empty(mcap * H, dtype=torch.uint8, device=device),
                 xr8s=torch.empty(mcap, dtype=torch.float32, device=device),
                 h=torch.empty(mcap * (ext.TOPK + 1), I, dtype=torch.float16, device=device),
                 h8=torch.empty(max(nh8, 1), dtype=torch.uint8, device=device),
                 y=torch.empty(ny, dtype=torch.uint8, device=device),
                 # rows() leaves its output in y; the kernel still takes an out tensor, which it never writes there.
                 out0=torch.empty(1, H, dtype=torch.bfloat16, device=device))
        _scratch[key] = s
        mib = sum(t.numel() * t.element_size() for t in s.values()) / 2**20
        logger.info("exlr8: MoE scratch for %d tokens, %.0f MiB", mcap, mib)
    return key


class Exlr8MoEMethod(FusedMoEMethodBase):
    """A layer's routed experts and its shared expert, as stored, on this rank.

    The weights are four parameters (the trellis slab and its side data) whose sizes follow from the widths alone,
    so a dummy load and the weight snapshot build the same objects as a real load. The exlr8 load format fills them
    with load()."""

    def __init__(self, config: Exlr8Config, moe, prefix: str) -> None:
        super().__init__(moe)
        self.config = config
        m = re.search(r"layers\.(\d+)\.", prefix)
        if m is None:
            raise ValueError(f"exlr8: no layer index in {prefix!r}")
        self.layer_idx = int(m.group(1))

    def create_weights(self, layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        pc = self.moe.moe_parallel_config
        if pc.ep_size > 1 or num_experts != extra_weight_attrs.get("global_num_experts", num_experts):
            raise NotImplementedError("exlr8 MoE supports tensor parallelism only")
        from vllm.config import get_current_vllm_config

        tp, rank = pc.tp_size, pc.tp_rank
        # A TP=3 config pads the intermediate size (tp3pad); the release keeps the original.
        text = get_current_vllm_config().model_config.hf_text_config
        inter = (getattr(text, "tp_pad_orig", None) or {}).get("moe_intermediate_size", intermediate_size_per_partition * tp)
        if intermediate_size_per_partition % 128:
            raise ValueError(f"exlr8: {intermediate_size_per_partition} intermediate channels per rank, not 128-aligned")
        upr = intermediate_size_per_partition // 128
        geo = release.Geometry(hidden_size, inter, num_experts, 0, 0)
        plan = release.plan_moe(geo, self.layer_idx, tp, rank, release.Widths(self.config.widths), upr)
        E1, H = num_experts + 1, hidden_size

        def param(name, shape, dtype):
            p = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)

        param("exlr8_slab", (plan.nbytes,), torch.uint8)
        param("exlr8_signs", (2, H), torch.float16)
        param("exlr8_blk", (E1, 3, H // 128), torch.float32)
        param("exlr8_ch", (E1, 3, upr * 128), torch.float16)
        if _SHARED_DENSE:
            I = upr * 128
            param("exlr8_sh_gu", (2 * I, H), torch.float8_e4m3fn)
            param("exlr8_sh_gu_s", (1, 2 * I), torch.float32)
            param("exlr8_sh_d", (H, I), torch.float8_e4m3fn)
            param("exlr8_sh_d_s", (1, H), torch.float32)
        # Offsets and widths are fixed by the plan; they are not weights, so no loader or snapshot touches them.
        layer.exlr8_off = plan.off.to(layer.exlr8_slab.device)
        layer.exlr8_k = plan.k.to(layer.exlr8_slab.device)
        layer.exlr8_plan = plan
        layer.exlr8_geometry = (H, num_experts, upr)

    def load(self, layer, root: str) -> None:
        """Read this rank's units of the layer from the release into the parameters."""
        release.read_moe(root, self.layer_idx, layer.exlr8_plan, layer.exlr8_slab.data, layer.exlr8_signs.data,
                         layer.exlr8_blk.data, layer.exlr8_ch.data)
        if _SHARED_DENSE:
            from vllm import _custom_ops as ops

            from . import reference

            H, E, upr = layer.exlr8_geometry
            gu, down = reference.shared_dense_weights(layer.exlr8_slab.data, layer.exlr8_off, layer.exlr8_k,
                                                      layer.exlr8_blk.data, layer.exlr8_ch.data,
                                                      layer.exlr8_signs.data, H, E, upr)
            for w, q, qs in ((gu, layer.exlr8_sh_gu, layer.exlr8_sh_gu_s), (down, layer.exlr8_sh_d, layer.exlr8_sh_d_s)):
                wq, ws = ops.scaled_fp8_quant(w.to(torch.bfloat16).contiguous(), use_per_token_if_dynamic=True)
                q.data.copy_(wq)
                qs.data.copy_(ws.view(1, -1).float())

    def process_weights_after_loading(self, layer) -> None:
        from vllm.config import get_current_vllm_config

        H, E, upr = layer.exlr8_geometry
        self._ext = _ext(H, E, upr)
        mcap = max(1, get_current_vllm_config().scheduler_config.max_num_batched_tokens)
        self._mcap = mcap
        self._scratch_key = _scratch_for(self._ext, layer.exlr8_slab.device, H, upr * 128, mcap)
        limit = getattr(layer, "swiglu_limit", None)
        self._limit = float("inf") if limit is None else float(limit)

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
              shared_experts, shared_experts_input) -> torch.Tensor:
        if shared_experts is not None:
            raise RuntimeError("exlr8 runs the shared expert inside its MoE kernel; the model must not build one")
        H = layer.exlr8_signs.shape[1]
        M = x.shape[0]
        out = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
        if M == 0:
            return out
        x = x.to(torch.bfloat16).contiguous()
        ids = topk_ids.to(torch.int64).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        s = _scratch[self._scratch_key]
        for a in range(0, M, self._mcap):
            b = min(M, a + self._mcap)
            self._ext.moe(x[a:b], ids[a:b], w[a:b], True, 1.0, self._limit, layer.exlr8_slab, layer.exlr8_off,
                          layer.exlr8_k, layer.exlr8_signs, layer.exlr8_blk, layer.exlr8_ch, s["si"], s["sf"],
                          s["xr"], s["xr8"], s["xr8s"], s["h"], s["h8"], s["y"], out[a:b], _KS_MAX, self._mcap,
                          False, False)
        return out

    # The fused sequence-parallel path (Glm5NextDecoderLayer._sp_moe_fused_exlr8): each rank rotates its shard to
    # E4M3, the shards are gathered, the MoE runs on the gathered rows and leaves one rotated int8 row per route,
    # which arxbig's moe_finalize_rs sums into the reduce-scatter.
    def rotate8(self, layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_rot of this rank's shard as E4M3 [n, H] and its per-token scales [n]."""
        n, H = x.shape
        xr8 = torch.empty(n, H, dtype=torch.uint8, device=x.device)
        xr8s = torch.empty(n, dtype=torch.float32, device=x.device)
        self._ext.rot8(x.to(torch.bfloat16).contiguous(), layer.exlr8_signs, xr8, xr8s)
        return xr8, xr8s

    @property
    def shared_dense(self) -> bool:
        return _SHARED_DENSE

    def shared(self, layer, xr8: torch.Tensor, xr8s: torch.Tensor) -> torch.Tensor:
        """The shared expert on gathered rotated rows, bf16 [M, H] in the hidden space (shared_dense only)."""
        from vllm import _custom_ops as ops

        a = xr8.view(torch.float8_e4m3fn)
        gu = ops.cutlass_scaled_mm(a, layer.exlr8_sh_gu.t(), xr8s.view(-1, 1), layer.exlr8_sh_gu_s, torch.bfloat16)
        act = torch.empty(gu.shape[0], gu.shape[1] // 2, dtype=torch.bfloat16, device=gu.device)
        torch.ops._C.silu_and_mul_with_clamp(act, gu, self._limit, 1.0, 0.0)
        hq, hs = ops.scaled_fp8_quant(act, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(hq, layer.exlr8_sh_d.t(), hs, layer.exlr8_sh_d_s, torch.bfloat16)

    def rows(self, layer, xr8: torch.Tensor, xr8s: torch.Tensor, topk_ids: torch.Tensor,
             topk_weights: torch.Tensor, shared: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The MoE on gathered rotated rows: (int8 rows [M * R, H], their scales [M * R, H / 128], R), route
        r = token * R + slot, routing weight and output rotation already applied. shared: the shared expert is
        route R - 1."""
        M, H = xr8.shape
        if M > self._mcap:
            raise ValueError(f"exlr8: SP MoE call of {M} tokens, more than the {self._mcap}-token scratch")
        s = _scratch[self._scratch_key]
        R = self._ext.TOPK + int(shared)
        ids = topk_ids.to(torch.int64).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        self._ext.moe(xr8, ids, w, shared, 1.0, self._limit, layer.exlr8_slab, layer.exlr8_off, layer.exlr8_k,
                      layer.exlr8_signs, layer.exlr8_blk, layer.exlr8_ch, s["si"], s["sf"], s["xr"], xr8.view(-1),
                      xr8s, s["h"], s["h8"], s["y"], s["out0"], _KS_MAX, self._mcap, True, True)
        y8 = s["y"][: M * R * H].view(M * R, H)
        y8s = s["y"][M * R * H: M * R * H + M * R * (H // 128) * 4].view(torch.float32).view(M * R, H // 128)
        return y8, y8s, R
