# SPDX-License-Identifier: Apache-2.0
# Reference decode of exlr8 v2 weights, in plain PyTorch, for checking the kernels.
#
# From kindlingai/exlr8 loader/exlr8.py, generalised to any hidden size. That loader is a derivative of exllamav3 by
# turboderp (https://github.com/turboderp-org/exllamav3): the trellis codebook, bit packing, tile order and the
# 128x128 Hadamard rotation reproduce exllamav3 0.0.43. MIT licence, Copyright (c) 2025 Turboderp.
"""fp32 reference decode of exlr8 v2 tiles, units and dense records."""

import math

import torch

MCG = 0xCBAC1FED
TRELLIS, INT4, MXFP8, FP8_BLOCK, FP8_ROW, BF16, FP32 = 1, 2, 3, 4, 5, 6, 7
SIDE_IN = 2
DATA = {TRELLIS: ("trellis", "suh", "svh"), INT4: ("q4", "s4"), MXFP8: ("fp8", "e8m0"), FP8_BLOCK: ("fp8", "scale"),
        FP8_ROW: ("fp8", "scale"), BF16: ("bf16",), FP32: ("fp32",)}


def tile_perm() -> torch.Tensor:
    """perm[j] = row * 16 + col of the tile element at stream position j. Rows are the input side."""
    perm = []
    for t in range(32):
        r0, c0 = 2 * (t % 4), t // 4
        for r, c in ((r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                     (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)):
            perm.append(16 * r + c)
    return torch.tensor(perm)


def codebook(states: torch.Tensor) -> torch.Tensor:
    """16-bit states -> fp32: the mcg codebook, then E4M3 rounding (nearest even, saturating), which the format
    requires: the encoder searched over the rounded values."""
    x = ((states & 0xFFFF) * MCG) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60

    def fp16(v):
        return ((v ^ 0x8000) - 0x8000).to(torch.int16).view(torch.float16)
    v = (fp16(x & 0xFFFF) + fp16(x >> 16)).float()
    return v.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def decode_tiles(raw: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [n, 32K] -> fp32 [n, 16, 16], (input, output) within each tile."""
    n = raw.shape[0]
    b = raw.reshape(n, 16 * K, 2).long()
    words = (b[..., 0] | (b[..., 1] << 8)).reshape(n, 8 * K, 2).flip(-1).reshape(n, 16 * K)
    bits = (words.unsqueeze(-1) >> torch.arange(15, -1, -1, device=raw.device)) & 1
    vals = (bits.reshape(n, 256, K) << torch.arange(K - 1, -1, -1, device=raw.device)).sum(-1)
    states = torch.zeros_like(vals)
    for j in range(-(-16 // K)):
        states |= torch.roll(vals, j, dims=1) << (j * K)
    out = torch.empty(n, 256, dtype=torch.float32, device=raw.device)
    out[:, tile_perm().to(raw.device)] = codebook(states)
    return out.view(n, 16, 16)


def tiles_out_major(raw: torch.Tensor, K: int, rows: int, cols: int) -> torch.Tensor:
    """Tiles stored [k16][c] (gate/up units, output-sharded dense entries) -> (rows, cols)."""
    return decode_tiles(raw.reshape(-1, 32 * K), K).view(rows // 16, cols // 16, 16, 16) \
        .permute(0, 2, 1, 3).reshape(rows, cols)


def tiles_in_major(raw: torch.Tensor, K: int, cols: int) -> torch.Tensor:
    """Tiles stored [b][r][c] (down units, input-sharded dense entries) -> (128, cols)."""
    return decode_tiles(raw.reshape(-1, 32 * K), K).view(cols // 128, 8, 8, 16, 16) \
        .permute(1, 3, 0, 2, 4).reshape(128, cols)


def hadamard(n: int = 128, device="cpu") -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * (1 / math.sqrt(n))).to(device)  # fp32: callers may run under a bf16 default dtype


def reconstruct(T: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """W (in, out) = diag(suh) Hb T Hb diag(svh), Hb block-diagonal with 128x128 blocks."""
    k, n = T.shape
    had = hadamard(128, T.device)
    x = (had @ T.view(-1, 128, n)).view(k, n) * suh.float().unsqueeze(1)
    return (x.view(k, -1, 128) @ had).view(k, n) * svh.float().unsqueeze(0)


def expert_unit(raw: torch.Tensor, proj: str, K: int, H: int, chan: torch.Tensor, block: torch.Tensor,
                signs: torch.Tensor) -> torch.Tensor:
    """One 128-channel unit of an expert projection as W (in, out) fp32.

    gate/up: (H, 128), chan = the unit's svh, block = su_block, signs = signs.in.
    down: (128, H), chan = the unit's suh, block = sv_block, signs = signs.out."""
    hside = signs.float() * block.float().repeat_interleave(128)
    if proj == "down":
        return reconstruct(tiles_in_major(raw, K, H), chan, hside)
    return reconstruct(tiles_out_major(raw, K, H, 128), hside, chan)


def dense(rec, t: dict) -> torch.Tensor:
    """One dense record as an HF weight (out, in) fp32. t: {suffix: tensor} as stored."""
    fmt, out, inp, K, side = (int(x) for x in rec[:5])
    if fmt == TRELLIS:
        ent = t["trellis"]
        if side == SIDE_IN:
            T = torch.cat([tiles_in_major(ent[g], K, out) for g in range(ent.shape[0])], 0)
        else:
            T = torch.cat([tiles_out_major(ent[g], K, inp, 128) for g in range(ent.shape[0])], 1)
        return reconstruct(T, t["suh"], t["svh"]).T
    if fmt == INT4:
        q4, s4 = t["q4"], t["s4"]
        if side == SIDE_IN:
            q4, s4 = q4.transpose(0, 1).reshape(out, -1), s4.transpose(0, 1).reshape(out, -1)
        n = torch.stack([(q4 & 15).int() - 8, (q4 >> 4).int() - 8], -1).reshape(out, inp).float()
        return n * s4.float().repeat_interleave(32, 1)
    w = t["fp8"].view(torch.float8_e4m3fn).float() if "fp8" in t else None
    if fmt == MXFP8:
        e = t["e8m0"].int() - 127
        return w * torch.ldexp(torch.ones_like(e, dtype=torch.float32), e).repeat_interleave(32, 1)
    if fmt == FP8_BLOCK:
        return w * t["scale"].repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inp]
    if fmt == FP8_ROW:
        return w * t["scale"].unsqueeze(1)
    return t["bf16" if fmt == BF16 else "fp32"].float()


def fp8_pos(k: torch.Tensor) -> torch.Tensor:
    """Position of physical channel k in the kernel's E4M3 row order (exlr8_moe.cu fp8_pos)."""
    return (k & ~31) + (k & 16) + 4 * ((k & 7) >> 1) + 2 * ((k >> 3) & 1) + (k & 1)


def shared_dense_weights(slab, off, ks, blk, ch, signs, H: int, E: int, upr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's shared expert as two dense fp32 weights that take the MoE's rotated E4M3 rows directly.

    gate_up [2 I, H]: x_rot (fp8_pos order) @ gate_up.T = x @ (W_gate | W_up). x_rot = (x * signs.in) Hb blockwise
    and the gate/up input scale is signs.in x a per-block scale, so W' = diag(block scale) T Hb diag(svh); its input
    rows are permuted to fp8_pos order like the rows it meets. down [H, I]: the plain down weight, output signs and
    Hadamard included, so the product lands in the hidden space the routed rows are summed in."""
    had = hadamard(128, slab.device)
    gu = []
    for p in (0, 1):
        K = int(ks[E, p])
        hside = blk[E, p].float().repeat_interleave(128)
        cols = []
        for j in range(upr):
            o = int(off[E, p, j])
            T = tiles_out_major(slab[o:o + H * 16 * K], K, H, 128)
            cols.append(hside[:, None] * (T @ had) * ch[E, p, 128 * j:128 * j + 128].float()[None, :])
        gu.append(torch.cat(cols, 1))
    w = torch.cat(gu, 1)                                   # (H, 2 I), rows = physical input channels
    wp = torch.empty_like(w)
    wp[fp8_pos(torch.arange(H, device=w.device))] = w
    K = int(ks[E, 2])
    svh = signs[1].float() * blk[E, 2].float().repeat_interleave(128)
    rows = []
    for j in range(upr):
        o = int(off[E, 2, j])
        T = tiles_in_major(slab[o:o + H * 16 * K], K, H)
        rows.append(reconstruct(T, ch[E, 2, 128 * j:128 * j + 128], svh))
    return wp.T.contiguous(), torch.cat(rows, 0).T.contiguous()
