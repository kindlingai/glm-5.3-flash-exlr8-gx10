# SPDX-License-Identifier: Apache-2.0
"""Reading an exlr8 v2 release (kindlingai/exlr8 FORMAT.md) as stored.

A rank reads only the expert units it owns into one device slab per MoE layer. The bytes are not transformed: the
kernels read the trellis tiles in the file's order.
"""

import json
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

PROJS = ("gate", "up", "down")
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "U64": torch.int64}


class SafeFile:
    """Named tensors and raw byte ranges of one safetensors file. U64 reads as int64."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.meta = self.header.pop("__metadata__", None) or {}
        self.start = 8 + n

    def names(self) -> list[str]:
        return [k for k in self.header if not k.startswith("__pad.")]

    def span(self, name: str) -> tuple[int, int]:
        """(absolute file offset, bytes) of a tensor's data."""
        a, b = self.header[name]["data_offsets"]
        return self.start + a, b - a

    def get(self, name: str) -> torch.Tensor:
        info = self.header[name]
        off, n = self.span(name)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            buf = os.pread(fd, n, off)
        finally:
            os.close(fd)
        if len(buf) != n:
            raise OSError(f"{self.path}: short read of {name}")
        return torch.frombuffer(bytearray(buf), dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])


@dataclass(frozen=True)
class Geometry:
    hidden: int
    inter: int
    experts: int
    layers: int
    first_moe: int

    @property
    def units(self) -> int:
        return self.inter // 128

    @staticmethod
    def from_meta(meta: dict) -> "Geometry":
        return Geometry(int(meta["hidden_size"]), int(meta["moe_intermediate_size"]),
                        int(meta.get("routed_experts", meta.get("n_routed_experts"))),
                        int(meta["num_hidden_layers"]), int(meta["first_moe_layer"]))


def owned_units(geo: Geometry, layer: int, tp: int, rank: int) -> list[list[int]]:
    """Per expert (index E = the shared expert), the units this rank owns, ascending.

    Unit u of expert e of layer L goes to rank (U*e + u + L) mod tp; the shared expert follows the routed ones."""
    U = geo.units
    return [[u for u in range(U) if (U * e + u + layer) % tp == rank] for e in range(geo.experts + 1)]


class Widths:
    """Width per (layer, expert, projection, unit): one K everywhere ("k4"), or a v2 manifest file."""

    def __init__(self, spec: str):
        self.k = None
        self.layers = None
        if spec.startswith("k") and spec[1:].isdigit():
            self.k = int(spec[1:])
        else:
            with open(spec) as f:
                m = json.load(f)
            if m.get("format") != "exlr8-v2-manifest":
                raise ValueError(f"{spec}: not an exlr8 v2 manifest")
            self.layers = m["layers"]

    def get(self, layer: int, e: int, shared: bool, proj: str, u: int) -> int:
        if self.k is not None:
            return self.k
        m = self.layers[str(layer)]
        if shared:
            s = m["shared"][proj]
            return int(s if proj == "down" else s[u])
        return int(m[proj][e] if proj == "down" else m[proj][e][u])


class ExpertLayer:
    """One MoE layer: its index file and the per-width trellis files."""

    def __init__(self, root: str, layer: int):
        self.layer = layer
        base = os.path.join(root, "experts", f"exlr8-layer-{layer:03d}")
        self.index = SafeFile(base + ".index.safetensors")
        self.geo = Geometry.from_meta(self.index.meta)
        self.widths = [int(k) for k in self.index.meta["widths"].split(",")]
        self.shared_widths = [int(k) for k in self.index.meta["shared_widths"].split(",")]
        self.base = base
        self._t: dict[str, torch.Tensor] = {}
        self._files: dict[int, SafeFile] = {}

    def t(self, name: str) -> torch.Tensor:
        if name not in self._t:
            self._t[name] = self.index.get(name)
        return self._t[name]

    def trellis_file(self, K: int) -> SafeFile:
        if K not in self._files:
            self._files[K] = SafeFile(f"{self.base}.k{K}.safetensors")
        return self._files[K]

    def unit(self, e: int, proj: str, u: int, K: int) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        """(file offset, bytes, 128 channel values, H/128 block scales) of unit u of a projection at width K.
        e = experts means the shared expert. Index offsets are absolute file offsets."""
        shared = e == self.geo.experts
        if K not in (self.shared_widths if shared else self.widths):
            raise ValueError(f"layer {self.layer} expert {e} {proj}: K{K} is not stored")
        pre, j = (f"shared.k{K}.", 0) if shared else (f"k{K}.", e)
        off, n = (int(v) for v in self.t(pre + "index")[j, PROJS.index(proj), u])
        c = slice(128 * u, 128 * u + 128)
        if proj == "down":
            chan, block = self.t(pre + "down.suh")[j, c], self.t(pre + "down.sv_block")[j]
        else:
            chan, block = self.t(pre + f"{proj}.svh")[j, c], self.t(pre + f"{proj}.su_block")[j]
        return off, n, chan, block


def read_ranges(ranges: list[tuple[str, int, int, int]], dst: torch.Tensor, chunk: int = 256 << 20) -> None:
    """Copy file byte ranges (path, offset, bytes, dst offset) into a uint8 device tensor through one pinned
    staging buffer. Ranges are read in file order; consecutive ranges of one file coalesce into one read."""
    ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
    stage = torch.empty(chunk, dtype=torch.uint8, pin_memory=dst.is_cuda)
    view = memoryview(stage.numpy())
    fds: dict[str, int] = {}
    try:
        i = 0
        while i < len(ranges):
            path, off, n, _ = ranges[i]
            # One read covers ranges i..j-1 when they are back to back in the file and fit the buffer.
            j, end = i + 1, off + n
            while j < len(ranges) and ranges[j][0] == path and ranges[j][1] == end and end + ranges[j][2] - off <= chunk:
                end += ranges[j][2]
                j += 1
            if end - off > chunk:
                raise ValueError(f"range of {end - off} bytes is larger than the staging buffer")
            fd = fds.get(path)
            if fd is None:
                fd = fds[path] = os.open(path, os.O_RDONLY)
            got = os.preadv(fd, [view[:end - off]], off)
            if got != end - off:
                raise OSError(f"{path}: short read at {off}")
            for k in range(i, j):
                _, o, m, d = ranges[k]
                dst[d:d + m].copy_(stage[o - off:o - off + m], non_blocking=False)
            i = j
    finally:
        for fd in fds.values():
            os.close(fd)


@dataclass
class MoePlan:
    """Where one rank's units of one MoE layer go in its slab, from the widths alone (no file reads).

    units[e]: owned units of expert e (index E = the shared expert), ascending. off[e, p, j]: byte offset of unit j of
    projection p (gate, up, down) in the slab. k[e, p]: the projection's width; gate and up match.

    When TP does not divide the 16 units (TP=3: 5 or 6 per expert), every expert gets upr slots; a slot past its
    expert's units reads the expert's first unit and gets zero channel values (read_moe), so its h is exactly zero and
    its down projection adds nothing."""
    units: list[list[int]]
    off: torch.Tensor
    k: torch.Tensor
    nbytes: int


def plan_moe(geo: Geometry, layer: int, tp: int, rank: int, widths: Widths, upr: int | None = None) -> MoePlan:
    units = owned_units(geo, layer, tp, rank)
    most = max(len(u) for u in units)
    upr = most if upr is None else upr
    if upr < most:
        raise ValueError(f"TP={tp} rank {rank}: an expert owns {most} units, more than the {upr} slots")
    E1 = geo.experts + 1
    off = torch.zeros(E1, 3, upr, dtype=torch.int64)
    ks = torch.zeros(E1, 3, dtype=torch.int32)
    pos = 0
    for e in range(E1):
        shared = e == geo.experts
        for p, proj in enumerate(PROJS):
            kset = {widths.get(layer, e, shared, proj, u) for u in units[e]}
            if len(kset) != 1:
                raise ValueError(f"layer {layer} expert {e} {proj}: owned units take widths {sorted(kset)}")
            K = kset.pop()
            if p == 1 and K != int(ks[e, 0]):
                raise ValueError(f"layer {layer} expert {e}: gate K{int(ks[e, 0])} != up K{K}")
            ks[e, p] = K
            for j in range(len(units[e])):
                off[e, p, j] = pos
                pos += geo.hidden * 16 * K
            off[e, p, len(units[e]):] = off[e, p, 0]
    return MoePlan(units, off, ks, pos)


def read_moe(root: str, layer: int, plan: MoePlan, slab: torch.Tensor, signs: torch.Tensor, blk: torch.Tensor,
             ch: torch.Tensor) -> None:
    """Fill a planned slab and its side data from the release: slab uint8 [nbytes], signs f16 [2, H],
    blk f32 [E + 1, 3, H / 128] (signed hidden-side block scales), ch f16 [E + 1, 3, UPR * 128]."""
    x = ExpertLayer(root, layer)
    ranges = []
    blk_c = torch.empty(blk.shape, dtype=torch.float32)
    ch_c = torch.zeros(ch.shape, dtype=torch.float16)
    for e, us in enumerate(plan.units):
        for p, proj in enumerate(PROJS):
            K = int(plan.k[e, p])
            for j, u in enumerate(us):
                fo, n, chan, block = x.unit(e, proj, u, K)
                if n != x.geo.hidden * 16 * K:
                    raise ValueError(f"layer {layer} expert {e} {proj} unit {u}: {n} bytes at K{K}")
                ranges.append((x.trellis_file(K).path, fo, n, int(plan.off[e, p, j])))
                ch_c[e, p, 128 * j:128 * j + 128] = chan
            blk_c[e, p] = block.float()
    read_ranges(ranges, slab)
    signs.copy_(torch.stack([x.t("signs.in"), x.t("signs.out")]).to(torch.float16))
    blk.copy_(blk_c)
    ch.copy_(ch_c)


@dataclass
class MoeSlab:
    """One rank's experts of one MoE layer as read: the plan's tables and slab (see MoePlan, read_moe)."""
    slab: torch.Tensor
    off: torch.Tensor
    k: torch.Tensor
    signs: torch.Tensor
    blk: torch.Tensor
    ch: torch.Tensor
    units: list[list[int]]


def load_moe_slab(root: str, layer: int, tp: int, rank: int, widths: Widths, device) -> MoeSlab:
    """Plan and read one MoE layer for one rank (tests and tools)."""
    x = ExpertLayer(root, layer)
    plan = plan_moe(x.geo, layer, tp, rank, widths)
    E1, H, upr = x.geo.experts + 1, x.geo.hidden, plan.off.shape[2]
    slab = torch.empty(plan.nbytes, dtype=torch.uint8, device=device)
    signs = torch.empty(2, H, dtype=torch.float16, device=device)
    blk = torch.empty(E1, 3, H // 128, dtype=torch.float32, device=device)
    ch = torch.empty(E1, 3, upr * 128, dtype=torch.float16, device=device)
    read_moe(root, layer, plan, slab, signs, blk, ch)
    return MoeSlab(slab, plan.off.to(device), plan.k.to(device), signs, blk, ch, plan.units)
