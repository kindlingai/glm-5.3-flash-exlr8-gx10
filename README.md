# GLM-5.3-Flash exlr8 recipe (GB10, TP=1/2/3/4)

A serving recipe for GLM-5.3-Flash with **exlr8** routed experts — EXL3-trellis
units decoded by a custom MoE kernel — on ASUS GX10 / NVIDIA GB10. It is the
NVFP4 Flash recipe (see [README-nvfp4-base.md](README-nvfp4-base.md)) with the
`kindlingai/vllm-exlr8` stack layered on top: the routed experts come from
trellis units, the dense linears stay FP8/NVFP4, and everything else — DFlash2
speculation, adaptive-k, RecoverSSM, sparse MLA, the mentat control plane — is
unchanged.

Private while the Flash quant is unreleased.

## Why exlr8

Against the shipping NVFP4 Flash build, measured at TP=4:

- Decode **+3–8%** (code +7.7%, prose +3.0%).
- NLL vs bf16 **equal or better** (+0.0034 / +0.0017 / +0.0020 on the three sets).
- KV pool **~+15%** like-for-like (smaller expert weights free memory for KV).
- Prefill is currently **−7.7%** at 32k/128k — the one open gap, entirely in the
  routed MoE kernel at the 8,192-token prefill budget. Work in progress.

Expert width trades size for quality and is chosen at boot
(`VLLM_EXLR8_WIDTHS`): **k2 ≈ 2.25 bpw**, **k3 ≈ 3 bpw**, **k4 ≈ 3.38 bpw**.

## Topologies

One assembled release holds all three widths (the `experts/` are per-layer
`k2`/`k3`/`k4` files; width is picked at runtime). Only the config padding is
topology-specific, so TP=1 and TP=2 reuse the TP=4 release.

| TP | Boxes | Width | Release (`MODEL_DIR`) | Notes |
|---|---|---|---|---|
| 1 | 1 | k2 | `exlr8-release` | ~87 GiB, fits one 128 GB box |
| 2 | 2 | k3 | `exlr8-release` | 64 heads / 2048 inter split by 2, no padding |
| 3 | 3 | k4 | `exlr8-release-tp3` | padded config (66 / 2304), `tp3.yaml` |
| 4 | 4 | k4 | `exlr8-release` | the validated default |

## RigMark (clocks capped at 2200 MHz)

RigMark 1.0.0, `reasoning=low`, 8K prefill depth. Pending the test runs in this
session; filled per topology as each completes.

| TP (width) | code tok/s | prose tok/s | structured tok/s | 8K prefill cold | multi-agent C1 / C2 / C4 |
|---|---|---|---|---|---|
| 1 (k2) | _pending_ | | | | |
| 2 (k3) | _pending_ | | | | |
| 3 (k4) | _pending_ | | | | |
| 4 (k4) | _pending_ | | | | |

### Decode and prefill, measured 2026-10-19 (image v26, clocks TP-locked at 2200 MHz)

`gate/bench/bench.py` decode (16 runs, code and prose; spec-decode DFlash2 +
adaptive-k) and `prefill_sweep.py`. Different harness from RigMark above, which
stays pending. tok/s.

| TP (width) | decode code | decode prose | prefill 8K | prefill 32K |
|---|---|---|---|---|
| 1 (k2) | — | — | — | — |
| 2 (k3) | 84.2 | 50.1 | 2,609 | 2,687 |
| 3 (k4) | _pending_ | _pending_ | _pending_ | _pending_ |
| 4 (k4) | 128.0 | 72.5 | 4,393 | 4,190 |

- **TP=1** (one box, k2) needs `tp1.yaml` and a distinct `MENTAT_GROUP` so the
  box does not join an existing cluster:
  `sudo env TP=1 MENTAT_GROUP=glm53-solo ./glm53f -f experimental/compose/tp1.yaml up -d`.
  The overlay pins KV small (exlr8.yaml's 38 GiB default does not fit beside the
  ~87 GiB weights on one box) and the entrypoint skips the NCCL fabric at TP=1.
- **TP=3** is pending: the padded `exlr8-release-tp3` config carries no
  `quantization` field, so vLLM loads it as bfloat16 rather than exlr8 and the
  engine fails to start. It needs the exlr8 quant forced on the padded path.

## Running it

The `./glm53f` launcher stacks `compose/glm53.yaml` + every overlay in
`experimental/compose` (including `exlr8.yaml` last). Run the same command on
every box; mentat elects the head and the workers. `EXLR8_TREE` defaults to this
repo's `tree/`.

Docker needs root on these hosts, so run under `sudo env` (it passes the knobs
through). Pin the head with `HEAD_HOST` + `ROLE` when you want a known API box.

```bash
# TP=4, k4 (default)
sudo env MODELS_ROOT=/mnt/nfs/models ./glm53f up -d

# TP=2, k3
sudo env TP=2 VLLM_EXLR8_WIDTHS=k3 ./glm53f up -d

# TP=1, k2 (single box)
sudo env TP=1 VLLM_EXLR8_WIDTHS=k2 ./glm53f up -d

# TP=3, k4 (padded config overlay)
sudo env TP=3 VLLM_EXLR8_WIDTHS=k4 MODEL_DIR=/models/nas/glm-5.3-flash/exlr8-release-tp3 \
  ./glm53f -f experimental/compose/tp3.yaml up -d
```

`EXLR8_TAG` names the weight snapshot (a re-boot at the same tag restores in
~15 s; a new tag does a full load from the release). Wait for `self-test:
PASSED` in `sudo ./glm53f logs`.

## What the overlay changes

`experimental/compose/exlr8.yaml` mounts the `tree/` of patched vLLM files over
the image (they win over the other overlays' copies) and sets
`--quantization exlr8 --load-format exlr8`. The files are
`kindlingai/vllm-exlr8` @ `a2e163cb05`; re-sync them with:

```bash
rsync -a /path/to/vllm-exlr8/vllm/<file> tree/vllm/<file>
```

The release (`MODEL_DIR`) is config + tokenizer + symlinks to the shared dense
weights and the `k2`/`k3`/`k4` expert units. It is small; the experts it points
at are the large files.

## Provenance

- vLLM stack: `kindlingai/vllm-exlr8` branch `exlr8` @ `a2e163cb05` (private).
- Base recipe: `mmastrac/glm-5.3-flash-4x-gx10` @ `0d1f462`.
- Expert units: `kindlingai/glm-5.3-flash-exlr8-k2-k3-k4`.
- Kernel: `kindlingai/exl3rate` (exl3rate MoE, ported to vLLM as `exlr8_moe.cu`).
