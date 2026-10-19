# GLM-5.3-Flash exlr8 recipe (GB10, TP=1/2/3/4)

A serving recipe for GLM-5.3-Flash with **exlr8** routed experts — EXL3-trellis
units decoded by a custom MoE kernel — on ASUS GX10 / NVIDIA GB10. It is the
NVFP4 Flash recipe (see [README-nvfp4-base.md](README-nvfp4-base.md)) with the
`kindlingai/vllm-exlr8` stack layered on top: the routed experts come from
trellis units, the dense linears stay FP8/NVFP4, and everything else — DFlash2
speculation, adaptive-k, RecoverSSM, sparse MLA, the mentat control plane — is
unchanged.

## Limits

 - Images work at every TP, but DFlash2 drafts from the text alone, so draft
   acceptance on a turn with an image can be lower than on a text-only turn.

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

## Decode and prefill, measured 2026-10-19 (clocks TP-locked at 2200 MHz)

`gate/bench/bench.py` decode (16 runs, code and prose; spec-decode DFlash2 +
adaptive-k) and `prefill_sweep.py`. tok/s.

| TP (width) | decode code | decode prose | prefill 8K | prefill 32K |
|---|---|---|---|---|
| 1 (k2) | 53.0 | 29.5 | 495 | 523 |
| 2 (k3) | 84.2 | 50.1 | 2,609 | 2,687 |
| 3 (k4) | 105.8 | 57.8 | 3,002 | 3,391 |
| 4 (k4) | 128.0 | 72.5 | 4,393 | 4,190 |

- **TP=1** (one box, k2) needs `tp1.yaml` and a distinct `MENTAT_GROUP` so the
  box does not join an existing cluster (add `SERVED_NAME=glm53-solo` too if a
  `glm53` cluster is up at the same time, so mentat-serve does not route to it):
  `sudo env TP=1 MENTAT_GROUP=glm53-solo ./glm53f -f experimental/compose/tp1.yaml up -d`.
  The entrypoint skips the NCCL fabric at TP=1.
- The TP=3 row used a 38 GiB KV pin, which left the head 0.5 GiB of free RAM
  and 4 GiB of swap in use. The shipped TP=3 pin is 26 GiB (see below).

## KV cache per TP

The entrypoint picks the KV pin from `TP` when `QUANT=exlr8`. Each pin is sized
for DGX OS: with the vision tower loaded, it left the head about 4.5 GiB of free
RAM at its lowest point. The load was 480k-token prefills and four concurrent
128k requests at TP=3 and TP=4, and 128k-160k prefills and four concurrent 32k
requests at TP=1 and TP=2, whose context is 160k. On kindling-spark-os,
`dispram.yaml` adds `KV_EXTRA_MEMORY` (2 GiB, the RAM its 64 KiB kernel returns)
to the pin at TP=2 and up, and dispramd lends the display carveout on top. TP=1
takes no extra: at 12 GiB its one box went 1.7 GiB into swap. Set
`KV_CACHE_MEMORY` to override.

| TP (width) | DGX OS pin | DGX OS KV tokens | spark-os pin | spark-os KV tokens | head's lowest free RAM, DGX OS |
|---|---|---|---|---|---|
| 1 (k2) | 10 GiB | 1,122,607 | 10 GiB | 1,359,265 | 5.1 GiB |
| 2 (k3) | 30 GiB | 4,208,867 | 32 GiB | 4,776,846 | ~4.4 GiB |
| 3 (k4) | 26 GiB | 4,258,159 | 28 GiB | 4,926,962 | ~4.7 GiB |
| 4 (k4) | 38 GiB | 6,459,582 | 40 GiB | 7,147,571 | ~8.2 GiB |

TP=1 DGX OS figures come from a GX10 on stock Ubuntu. The other DGX OS token
counts come from spark-os boots with `DISPRAM=0` and no `KV_EXTRA_MEMORY`,
which give the pool DGX OS gets at that pin. Their free-RAM figures are the
spark-os low point less 2 GiB. TP=4 keeps a wider margin: its pool already holds
12 requests at the full 512k context on DGX OS.

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
weights and the `k2`/`k3`/`k4` expert units. `dense/vision-bf16.safetensors`
holds the vision tower; with every `LIMIT_MM` limit at 0 the loader skips it. It is small; the experts it points
at are the large files.

## Provenance

- vLLM stack: `kindlingai/vllm-exlr8` branch `exlr8` @ `a2e163cb05` (private).
- Base recipe: `mmastrac/glm-5.3-flash-4x-gx10` @ `0d1f462`.
- Expert units: `kindlingai/glm-5.3-flash-exlr8-k2-k3-k4`.
- Vision tower: the 347 `model.visual.*` BF16 tensors of the GLM-5.3-Flash BF16
  checkpoint (revision `91b69215804f2d2036a8867f6ce44848e278dcda`), copied byte
  for byte. They are identical to the tower in the NVFP4 checkpoint.
- Kernel: `kindlingai/exl3rate` (exl3rate MoE, ported to vLLM as `exlr8_moe.cu`).
