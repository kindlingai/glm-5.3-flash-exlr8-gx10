# SPDX-License-Identifier: Apache-2.0
"""Load format "exlr8": an exlr8 v2 release (kindlingai/exlr8 FORMAT.md) read as stored.

Experts: each MoE layer's Exlr8MoEMethod reads its rank's units into its own parameters. Dense matrices and vectors:
yielded to the model's load_weights under their HF names, matrices decoded to BF16 on the GPU from their records.
Any other model directory (a BF16 drafter, say) loads as the default loader would.
"""

import dataclasses
import json
import os
import time

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)


def is_exlr8_release(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "dense", "exlr8-vocab.safetensors")) and \
        os.path.isdir(os.path.join(path, "experts"))


def _file_tensors(f, device) -> "iter[tuple[str, torch.Tensor]]":
    """One dense or vocabulary file's tensors under HF names; matrices as BF16 (out, in)."""
    # The quantization package imports the MoE layers, which import this package: import it late.
    from vllm.model_executor.layers.quantization.exlr8 import reference as ref
    names = json.loads(f.meta["source_names"])
    stored = f.names()
    owned = set()
    for m in (n[:-len(".record")] for n in stored if n.endswith(".record")):
        rec = f.get(m + ".record")
        fmt = int(rec[0])
        parts = {k: f.get(f"{m}.{k}").to(device) for k in ref.DATA[fmt]}
        w = ref.dense(rec, parts).to(torch.bfloat16)
        hf = names[m]
        yield (hf if hf.endswith(".weight") else hf + ".weight"), w
        owned.update(f"{m}.{k}" for k in ("record", "checksum") + ref.DATA[fmt])
    for n in stored:
        if n not in owned:
            yield names.get(n, n), f.get(n)


def dense_weights(root: str, layers: int, device) -> "iter[tuple[str, torch.Tensor]]":
    """Every dense layer below `layers` (the MTP layers are the drafter's), then the vocabulary file."""
    paths = [os.path.join(root, "dense", f"exlr8-dense-{L:03d}.safetensors") for L in range(layers)]
    paths.append(os.path.join(root, "dense", "exlr8-vocab.safetensors"))
    from vllm.model_executor.layers.quantization.exlr8 import release

    for p in paths:
        yield from _file_tensors(release.SafeFile(p), device)
        # The decode's fp32 temporaries are freed; hand their blocks back before the next file (unified memory:
        # reserved blocks are memory the rest of the box cannot use).
        torch.cuda.empty_cache()


class Exlr8ModelLoader(DefaultModelLoader):
    def _default(self) -> DefaultModelLoader:
        """The default loader for any other directory, e.g. a BF16 drafter, which the speculator loads with the
        target's load format."""
        return DefaultModelLoader(dataclasses.replace(self.load_config, load_format="auto"))

    def download_model(self, model_config: ModelConfig) -> None:
        if not is_exlr8_release(model_config.model):
            self._default().download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        root = model_config.model
        if not is_exlr8_release(root):
            return self._default().load_weights(model, model_config)
        from vllm.model_executor.layers.quantization.exlr8 import Exlr8MoEMethod

        t0 = time.perf_counter()
        text = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        device = torch.device("cuda", torch.cuda.current_device())
        with torch.no_grad():
            loaded = model.load_weights(dense_weights(root, text.num_hidden_layers, device)) or set()
            n = 0
            for name, m in model.named_modules():
                method = getattr(m, "quant_method", None)
                if isinstance(method, Exlr8MoEMethod):
                    method.load(m, root)
                    loaded.update(f"{name}.{p}" for p, _ in m.named_parameters(recurse=False))
                    n += 1
        torch.cuda.synchronize()
        missing = sorted(p for p, _ in model.named_parameters() if p not in loaded)
        logger.info("exlr8: loaded %d dense tensors and %d MoE layers in %.1f s", len(loaded), n,
                    time.perf_counter() - t0)
        if missing:
            shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
            logger.warning("exlr8: %d parameters have no tensor in the release: %s", len(missing), shown)
