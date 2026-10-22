#!/usr/bin/env python3
"""Make model/all_merged importable by Ollama, without changing it.

Gemma 4 E4B shares K/V in its last 18 layers (24-41): they reuse the K/V of layer 22 (sliding) or 23 (full), so their
own k_proj, v_proj and k_norm are never used, and transformers 5.x does not save them. Ollama's loader still requires
them. This writes cache/import/: links to the real files plus kv_shared.safetensors with those 54 tensors, copied from
the layer each one shares with (or zeros with --zeros, to show that the values do not change the output).
transformers 5.x also writes the full-attention head size as per_layer_config ({"05": {"head_dim": 512}, ...}) instead
of Google's global_head_dim: 512, which is the only form Ollama reads; config.json here gets it back.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SRC = HERE / "model" / "all_merged"
DST = HERE / "cache" / ("import_zeros" if "--zeros" in sys.argv else "import")
cfg = json.loads((SRC / "config.json").read_text())
text = cfg.get("text_config", cfg)
types, first = text["layer_types"], text["num_hidden_layers"] - text["num_kv_shared_layers"]
DST.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.is_file() and f.name != "config.json" and not (DST / f.name).exists():
        (DST / f.name).symlink_to(f)
per_layer = text.get("per_layer_config") or {}
if "global_head_dim" not in text and per_layer:
    dims = {v["head_dim"] for k, v in per_layer.items() if types[int(k)] == "full_attention" and "head_dim" in v}
    assert len(dims) == 1, dims
    text["global_head_dim"] = dims.pop()
    text.setdefault("num_global_key_value_heads", None)
(DST / "config.json").unlink(missing_ok=True)
(DST / "config.json").write_text(json.dumps(cfg, indent=2))
extra = {}
with safe_open(SRC / "model.safetensors", "pt") as weights:
    prefix = next(k for k in weights.keys() if k.endswith("layers.0.self_attn.k_proj.weight")).split("layers.0.")[0]
    for layer in range(first, len(types)):
        source = max(i for i in range(first) if types[i] == types[layer])  # the layer whose K/V it reuses
        for part in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            name = f"{prefix}layers.{layer}.self_attn.{part}"
            tensor = weights.get_tensor(f"{prefix}layers.{source}.self_attn.{part}")
            extra[name] = torch.zeros_like(tensor) if "--zeros" in sys.argv else tensor.clone()
save_file(extra, DST / "kv_shared.safetensors", metadata={"format": "pt"})
# a standard sharded-checkpoint index, so the importer reads both files
with safe_open(SRC / "model.safetensors", "pt") as weights:
    weight_map = {name: "model.safetensors" for name in weights.keys()}
weight_map.update({name: "kv_shared.safetensors" for name in extra})
size = (SRC / "model.safetensors").stat().st_size + (DST / "kv_shared.safetensors").stat().st_size
(DST / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map}, indent=1))
print(f"{DST}: global_head_dim {text.get('global_head_dim')}; {len(extra)} tensors for layers {first}-{len(types) - 1} ({(DST / 'kv_shared.safetensors').stat().st_size / 1e6:.0f} MB)")
