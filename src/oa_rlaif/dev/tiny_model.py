"""Build a tiny, randomly initialised Gemma 4 (same architecture as E4B, ~26M parameters) for the Mac smoke test.

Uses only the cached E4B config and tokenizer (no weights are downloaded). The model is useless for quality and only
exercises the code paths: OPSD, SFT and GRPO trainers, merging, evaluation and the agent loop.

    python -m oa_rlaif.dev.tiny_model outputs/smoke/tiny-gemma4
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def build(out: Path) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForImageTextToText

    src = Path(snapshot_download("google/gemma-4-E4B-it", local_files_only=True,
                                 allow_patterns=["config.json", "tokenizer*", "chat_template.jinja"]))
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    layers = 2
    tc.update(hidden_size=64, num_hidden_layers=layers, intermediate_size=128, num_attention_heads=2,
              num_key_value_heads=1, head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
              hidden_size_per_layer_input=16, num_kv_shared_layers=0, layer_types=tc["layer_types"][:layers],
              expert_intermediate_size=64)
    for key in ("vision_config", "audio_config"):
        for k, v in (("hidden_size", 32), ("intermediate_size", 64), ("num_hidden_layers", 1),
                     ("num_attention_heads", 2), ("num_key_value_heads", 1), ("head_dim", 16)):
            if k in cfg[key]:
                cfg[key][k] = v
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))
    model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(out))
    model.save_pretrained(out)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(src / name, out / name)
    print(f"tiny Gemma 4: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters -> {out}")
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/smoke/tiny-gemma4"))
