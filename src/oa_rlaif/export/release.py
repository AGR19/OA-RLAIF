"""Merge LoRA adapters, export GGUF and write an Ollama Modelfile (plan step 16). Runs locally on the GPU box.

    python -m oa_rlaif.export.release merge --adapter outputs/checkpoints/sdft/final --out outputs/merged/sdft
    python -m oa_rlaif.export.release gguf  --merged outputs/merged/grpo --out outputs/release/oa-rlaif-e4b
    ollama create oa-rlaif-e4b -f outputs/release/oa-rlaif-e4b/Modelfile

GGUF conversion uses llama.cpp's convert_hf_to_gguf.py and llama-quantize from a local llama.cpp checkout
(LLAMA_CPP_DIR; scripts/lambda_setup.sh clones a pinned tag once). The GGUF embeds the Gemma 4 chat template
used in training; check it with `ollama show oa-rlaif-e4b --template`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ..train.common import load_config, load_model


def merge(adapter: Path, out: Path, base: str | None = None) -> None:
    from peft import PeftModel

    cfg = load_config()
    model, tokenizer = load_model(cfg, base)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    from transformers import AutoProcessor, AutoTokenizer

    try:
        AutoProcessor.from_pretrained(cfg["model"]["id"]).save_pretrained(str(out))
    except Exception:  # text-only checkpoint (smoke test)
        AutoTokenizer.from_pretrained(cfg["model"]["id"]).save_pretrained(str(out))


def gguf(merged: Path, out: Path, quant: str = "Q4_K_M") -> Path:
    llama = Path(os.environ.get("LLAMA_CPP_DIR", "third_party/llama.cpp"))
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "oa-rlaif-e4b-f16.gguf"
    final = out / f"oa-rlaif-e4b-{quant}.gguf"
    subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"],
                   check=True)
    quantize = next((p for p in [llama / "build/bin/llama-quantize", llama / "llama-quantize"] if p.exists()), None)
    if quantize is None:
        raise SystemExit("llama-quantize not built; run `cmake -B build && cmake --build build -j` in LLAMA_CPP_DIR")
    subprocess.run([str(quantize), str(f16), str(final), quant], check=True)
    write_modelfile(out, final.name)
    return final


def write_modelfile(out: Path, gguf_name: str) -> Path:
    path = out / "Modelfile"
    path.write_text(
        f"FROM ./{gguf_name}\n"
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 8192\n"
        "PARAMETER stop \"<turn|>\"\n"  # Gemma 4 end-of-turn token (id 106); Gemma 2/3 used <end_of_turn>
        "SYSTEM \"You are OA-RLAIF, an ontology alignment agent.\"\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--base", default=None, help="merged base to stack on (e.g. the merged SDFT model for GRPO)")
    g = sub.add_parser("gguf")
    g.add_argument("--merged", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()
    if args.command == "merge":
        merge(args.adapter, args.out, args.base)
    else:
        print(gguf(args.merged, args.out, args.quant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
