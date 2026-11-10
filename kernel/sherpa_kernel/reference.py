"""Generate the correctness reference for a model with HF transformers (independent of any serving
framework). Run once per (model, prompt set); scoring then needs no torch.

    <python-with-torch> -m sherpa_kernel.reference --model <hf_dir> --prompts <jsonl> --out <json>

For every prompt: the prompt token ids, the reference greedy continuation, and per generated
position the top-K token ids / logprobs. For a few "sampling probes" the full top-K distribution
of the first generated position is kept for the sampler distribution test.

This file must import nothing beyond the standard library at module level, so that the judge can
import it without torch; torch/transformers are imported inside main().
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

TOP_K = 20
PROBE_TOP_K = 64


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    a = ap.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    free = torch.cuda.mem_get_info()[0]
    n_params_guess = sum(p.stat().st_size for p in Path(a.model).glob("*.safetensors")) / 2  # bf16 files
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(a.dtype) or (
        torch.float32 if n_params_guess * 4 < 0.6 * free else torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).cuda().eval()

    prompts = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    eos = set()
    for src in (cfg.get("eos_token_id"), tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        eos |= set(src if isinstance(src, list) else [src] if src is not None else [])
    out = {"model": a.model, "config": {k: cfg.get(k) for k in ("model_type", "vocab_size", "num_hidden_layers")},
           "eos_ids": sorted(eos),
           "dtype": str(dtype).replace("torch.", ""), "gen_tokens": a.gen, "top_k": TOP_K,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
           "host": platform.node(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "items": []}
    t0 = time.time()
    with torch.no_grad():
        for p in prompts:
            ids = tok(p["text"], add_special_tokens=True)["input_ids"]
            x = torch.tensor([ids], device="cuda")
            past, gen, steps = None, [], []
            cur = x
            n_gen = p.get("gen", a.gen)
            for _ in range(n_gen):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
                k = PROBE_TOP_K if (p.get("probe") and not steps) else TOP_K
                v, i = torch.topk(lp, k)
                nxt = int(i[0])
                ent = float(-(lp.exp() * lp).sum())
                steps.append({"ids": i.tolist(), "lp": [round(float(t), 5) for t in v], "entropy": round(ent, 4)})
                gen.append(nxt)
                cur = torch.tensor([[nxt]], device="cuda")
                if p.get("probe"):
                    break
            out["items"].append({"id": p["id"], "kind": p.get("kind", "text"), "probe": bool(p.get("probe")),
                                 "prompt_ids": ids, "gen_ids": gen, "steps": steps})
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"items": len(out["items"]), "dtype": out["dtype"], "seconds": out["seconds"]}))
    return 0


def margin(step: dict) -> float:
    """Reference confidence at a position: logprob gap between its top-1 and top-2 tokens."""
    lp = step["lp"]
    return lp[0] - lp[1] if len(lp) > 1 else math.inf


if __name__ == "__main__":
    raise SystemExit(main())
