"""Load generator of the measurement contract (OpenAI-compatible /v1/completions, streaming).

Owned by the judge so that every number is computed here from raw per-request timestamps.
Cross-validated against `vllm bench serve` (see bench/reports/loadgen-selection.md).

Contract enforced on every request:
  - prompts are token-id arrays, random and unique per request and per session -> no prefix reuse,
    and `cache_prompt: false` is sent as well (llama.cpp extension)
  - `ignore_eos` + `max_tokens` -> fixed output length, verified from `usage` and the stream
  - streaming with usage -> TTFT / inter-token latency measured client side
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field

import aiohttp

TOKEN_LO = 1000          # avoid control / special tokens at the bottom of the vocab

# Every judge request states every sampler parameter. llama-server takes its defaults from the
# model's generation_config (embedded in the GGUF by the converter): Qwen2.5-Instruct silently gets
# repeat_penalty 1.1 / top_p 0.8, which turned "greedy" into something else (34/44 oracle prompts
# diverged on the framework's own default build). Task sampling params are layered on top.
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "dry_multiplier": 0.0, "xtc_probability": 0.0, "top_n_sigma": -1.0, "mirostat": 0}


def sampling(**override) -> dict:
    return {**NEUTRAL_SAMPLING, **override}


@dataclass
class Workload:
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    num_requests: int
    warmup_requests: int
    sampling: dict = field(default_factory=lambda: {"temperature": 0.8, "top_k": 40, "top_p": 0.95})
    vocab_hi: int = 100_000
    slo: dict | None = None     # e.g. {"ttft_ms": 1000, "tpot_ms": 50}

    @classmethod
    def from_dict(cls, d: dict) -> "Workload":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_prompts(wl: Workload, n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(TOKEN_LO, wl.vocab_hi) for _ in range(wl.prompt_tokens)] for _ in range(n)]


async def _one(session: aiohttp.ClientSession, url: str, prompt: list[int], wl: Workload, seed: int) -> dict:
    body = {"model": "judge", "prompt": prompt, "max_tokens": wl.output_tokens, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, "cache_prompt": False,
            "seed": seed, **sampling(**wl.sampling)}
    rec = {"t_send": time.perf_counter(), "chunk_t": [], "n_chunks": 0, "usage": None, "timings": None,
           "error": None, "text_len": 0}
    try:
        async with session.post(url + "/v1/completions", json=body) as r:
            if r.status != 200:
                rec["error"] = f"http {r.status}: {(await r.text())[:200]}"
                return rec
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    rec["usage"] = ev["usage"]
                if ev.get("timings"):
                    rec["timings"] = ev["timings"]
                for ch in ev.get("choices") or []:
                    txt = ch.get("text") or ""
                    if txt:
                        rec["chunk_t"].append(time.perf_counter())
                        rec["n_chunks"] += 1
                        rec["text_len"] += len(txt.encode())
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_end"] = time.perf_counter()
    return rec


async def _run(url: str, prompts: list[list[int]], wl: Workload, seed: int) -> tuple[float, list[dict]]:
    sem = asyncio.Semaphore(wl.concurrency)
    # One connection per request: reusing keep-alive connections across a long server-side queue
    # wait produced spurious ServerDisconnected errors (observed with llama-server at C > slots).
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        async def guarded(i, p):
            async with sem:
                return await _one(s, url, p, wl, seed + i)
        t0 = time.perf_counter()
        recs = await asyncio.gather(*(guarded(i, p) for i, p in enumerate(prompts)))
        wall = time.perf_counter() - t0
    return wall, list(recs)


def run_load(url: str, wl: Workload, seed: int) -> dict:
    """Warmup + one measured pass. Returns raw per-request records and the wall time."""
    if wl.warmup_requests:
        asyncio.run(_run(url, make_prompts(wl, wl.warmup_requests, seed ^ 0x5EED), wl, seed ^ 0x5EED))
    prompts = make_prompts(wl, wl.num_requests, seed)
    wall, recs = asyncio.run(_run(url, prompts, wl, seed))
    return {"wall_s": wall, "requests": [_compact(r) for r in recs], "seed": seed}


def _compact(r: dict) -> dict:
    """Keep raw timings (relative to send) so every metric can be recomputed later."""
    t0 = r["t_send"]
    ct = [round((t - t0) * 1000, 3) for t in r["chunk_t"]]
    return {"send_s": t0, "end_ms": round((r.get("t_end", t0) - t0) * 1000, 3), "chunk_ms": ct,
            "n_chunks": r["n_chunks"], "text_bytes": r["text_len"], "usage": r["usage"], "timings": r["timings"],
            "error": r["error"]}


# ------------------------------------------------------------------------------ metrics (derived)
def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def metrics(raw: dict, wl: Workload) -> dict:
    """All metrics are derived here from raw records; nothing reported by the server is trusted
    except `usage` (cross-checked against the number of streamed chunks)."""
    reqs = raw["requests"]
    ok = [r for r in reqs if not r["error"]]
    n_out = [n_generated(r) or r["n_chunks"] for r in ok]
    ttft = [r["chunk_ms"][0] for r in ok if r["chunk_ms"]]
    tpot = [(r["chunk_ms"][-1] - r["chunk_ms"][0]) / (n - 1) for r, n in zip(ok, n_out) if r["chunk_ms"] and n > 1]
    e2e = [r["end_ms"] for r in ok]
    total_out = sum(n_out)
    m = {
        "output_tok_per_s": total_out / raw["wall_s"],
        "req_per_s": len(ok) / raw["wall_s"],
        "ttft_ms_mean": sum(ttft) / len(ttft) if ttft else None, "ttft_ms_p50": _pct(ttft, .5),
        "ttft_ms_p99": _pct(ttft, .99),
        "tpot_ms_mean": sum(tpot) / len(tpot) if tpot else None, "tpot_ms_p50": _pct(tpot, .5),
        "tpot_ms_p99": _pct(tpot, .99),
        "e2e_ms_p50": _pct(e2e, .5), "e2e_ms_p99": _pct(e2e, .99),
        "n_ok": len(ok), "n_err": len(reqs) - len(ok), "total_output_tokens": total_out,
    }
    if wl.slo:
        good = [n for r, n, a, b in zip(ok, n_out, ttft, tpot)
                if a <= wl.slo.get("ttft_ms", float("inf")) and b <= wl.slo.get("tpot_ms", float("inf"))]
        m["goodput_tok_per_s"] = sum(good) / raw["wall_s"]
        m["slo_attainment"] = len(good) / len(reqs) if reqs else 0.0
    return m


def n_generated(r: dict) -> int | None:
    """Generated-token count: `usage` if present, else the server's timings (older llama.cpp builds
    occasionally omit the usage chunk on a stream)."""
    u = (r.get("usage") or {}).get("completion_tokens")
    if u is not None:
        return u
    return (r.get("timings") or {}).get("predicted_n")


def contract_violations(raw: dict, wl: Workload) -> list[str]:
    """Anti-cheat checks on the measured pass itself."""
    v = []
    reqs = raw["requests"]
    errs = [r for r in reqs if r["error"]]
    if errs:
        v.append(f"{len(errs)}/{len(reqs)} requests failed: {errs[0]['error']}")
    counted = [(r, n_generated(r)) for r in reqs if not r["error"]]
    unknown = [r for r, n in counted if n is None]
    if unknown:
        v.append(f"{len(unknown)} requests reported no token count (neither usage nor timings)")
    short = [r for r, n in counted if n is not None and n < wl.output_tokens]
    if short:
        v.append(f"{len(short)} requests produced fewer than {wl.output_tokens} tokens (ignore_eos not honoured)")
    # Chunk counts are not token counts (servers merge byte-fallback tokens that are incomplete UTF-8;
    # observed: < 50% chunks/tokens on honest short random outputs). Every generated token yields at
    # least one byte of text, so far fewer streamed bytes than claimed tokens means invented work.
    lying = [r for r in reqs if not r["error"] and r["usage"]
             and r.get("text_bytes", r["n_chunks"]) < 0.5 * r["usage"]["completion_tokens"]]
    if lying:
        v.append(f"{len(lying)} requests: usage.completion_tokens disagrees with streamed chunks")
    skipped = [r for r in reqs if not r["error"] and r["timings"] and r["timings"].get("prompt_n") is not None
               and r["timings"]["prompt_n"] < 0.9 * wl.prompt_tokens]
    if skipped:
        v.append(f"{len(skipped)} requests skipped prefill (prompt_n < 90% of prompt): prompt cache or shortcut")
    no_prompt = [r for r in reqs if not r["error"] and r["usage"]
                 and r["usage"].get("prompt_tokens", wl.prompt_tokens) < wl.prompt_tokens]
    if no_prompt:
        v.append(f"{len(no_prompt)} requests: server saw a shorter prompt than sent")
    return v
