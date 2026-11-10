"""Correctness oracle: does the deliverable still serve the same model, with the same API contract?

Reference = HF transformers (fp32 when it fits), independent of any serving framework
(reference.py). Every check runs *while the server carries the task's own load*, so batching,
scheduling and KV paths are exercised as during measurement (v0 only checked batch=1).

tokens    greedy continuations of fixed natural-text prompts (short + long context) compared with
          the reference trajectory position by position:
            match  identical over the whole length
            tie    first divergence where the reference found the candidate's token almost as likely
                   (delta = lp_ref(top1) - lp_ref(cand) < tau): floating-point noise
            hard   divergence at a confident position: the computation changed
logprobs  |logprob error| of the chosen token and a top-k KL estimate on the common prefix
          (a deliverable that returns wrong logprobs breaks the API even if tokens are right;
          observed: llama.cpp `-bs` under concurrent load)
sampling  many 1-token samples on high-entropy probes, compared with the distribution implied by the
          reference logits under llama.cpp's sampler semantics: G-test against a Monte-Carlo null,
          plus the mass sampled outside the nucleus. Greedy checks never touch this path (v0 DES-14).
context   one request of the task's contracted context length must be served while the server carries
          the full task workload (v0's `-np 32` silently cut the per-request context; a shared KV
          pool sized only for the workload fails every in-flight request when it overflows).

Thresholds come from bench/oracle/policy.yaml, calibrated on default builds
(bench/reports/oracle-calibration.md), never from the deliverable.
"""
from __future__ import annotations

import asyncio
import math
import random
from collections import Counter

import aiohttp

from . import loadgen as G

TOP_LOGPROBS = 8
# T < 1 so that a dropped temperature is detectable; llama.cpp applies top_k -> top_p -> min_p on the
# raw distribution and temperature last (default sampler chain), which expected_dist() reproduces.
SAMPLING = {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0}
BG_OUTPUT_TOKENS = 32


# ------------------------------------------------------------------------------------- requests
# llama.cpp native /completion: `tokens` is the exact generated id sequence. Both OpenAI
# `logprobs.content` and native `completion_probabilities` merge byte-fallback tokens that are
# incomplete UTF-8, so neither is a faithful token list (observed on prompt t25).
def align_probs(tokens: list[int], probs: list[dict]) -> tuple[list, list]:
    """Map probability entries (possibly fewer, merged) onto token positions by id, in order."""
    lp, top = [None] * len(tokens), [None] * len(tokens)
    j = 0
    for e in probs:
        while j < len(tokens) and tokens[j] != e.get("id"):
            j += 1
        if j >= len(tokens):
            break
        lp[j] = e.get("logprob")
        top[j] = {t["id"]: t["logprob"] for t in e.get("top_logprobs") or []}
        j += 1
    return lp, top


async def _post(s: aiohttp.ClientSession, url: str, body: dict) -> dict:
    async with s.post(url + "/completion", json=body) as r:
        return await r.json(content_type=None)


GREEDY = G.sampling(temperature=0.0, top_k=1)


async def _greedy(s, url: str, item: dict) -> dict:
    body = {"prompt": item["prompt_ids"], "n_predict": len(item["gen_ids"]), **GREEDY,
            "n_probs": TOP_LOGPROBS, "return_tokens": True, "cache_prompt": False, "seed": 0}
    try:
        d = await _post(s, url, body)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"id": item["id"], "error": f"{type(e).__name__}: {e}"}
    toks = d.get("tokens")
    if toks is None:
        return {"id": item["id"], "error": f"no tokens: {str(d.get('error') or d)[:240]}"}
    lp, top = align_probs(toks, d.get("completion_probabilities") or [])
    return {"id": item["id"], "ids": toks, "lp": lp, "top": top,
            "finish": "length" if d.get("stop_type") == "limit" else "stop", "error": None}


async def _sample(s, url: str, item: dict, seed: int) -> int | None:
    body = {"prompt": item["prompt_ids"], "n_predict": 1, "return_tokens": True, "cache_prompt": False,
            "seed": seed, **G.sampling(**SAMPLING)}
    try:
        return (await _post(s, url, body))["tokens"][0]
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, TypeError):
        return None


async def _context_probe(s, url: str, n_ctx: int, vocab_hi: int, seed: int) -> dict:
    rng = random.Random(seed)
    n_gen = 8
    prompt = [rng.randrange(G.TOKEN_LO, vocab_hi) for _ in range(n_ctx - n_gen)]
    try:
        d = await _post(s, url, {"prompt": prompt, "n_predict": n_gen, **GREEDY, "ignore_eos": True,
                                 "return_tokens": True, "cache_prompt": False})
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"n_ctx": n_ctx, "ok": False, "error": f"{type(e).__name__}: {e}"}
    ok = len(d.get("tokens") or []) == n_gen
    return {"n_ctx": n_ctx, "ok": ok, "error": None if ok else str(d.get("error") or d)[:240]}


async def _background(s, url: str, wl: G.Workload, seed: int, stop: asyncio.Event) -> int:
    done = 0
    rng = random.Random(seed)

    async def worker(w: int):
        nonlocal done
        i = 0
        while not stop.is_set():
            p = [rng.randrange(G.TOKEN_LO, wl.vocab_hi) for _ in range(wl.prompt_tokens)]
            r = await G._one(s, url, p, wl, seed + 1000 * w + i)
            i += 1
            done += 0 if r["error"] else 1
    await asyncio.gather(*(worker(w) for w in range(wl.concurrency)))
    return done


def split_load(wl: G.Workload | None) -> tuple[G.Workload | None, int]:
    """Half of the task's concurrency carries oracle requests, the other half short background
    requests: the server sees the task's batch sizes and the oracle does not queue behind long work."""
    if not wl or wl.concurrency <= 1:
        return None, 1
    oracle_c = max(1, wl.concurrency // 2)
    bg = G.Workload(prompt_tokens=wl.prompt_tokens, output_tokens=min(wl.output_tokens, BG_OUTPUT_TOKENS),
                    concurrency=wl.concurrency - oracle_c, num_requests=0, warmup_requests=0,
                    sampling=wl.sampling, vocab_hi=wl.vocab_hi)
    return bg, oracle_c


async def _under_load(url: str, bg: G.Workload | None, jobs, seed: int, concurrency: int):
    conn = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600)) as s:
        stop = asyncio.Event()
        task = asyncio.create_task(_background(s, url, bg, seed, stop)) if bg else None
        if task:
            await asyncio.sleep(1.0)
        sem = asyncio.Semaphore(concurrency)

        async def guarded(job):
            async with sem:
                return await job(s)
        out = await asyncio.gather(*(guarded(j) for j in jobs))
        stop.set()
        n_bg = await task if task else 0
    return out, n_bg


def run(url: str, ref: dict, wl: G.Workload | None, contract: dict, n_samples: int = 500, seed: int = 4242) -> dict:
    """Raw oracle observations under the task's workload `wl` (None = idle server)."""
    max_ctx = contract.get("max_context")
    fits = [it for it in ref["items"] if not it["probe"]
            and (not max_ctx or len(it["prompt_ids"]) + len(it["gen_ids"]) <= max_ctx)]
    short = [it for it in fits if it["kind"] != "long"]
    long_ = [it for it in fits if it["kind"] == "long"]
    probes = [it for it in ref["items"] if it["probe"]]
    bg, conc = split_load(wl)
    vocab_hi = wl.vocab_hi if wl else 100_000
    # phase 1: short prompts, interleaved with background load at the task's batch size
    greedy, n_bg1 = asyncio.run(_under_load(url, bg, [lambda s, it=it: _greedy(s, url, it) for it in short],
                                            seed, conc))
    # phase 2 (contract): ONE long request at a time while the server carries the FULL task workload.
    # Observed: when a shared KV pool overflows, llama.cpp fails every request in the batch, so a burst
    # of long requests is a different (stronger) contract than the one tasks state.
    full = G.Workload(**{**wl.__dict__, "num_requests": 0, "warmup_requests": 0}) if wl and wl.concurrency > 1 else None
    jobs = [lambda s, it=it: _greedy(s, url, it) for it in long_]
    if max_ctx:
        jobs.append(lambda s: _context_probe(s, url, max_ctx, vocab_hi, seed))
    out, n_bg_ctx = asyncio.run(_under_load(url, full, jobs, seed + 3, 1))
    greedy += out[:-1] if max_ctx else out
    ctx = out[-1] if max_ctx else None
    n_bg1 += n_bg_ctx
    order = [it["id"] for it in probes for _ in range(n_samples)]
    sjobs = [lambda s, it=it, k=k: _sample(s, url, it, seed + 7919 * k) for it in probes for k in range(n_samples)]
    perm = list(range(len(sjobs)))
    random.Random(seed).shuffle(perm)          # interleave probes so batches mix different prompts
    res, n_bg2 = asyncio.run(_under_load(url, bg, [sjobs[i] for i in perm], seed + 1, conc))
    samples: dict[str, list] = {it["id"]: [] for it in probes}
    for i, tok in zip(perm, res):
        samples[order[i]].append(tok)
    out = {"greedy": greedy, "context": ctx, "samples": samples, "background_completed": n_bg1 + n_bg2,
           "n_samples": n_samples, "seed": seed, "sampling": SAMPLING}
    if contract.get("accuracy"):
        acc, _ = asyncio.run(_under_load(url, bg, accuracy_jobs(url), seed + 2, conc))
        out["accuracy"] = acc
    return out


# ------------------------------------------------------------------------------------- accuracy
# Task-level quality for lossy contracts (quantisation etc.): GSM8K test[0:100], 4-shot, greedy.
# Token identity is not expected from a lossy deliverable; keeping the answers right is.
def _gsm8k():
    from .util import BENCH
    import json
    items = [json.loads(x) for x in (BENCH / "oracle" / "gsm8k-100.jsonl").read_text().splitlines() if x.strip()]
    shots = json.loads((BENCH / "oracle" / "gsm8k-shots.json").read_text())["shots"]
    head = "".join(f"Question: {s['question']}\nAnswer: {s['solution']}\n#### {s['answer']}\n\n" for s in shots)
    return items, head


def accuracy_jobs(url: str):
    items, head = _gsm8k()

    async def one(s, it):
        body = {"prompt": head + f"Question: {it['question']}\nAnswer:", "n_predict": 320, **GREEDY,
                "cache_prompt": False, "stop": ["\n\nQuestion:", "Question:"]}
        try:
            d = await _post(s, url, body)
            return {"id": it["id"], "text": d.get("content", ""), "error": None}
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return {"id": it["id"], "text": "", "error": str(e)}
    return [lambda s, it=it: one(s, it) for it in items]


def gsm8k_answer(text: str) -> str | None:
    import re
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    v = m_val.replace(",", "").rstrip(".")
    return v[:-2] if v.endswith(".0") else v


def accuracy_of(raw_acc: list[dict]) -> dict:
    items, _ = _gsm8k()
    gold = {it["id"]: it["answer"] for it in items}
    correct = {r["id"]: (gsm8k_answer(r["text"]) == gold[r["id"]]) for r in raw_acc}
    return {"n": len(correct), "accuracy": sum(correct.values()) / max(1, len(correct)), "correct": correct}


# ------------------------------------------------------------------------------ pure comparisons
def _delta(step: dict, tok: int) -> float:
    """How much less likely the reference found `tok` than its own top-1 (nats); inf if not in top-K."""
    try:
        return step["lp"][0] - step["lp"][step["ids"].index(tok)]
    except ValueError:
        return math.inf


def _kl_topk(step: dict, cand_top: dict) -> float:
    """KL(ref || cand) over the reference's top-k; missing cand entries floored at its own minimum."""
    cand_top = {int(k): v for k, v in (cand_top or {}).items()}   # JSON round-trip turns int keys into str
    floor = min(cand_top.values()) if cand_top else -30.0
    kl = sum(math.exp(lpr) * (lpr - cand_top.get(tid, floor))
             for tid, lpr in zip(step["ids"][:TOP_LOGPROBS], step["lp"][:TOP_LOGPROBS]))
    return max(0.0, kl)


def _res(item, verdict, pos, n, lp_err, kls, delta=None, cand_tok=None) -> dict:
    return {"id": item["id"], "kind": item["kind"], "verdict": verdict, "match_len": pos, "len": n,
            "delta": delta, "cand_tok": cand_tok, "lp_err": lp_err, "kl": kls}


def compare_trajectory(item: dict, cand: dict, eos_ids: set[int], tau: float) -> dict:
    if cand.get("error"):
        return {"id": item["id"], "kind": item["kind"], "verdict": "error", "error": cand["error"],
                "match_len": 0, "len": len(item["gen_ids"])}
    ref_ids, steps = item["gen_ids"], item["steps"]
    n = len(ref_ids)
    lp_err, kls = [], []
    for j in range(n):
        ref_tok = ref_ids[j]
        if j > 0 and ref_ids[j - 1] in eos_ids:
            # both emitted the same end-of-generation token at j-1 (it matched): the sequence is over.
            # The HF reference loop keeps generating past EOS; a server rightly stops (Qwen2.5 t10).
            return _res(item, "match", j, n, lp_err, kls)
        if j >= len(cand["ids"]):
            # The oracle asks for exactly len(ref) tokens, so stopping early is only legitimate on an
            # end-of-generation token. A "length" stop here means the server capped the output.
            if ref_tok in eos_ids:
                return _res(item, "match", j, n, lp_err, kls)
            if cand.get("finish") == "length":
                return _res(item, "hard", j, n, lp_err, kls, delta=math.inf, cand_tok="<truncated>")
            d = min((_delta(steps[j], e) for e in eos_ids), default=math.inf)
            return _res(item, "tie" if d < tau else "hard", j, n, lp_err, kls, delta=d, cand_tok="<stop>")
        c = cand["ids"][j]
        if c != ref_tok:
            if ref_tok in eos_ids:  # the reference would have stopped here
                return _res(item, "match", j, n, lp_err, kls)
            d = _delta(steps[j], c)
            return _res(item, "tie" if d < tau else "hard", j, n, lp_err, kls, delta=d, cand_tok=c)
        if cand["lp"][j] is not None:  # merged byte-fallback positions carry no probabilities
            lp_err.append(abs(cand["lp"][j] - steps[j]["lp"][0]))
            kls.append(_kl_topk(steps[j], cand["top"][j]))
    return _res(item, "match", n, n, lp_err, kls)


def expected_dist(step: dict, s: dict = SAMPLING) -> tuple[dict[int, float], int | None]:
    """Distribution implied by the reference logits under llama.cpp's chain
    (top_k -> top_p on the untempered softmax -> min_p -> temperature).
    Returns (q, boundary_token): the first token excluded by top_p is ambiguous under tiny logit
    differences, so its samples are ignored rather than counted as violations."""
    ids, lps = step["ids"][: s["top_k"]], step["lp"][: s["top_k"]]
    z = max(lps)
    ps = [math.exp(x - z) for x in lps]
    tot = sum(ps)
    ps = [p / tot for p in ps]
    keep, cum = 0, 0.0
    for p in ps:
        keep += 1
        cum += p
        if cum >= s["top_p"]:
            break
    if s.get("min_p"):
        keep = max(1, sum(1 for p in ps[:keep] if p >= s["min_p"] * ps[0]))
    w = [math.exp((lp - z) / s["temperature"]) for lp in lps[:keep]]
    tot = sum(w)
    q = {tid: x / tot for tid, x in zip(ids[:keep], w)}
    return q, ids[keep] if keep < len(ids) else None


def g_stat(samples: list[int], q: dict[int, float]) -> float:
    c = Counter(t for t in samples if t in q)
    n = sum(c.values())
    return 2.0 * sum(o * math.log(o / (n * q[t])) for t, o in c.items()) if n else 0.0


def g_null_quantile(q: dict[int, float], n: int, quantile: float, trials: int = 3000, seed: int = 0) -> float:
    rng = random.Random(seed)
    toks, ws = list(q), list(q.values())
    vals = sorted(g_stat(rng.choices(toks, ws, k=n), q) for _ in range(trials))
    return vals[min(trials - 1, int(quantile * trials))]


def evaluate(raw: dict, ref: dict, policies: dict, contract: dict, ref_accuracy: float | None = None) -> dict:
    """Derived verdict from raw observations + calibrated policy + task contract.

    lossless contract: tokens, logprobs, sampling and context all gate.
    lossy contract:    token identity and logprob error are reported, not gating (a quantised model
                       legitimately differs); task accuracy must stay within `max_drop` of the floor
                       build, the sampler must still behave like a sampler (looser policy), context gates.
    """
    lossy = bool(contract.get("lossy"))
    policy = policies["lossy" if lossy else "lossless"] if "lossless" in policies else policies
    eos = set(ref.get("eos_ids") or [])
    items = {it["id"]: it for it in ref["items"]}
    # observations made against an older prompt set: evaluate what both have, report the rest
    unknown = sorted({c["id"] for c in raw["greedy"]} - set(items)) + sorted(set(raw["samples"]) - set(items))
    traj = [compare_trajectory(items[c["id"]], c, eos, policy["tau"]) for c in raw["greedy"] if c["id"] in items]
    cnt = Counter(t["verdict"] for t in traj)
    n = len(traj)
    lp_err = sorted(e for t in traj for e in t.get("lp_err", []))
    kls = [k for t in traj for k in t.get("kl", [])]
    tokens = {"n": n, "match": cnt["match"], "tie": cnt["tie"], "hard": cnt["hard"], "error": cnt["error"],
              "hard_rate": cnt["hard"] / n if n else 1.0,
              "matched_frac": sum(t["match_len"] for t in traj) / max(1, sum(t["len"] for t in traj)),
              "bad_items": [{k: t.get(k) for k in ("id", "verdict", "match_len", "delta", "cand_tok", "error")}
                            for t in traj if t["verdict"] in ("hard", "error")]}
    tokens["pass"] = tokens["error"] == 0 and tokens["hard_rate"] <= policy["max_hard_rate"]
    lp = {"n": len(lp_err), "mae": sum(lp_err) / len(lp_err) if lp_err else None,
          "p99": lp_err[int(0.99 * (len(lp_err) - 1))] if lp_err else None,
          "kl_mean": sum(kls) / len(kls) if kls else None, "checked": contract.get("logprobs", True)}
    lp["pass"] = (not lp["checked"]) or (lp["mae"] is not None and lp["mae"] <= policy["max_lp_mae"]
                                          and lp["kl_mean"] <= policy["max_kl"])
    probes = {}
    for pid, toks in raw["samples"].items():
        if pid not in items:
            continue
        q, boundary = expected_dist(items[pid]["steps"][0], raw.get("sampling", SAMPLING))
        valid = [t for t in toks if t is not None and t != boundary]
        inside = [t for t in valid if t in q]
        oos = 1 - len(inside) / max(1, len(valid))
        g = g_stat(inside, q)
        g_lim = g_null_quantile(q, len(inside), policy["g_quantile"], seed=sum(map(ord, pid)))
        probes[pid] = {"n": len(valid), "n_outside": len(valid) - len(inside), "support": len(q),
                       "out_of_nucleus": round(oos, 4), "g": round(g, 2),
                       "g_limit": round(g_lim * policy["g_slack"], 2),
                       "pass": (len(valid) >= 0.9 * len(toks) and oos <= policy["max_out_of_nucleus_probe"]
                                and g <= g_lim * policy["g_slack"])}
    # Pooled over probes: a dropped top_p/top_k leaks a few % of mass outside the nucleus on every
    # probe, while boundary noise on a correct build stays well below 1% (calibration).
    n_all = sum(p["n"] for p in probes.values())
    pooled_oos = sum(p["n_outside"] for p in probes.values()) / max(1, n_all)
    sampling = {"probes": probes, "pooled_out_of_nucleus": round(pooled_oos, 4),
                "pass": bool(probes) and all(p["pass"] for p in probes.values())
                and pooled_oos <= policy["max_out_of_nucleus_pooled"]}
    ctx = raw.get("context")
    context = {"required": contract.get("max_context"), "pass": ctx is None or ctx["ok"], "detail": ctx}
    accuracy = None
    if contract.get("accuracy"):
        a = accuracy_of(raw.get("accuracy") or [])
        drop = contract["accuracy"].get("max_drop", 0.03)
        accuracy = {"accuracy": a["accuracy"], "n": a["n"], "reference": ref_accuracy, "max_drop": drop,
                    "pass": ref_accuracy is not None and a["n"] > 0 and a["accuracy"] >= ref_accuracy - drop}
    gates = [sampling["pass"], context["pass"]]
    gates += [accuracy["pass"]] if lossy else [tokens["pass"], lp["pass"]]
    if accuracy is not None and not lossy:
        gates.append(accuracy["pass"])
    if lossy:
        tokens["gating"] = lp["gating"] = False
    return {"pass": all(gates), "mode": "lossy" if lossy else "lossless", "tokens": tokens, "logprobs": lp,
            "sampling": sampling, "context": context, "accuracy": accuracy, "unknown_items": unknown,
            "trajectories": traj}
