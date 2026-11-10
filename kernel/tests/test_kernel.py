"""Unit tests for the judge's pure logic (no GPU, no network)."""
from __future__ import annotations

import math
import random

import pytest

from sherpa_kernel import guard, loadgen as G, oracle as O, records, stats


# ------------------------------------------------------------------------------------------ stats
def test_paired_ratio_direction():
    hib = stats.paired_ratio([110, 220], [100, 200], higher_is_better=True)
    lib = stats.paired_ratio([90, 180], [100, 200], higher_is_better=False)
    assert hib["gmean"] == pytest.approx(1.1)
    assert lib["gmean"] == pytest.approx(100 / 90)


def test_significance_uses_standard_error_and_floor():
    quiet = stats.paired_ratio([1.005, 1.006, 1.004], [1, 1, 1], True)
    assert not stats.significant(quiet)["significant"]          # 0.5% < 1% floor
    clear = stats.paired_ratio([1.20, 1.21, 1.19], [1, 1, 1], True)
    assert stats.significant(clear)["better"]
    noisy = stats.paired_ratio([1.3, 0.9, 1.25], [1, 1, 1], True)
    assert not stats.significant(noisy)["significant"]         # gain inside 2 x SE
    many = stats.paired_ratio([1.08, 1.02, 1.07, 1.03, 1.06, 1.04, 1.05, 1.05], [1] * 8, True)
    assert stats.significant(many)["better"]                    # more pairs shrink SE


def test_trustworthy_by_standard_error():
    assert stats.trustworthy(stats.summarize([1, 1.01, 0.99])) == []
    assert stats.trustworthy(stats.summarize([1, 1.01]))        # too few
    assert stats.trustworthy(stats.summarize([1, 1.5, 0.6]))    # SE too large
    observed = [0.556, 0.572, 0.537, 0.610, 0.624, 0.58]       # CV ~5.6% as observed on WSL2
    assert stats.trustworthy(stats.summarize(observed))         # SE 2.3% after 6 pairs: keep measuring
    assert stats.trustworthy(stats.summarize(observed + [0.57, 0.59, 0.58])) == []   # 9 pairs: SE < 2%


# ---------------------------------------------------------------------------------------- loadgen
def _req(n_out=8, chunks=None, prompt_n=16, err=None):
    chunks = chunks if chunks is not None else [10.0 + i for i in range(n_out)]
    return {"send_s": 0, "end_ms": chunks[-1] if chunks else 0, "chunk_ms": chunks, "n_chunks": len(chunks),
            "text_bytes": 3 * len(chunks),
            "usage": {"completion_tokens": n_out, "prompt_tokens": prompt_n}, "timings": {"prompt_n": prompt_n},
            "error": err}


WL = G.Workload(prompt_tokens=16, output_tokens=8, concurrency=2, num_requests=2, warmup_requests=0)


def test_metrics_from_raw():
    raw = {"wall_s": 2.0, "requests": [_req(), _req()]}
    m = G.metrics(raw, WL)
    assert m["output_tok_per_s"] == 8.0
    assert m["ttft_ms_mean"] == 10.0
    assert m["tpot_ms_mean"] == pytest.approx(1.0)


def test_contract_violations_detect_shortcuts():
    assert G.contract_violations({"requests": [_req(), _req()]}, WL) == []
    short = G.contract_violations({"requests": [_req(n_out=4, chunks=[1, 2, 3, 4])]}, WL)
    assert any("fewer than" in v for v in short)
    cached = G.contract_violations({"requests": [_req(prompt_n=2)]}, WL)
    assert any("skipped prefill" in v for v in cached)
    lying = G.contract_violations({"requests": [_req(n_out=32, chunks=[1.0, 2.0])]}, WL)  # 6 bytes for 32 tokens
    assert any("disagrees" in v for v in lying)
    failed = G.contract_violations({"requests": [_req(err="boom")]}, WL)
    assert any("failed" in v for v in failed)


def test_every_request_pins_every_sampler_parameter():
    """Model metadata must not change what a judge request means (Qwen2.5 repeat_penalty 1.1)."""
    body = G.sampling(**WL.sampling)
    assert set(G.NEUTRAL_SAMPLING) <= set(body)
    assert body["repeat_penalty"] == 1.0 and body["temperature"] == WL.sampling["temperature"]
    assert O.GREEDY["top_k"] == 1 and O.GREEDY["temperature"] == 0.0 and O.GREEDY["repeat_penalty"] == 1.0


def test_prompts_are_unique_and_deterministic():
    a = G.make_prompts(WL, 4, seed=1)
    assert a == G.make_prompts(WL, 4, seed=1)
    assert len({tuple(p) for p in a}) == 4
    assert a != G.make_prompts(WL, 4, seed=2)


# ----------------------------------------------------------------------------------------- oracle
def _step(ids, lps):
    return {"ids": ids, "lp": lps, "entropy": 0.0}


ITEM = {"id": "x", "kind": "text", "gen_ids": [5, 6, 7],
        "steps": [_step([5, 9], [-0.1, -3.0]), _step([6, 8], [-0.50, -0.52]), _step([7, 3], [-0.2, -4.0])]}


def _cand(ids, lp=None):
    lp = lp or [-0.1, -0.5, -0.2][: len(ids)]
    return {"ids": ids, "lp": lp, "top": [{i: v} for i, v in zip(ids, lp)], "finish": "length", "error": None}


def test_trajectory_match_tie_hard():
    assert O.compare_trajectory(ITEM, _cand([5, 6, 7]), set(), 0.2)["verdict"] == "match"
    tie = O.compare_trajectory(ITEM, _cand([5, 8, 1]), set(), 0.2)      # ref gap 0.02 at pos 1
    assert tie["verdict"] == "tie" and tie["match_len"] == 1
    hard = O.compare_trajectory(ITEM, _cand([9, 6, 7]), set(), 0.2)     # ref gap 2.9 at pos 0
    assert hard["verdict"] == "hard"
    unknown = O.compare_trajectory(ITEM, _cand([4, 6, 7]), set(), 0.2)  # not in ref top-k
    assert unknown["verdict"] == "hard" and math.isinf(unknown["delta"])


def test_comparison_survives_json_roundtrip():
    """Raw observations are stored as JSON and re-evaluated later: int dict keys become str."""
    import json
    cand = _cand([5, 6, 7])
    cand["top"] = [{5: -0.1, 9: -3.0}, {6: -0.5, 8: -0.52}, {7: -0.2, 3: -4.0}]
    before = O.compare_trajectory(ITEM, cand, set(), 0.2)
    after = O.compare_trajectory(ITEM, json.loads(json.dumps(cand)), set(), 0.2)
    assert before["kl"] == pytest.approx(after["kl"])
    assert max(after["kl"]) < 0.01


def test_evaluate_tolerates_observations_from_an_older_prompt_set():
    ref = {"eos_ids": [2], "items": [ITEM]}
    raw = {"greedy": [{**_cand([5, 6, 7]), "id": "x"}, {**_cand([1]), "id": "retired-prompt"}],
           "samples": {}, "context": None}
    pol = {"tau": 0.2, "max_hard_rate": 0.0, "max_lp_mae": 1.0, "max_kl": 1.0, "g_quantile": 0.999, "g_slack": 1.2,
           "max_out_of_nucleus_probe": 0.04, "max_out_of_nucleus_pooled": 0.01}
    ev = O.evaluate(raw, ref, pol, {})
    assert ev["unknown_items"] == ["retired-prompt"] and ev["tokens"]["n"] == 1


def test_trajectory_eos_and_early_stop():
    item = {**ITEM, "gen_ids": [5, 2, 7]}
    assert O.compare_trajectory(item, _cand([5]), {2}, 0.2)["verdict"] == "match"   # stopped where ref has EOS
    stopped = O.compare_trajectory(ITEM, {**_cand([5]), "finish": "stop"}, {2}, 0.2)
    assert stopped["verdict"] == "hard"                                              # stopped, ref did not
    capped = O.compare_trajectory(ITEM, {**_cand([5, 6]), "finish": "length"}, {2}, 0.2)
    assert capped["verdict"] == "hard" and capped["cand_tok"] == "<truncated>"      # server capped output
    # both emit EOS at position 1; the reference keeps generating past it, the server stops
    past_eos = {**ITEM, "gen_ids": [5, 2, 7], "steps": [ITEM["steps"][0], _step([2, 8], [-0.3, -1.0]), ITEM["steps"][2]]}
    assert O.compare_trajectory(past_eos, {**_cand([5, 2]), "finish": "stop"}, {2}, 0.2)["verdict"] == "match"


def test_align_probs_handles_merged_byte_tokens():
    toks = [10, 11, 12, 13]
    probs = [{"id": 10, "logprob": -1}, {"id": 12, "logprob": -2}, {"id": 13, "logprob": -3}]  # 11 merged away
    lp, _ = O.align_probs(toks, probs)
    assert lp == [-1, None, -2, -3]


def test_expected_dist_follows_llamacpp_chain():
    step = _step([1, 2, 3, 4], [math.log(0.5), math.log(0.3), math.log(0.15), math.log(0.05)])
    q, boundary = O.expected_dist(step, {"temperature": 1.0, "top_k": 40, "top_p": 0.9, "min_p": 0.0})
    assert set(q) == {1, 2, 3} and boundary == 4                     # 0.5+0.3+0.15 >= 0.9
    assert sum(q.values()) == pytest.approx(1.0)
    q_t, _ = O.expected_dist(step, {"temperature": 0.5, "top_k": 40, "top_p": 0.9, "min_p": 0.0})
    assert q_t[1] > q[1]                                             # temperature applied after top_p


def test_g_test_separates_correct_and_wrong_sampler():
    q = {1: 0.5, 2: 0.3, 3: 0.2}
    rng = random.Random(0)
    good = rng.choices(list(q), list(q.values()), k=500)
    flat = rng.choices(list(q), [1 / 3] * 3, k=500)
    lim = O.g_null_quantile(q, 500, 0.999)
    assert O.g_stat(good, q) < lim < O.g_stat(flat, q)


# ------------------------------------------------------------------------------------------ guard
def test_guard_rejects_judge_paths():
    guard.check_paths(["sherpa/runtime/engine.py", "assets/policies/plan.md"])
    for bad in (["kernel/sherpa_kernel/oracle.py"], ["bench/oracle/policy.yaml"], ["bench"],
                ["../kernel/x"], ["/etc/passwd"], ["sherpa/../kernel/x.py"]):
        with pytest.raises(guard.GuardViolation):
            guard.check_paths(bad)


def test_bench_seal_and_verify(tmp_path):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "a.yaml").write_text("id: a\n")
    assert guard.verify(tmp_path) == ["bench is not sealed (no MANIFEST.sha256)"]
    v1 = guard.seal(tmp_path)
    assert guard.verify(tmp_path) == []
    (tmp_path / "tasks" / "a.yaml").write_text("id: b\n")
    assert guard.verify(tmp_path) == ["modified: tasks/a.yaml"]
    (tmp_path / "tasks" / "new.yaml").write_text("x")
    assert "unsealed: tasks/new.yaml" in guard.verify(tmp_path)
    v2 = guard.seal(tmp_path)
    assert v2 != v1
    (tmp_path / "README.md").write_text("docs do not change what a score means")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.md").write_text("x")
    assert guard.verify(tmp_path) == [] and guard.seal(tmp_path) == v2


# ---------------------------------------------------------------------------------------- records
def test_records_append_only_and_artifact_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "root", lambda: tmp_path)
    rid = records.write("score", {"task": "t", "x": 1})
    ref = records.save_artifact(rid, "raw", {"a": [1, 2, 3]})
    assert records.load_artifact(ref) == {"a": [1, 2, 3]}
    records.write("score", {"task": "t", "x": 2})
    assert [r["x"] for r in records.read("score")] == [1, 2]
    assert records.get(rid)["judge"] == records.judge_version()
    import gzip
    p = tmp_path / ref["path"]
    p.write_bytes(gzip.compress(b'{"a":[9]}'))
    with pytest.raises(ValueError):
        records.load_artifact(ref)
