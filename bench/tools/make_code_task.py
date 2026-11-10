"""Turn a screened llama.cpp performance PR into a code task (human-run tool, writes into bench/).

    kernel/.venv/bin/python bench/tools/make_code_task.py <task-id> <sha>[+<fix>] <workload> <split>

workload: decode-c1 | chat-c32 | prefill-c8   (pick where screening showed the PR's gain)

  base      = parent of <sha>
  floor     = base with the task's fixed server args
  reference = base + expert patch (diff of the PR, plus the later commit that only fixes its build)
              stored as bench/deliverables/<task>/reference.diff so scoring never needs git history
  fixed     = args / env / cmake: a code task scores code, not configuration
"""
from __future__ import annotations

import sys

import yaml

from sherpa_kernel import llamacpp as L
from sherpa_kernel.util import BENCH

M = "rtx-5070-ti@tommao-pc4"
SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}
QWEN3_06B = {"hf": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "slug": "qwen3-0.6b",
             "reference": "oracle/references/qwen3-0.6b.json", "converter": "b12242"}
WORKLOADS = {
    "decode-c1": (dict(prompt_tokens=128, output_tokens=256, concurrency=1, num_requests=8, warmup_requests=2),
                  ["-np", "1", "-c", "4096", "-fit", "off", "-fa", "on"]),
    # shared KV pools sized 1.1 x (full workload + one 4096-token request): the context contract
    "chat-c32": (dict(prompt_tokens=256, output_tokens=128, concurrency=32, num_requests=128, warmup_requests=16),
                 ["-np", "32", "-kvu", "-c", "18432", "-fit", "off", "-fa", "on"]),
    "prefill-c8": (dict(prompt_tokens=2048, output_tokens=32, concurrency=8, num_requests=64, warmup_requests=8),
                   ["-np", "8", "-kvu", "-c", "23552", "-fit", "off", "-fa", "on"]),
}


def main(task_id: str, spec: str, workload: str, split: str) -> None:
    sha, _, fix = spec.partition("+")
    sha = L.resolve(sha)
    parent = L.resolve(f"{sha}^")
    info = L.commit_info(sha)
    patch = L.diff(parent, sha) + (L.diff(f"{fix}^", fix) if fix else "")
    wl, args = WORKLOADS[workload]
    metric = "req_per_s" if workload.startswith("prefill") else "output_tok_per_s"
    task = {"id": task_id, "kind": "code", "split": split,
            "description": f"Reproduce the speed-up of llama.cpp {sha[:9]}: {info['subject']}",
            "machine": M, "framework": {"name": "llama.cpp", "base": parent},
            "model": QWEN3_06B, "workload": {**wl, "sampling": SAMPLING, "vocab_hi": 100000},
            "objective": {"metric": metric, "higher_is_better": True},
            "contract": {"max_context": 4096, "logprobs": True, "model_dtypes": ["bf16"]},
            "fixed": {"args": args, "env": {}, "cmake": []},
            "upstream": {"commit": sha, "fix": L.resolve(fix) if fix else None, "date": info["date"],
                         "subject": info["subject"]},
            "measure": {"pairs": 3}}
    d = BENCH / "deliverables" / task_id
    d.mkdir(parents=True, exist_ok=True)
    (BENCH / "tasks" / f"{task_id}.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    common = {"framework": "llama.cpp", "base": parent, "model_dtype": "bf16", "args": args, "env": {}, "cmake": []}
    (d / "floor.yaml").write_text(yaml.safe_dump({**common, "source": "parent commit of the upstream PR"},
                                                 sort_keys=False))
    (d / "reference.diff").write_text(patch)
    (d / "reference.yaml").write_text(yaml.safe_dump(
        {**common, "patch_file": "reference.diff",
         "source": f"upstream {sha[:9]}{' + build fix ' + fix[:9] if fix else ''}: {info['subject']}"},
        sort_keys=False))
    print(task_id, parent[:10], len(patch.splitlines()), "patch lines")


if __name__ == "__main__":
    main(*sys.argv[1:5])
