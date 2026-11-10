"""Load a framework adapter from assets/adapters/<name>/adapter.py.

Contract (duck-typed; see assets/adapters/llama.cpp/adapter.py):
  meta, knobs
  build(src, log) -> {bin_dir}
  prepare_model(src, model_dir, out_dir, dtype, log) -> model file
  start(bin_dir, model, config, port, log, trace_out=None, gpu=0) -> Server ; stop(server)
  effective_config(server) -> {slots, kv_dtype, ...}
  bench_once(server, workload, repeat_idx, sampling) -> {metric: value}
  greedy(server, prompts, n_predict) -> [[token ids]]
  profile(server, workload, sampling, trace_out) -> {bench, nodes, top_kernels, ...}
  launch_script(bin_dir, model, config, port) -> str
"""
from __future__ import annotations

import importlib.util

from sherpa.util import ASSETS


def load(name: str, env: dict):
    path = ASSETS / "adapters" / name / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"sherpa_adapter_{name.replace('.', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Adapter(env)
