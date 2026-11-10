"""llama.cpp adapter: build / prepare model / deploy / bench / greedy / profile.

Everything framework-specific lives here; output formats follow sherpa.runtime.adapter contracts.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import requests

from sherpa.util import read_yaml, run

HERE = Path(__file__).parent

OP_PATTERNS = [
    ("op.gemm", re.compile(r"mul_mat|gemm|gemv|cublas|cutlass|xmma|mmq|mmvq|mmvf|Kernel2|splitKreduce", re.I)),
    ("op.attention", re.compile(r"flash_attn|fattn|soft_max", re.I)),
]


def classify_kernel(name: str) -> str:
    for node, pat in OP_PATTERNS:
        if pat.search(name):
            return node
    return "op.other"


class Server:
    def __init__(self, proc: subprocess.Popen, port: int, log: Path, cmd: list[str], env: dict):
        self.proc, self.port, self.log, self.cmd, self.env = proc, port, log, cmd, env
        self.url = f"http://127.0.0.1:{port}"

    def alive(self) -> bool:
        return self.proc.poll() is None


class Adapter:
    name = "llama.cpp"

    def __init__(self, env: dict):
        self.env = env
        self.meta = read_yaml(HERE / "adapter.yaml")
        self.knobs = read_yaml(HERE / "knobs.yaml")["knobs"]

    # ------------------------------------------------------------------ build
    def build(self, src: Path, log: Path) -> dict:
        cuda = self.env["cuda"]
        cc = self.env["gpus"][0]["compute_cap"].replace(".", "")
        args = list(self.meta["build"]["cmake_args"]) + [
            f"-DCUDAToolkit_ROOT={cuda['root']}", f"-DCMAKE_CUDA_COMPILER={cuda['nvcc']}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cc}",
        ]
        if cuda.get("pip_wheel"):
            # pip-wheel CUDA has no system linker path: transitive libcudart must be found explicitly (devlog ENV-2)
            lib = cuda["libdir"]
            args += [f"-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,{lib} -Wl,-rpath,{lib}",
                     f"-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,{lib}"]
        gen = ["-G", "Ninja"] if self.env["tools"].get("ninja") else []
        path_env = {"PATH": f"{Path(cuda['nvcc']).parent}:{os.environ['PATH']}"}
        t0 = time.time()
        run(["cmake", "-S", str(src), "-B", str(src / "build"), *gen, *args], env=path_env, log=log, timeout=1200)
        run(["cmake", "--build", str(src / "build"), "-j", str(max(1, (os.cpu_count() or 4) - 2)),
             "--target", *self.meta["build"]["targets"]], env=path_env, log=log, timeout=3600)
        return {"bin_dir": str(src / "build" / "bin"), "seconds": round(time.time() - t0, 1)}

    # ---------------------------------------------------------- model prepare
    def prepare_model(self, src: Path, model_dir: Path, out_dir: Path, dtype: str, log: Path) -> Path:
        out = out_dir / f"{dtype}.gguf"
        if out.exists():
            return out
        py = self.env.get("python_convert") or "python3"
        out_dir.mkdir(parents=True, exist_ok=True)
        if dtype in ("bf16", "f16", "f32", "q8_0"):
            run([py, str(src / "convert_hf_to_gguf.py"), str(model_dir), "--outtype", dtype, "--outfile", str(out)],
                log=log, timeout=3600)
        else:
            base = self.prepare_model(src, model_dir, out_dir, "bf16", log)
            run([str(src / "build/bin/llama-quantize"), str(base), str(out), dtype.upper()], log=log, timeout=3600)
        return out

    # ------------------------------------------------------------------ deploy
    @staticmethod
    def _validate_knob(name: str, value, spec: dict) -> None:
        """Enforce the types declared in knobs.yaml: catch bad values before they reach a measurement.

        Declared metadata that nothing checks is metadata that does not exist (devlog DES-22).
        """
        t = spec["type"]
        if t == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"knob '{name}' expects int, got {value!r} ({type(value).__name__})")
        elif t == "bool":
            if not isinstance(value, bool):
                raise TypeError(f"knob '{name}' expects bool, got {value!r}")
        elif t == "enum":
            allowed = spec.get("values") or []
            if value not in allowed:
                raise ValueError(f"knob '{name}' expects one of {allowed}, got {value!r}")

    def command(self, bin_dir: Path, model: Path, config: dict, port: int) -> tuple[list[str], dict]:
        cmd = [str(Path(bin_dir) / "llama-server"), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "999", "--no-webui", "--metrics"]
        env = {}
        for k, v in (config or {}).items():
            spec = self.knobs.get(k)
            if spec is None:
                raise KeyError(f"unknown knob {k}")
            self._validate_knob(k, v, spec)
            if spec["type"] == "bool":
                if "env_off" in spec:
                    if not v:
                        env.update(spec["env_off"])
                elif v and spec.get("flag_on"):
                    cmd.append(spec["flag_on"])
                elif not v and spec.get("flag_off"):
                    cmd.append(spec["flag_off"])
            else:
                cmd += [spec["flag"], str(v)]
        return cmd, env

    def start(self, bin_dir: Path, model: Path, config: dict, port: int, log: Path,
              trace_out: Path | None = None, gpu: int = 0, timeout: float = 180) -> Server:
        cmd, env = self.command(bin_dir, model, config, port)
        env = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        if trace_out:
            env["CUDA_INJECTION64_PATH"] = self.env["tracer_lib"]
            env["SHERPA_TRACE_OUT"] = str(trace_out)
        log.parent.mkdir(parents=True, exist_ok=True)
        f = open(log, "w")
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        srv = Server(proc, port, log, cmd, {k: v for k, v in env.items() if k.startswith(("GGML", "LLAMA", "CUDA_"))})
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not srv.alive():
                raise RuntimeError(f"llama-server exited early, see {log}:\n" + "\n".join(log.read_text().splitlines()[-20:]))
            try:
                if requests.get(srv.url + "/health", timeout=2).status_code == 200:
                    return srv
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.stop(srv)
        raise TimeoutError(f"llama-server not healthy in {timeout}s")

    def stop(self, srv: Server) -> None:
        if srv.alive():
            srv.proc.send_signal(signal.SIGINT)
            try:
                srv.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.proc.kill()
                srv.proc.wait()

    def effective_config(self, srv: Server) -> dict:
        """Resolve 'auto' values to what the server actually uses."""
        p = requests.get(srv.url + "/props", timeout=5).json()
        log = srv.log.read_text()
        eff = {"slots": p.get("total_slots"), "n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx")}
        m = re.search(r"flash_attn\s*=\s*(\w+)", log)
        eff["flash_attn"] = m.group(1) if m else "unresolved"   # resolved later from the kernel trace
        m = re.search(r"type_k = '(\w+)'", log) or re.search(r"K \((\w+)\)", log)
        eff["kv_dtype"] = m.group(1) if m else "f16"
        eff["cuda_graphs"] = "graphs reused" in log or "GGML_CUDA_DISABLE_GRAPHS" not in srv.env
        return eff

    # ------------------------------------------------------------------- bench
    @staticmethod
    def _prompts(wl, n: int, offset: int = 0) -> list[list[int]]:
        rng = random.Random(wl.seed + offset)
        return [[rng.randrange(1000, 100000) for _ in range(wl.prompt_tokens)] for _ in range(n)]

    async def _run_load(self, srv: Server, wl, prompts: list[list[int]], sampling: dict) -> dict:
        sem = asyncio.Semaphore(wl.concurrency)
        results = []
        timeout = aiohttp.ClientTimeout(total=1800)

        async with aiohttp.ClientSession(timeout=timeout) as s:
            async def one(i, p):
                body = {"prompt": p, "n_predict": wl.output_tokens, "ignore_eos": True, "cache_prompt": False,
                        "seed": wl.seed + i, **sampling}
                async with sem:
                    t0 = time.perf_counter()
                    async with s.post(srv.url + "/completion", json=body) as r:
                        d = await r.json()
                    t1 = time.perf_counter()
                tm = d.get("timings", {})
                results.append({"e2e_ms": (t1 - t0) * 1000, "n_out": d.get("tokens_predicted", 0),
                                "decode_ms": tm.get("predicted_ms", 0.0), "prompt_ms": tm.get("prompt_ms", 0.0),
                                "tpot_ms": tm.get("predicted_per_token_ms", 0.0)})

            t0 = time.perf_counter()
            await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
            wall = time.perf_counter() - t0
        return {"wall_s": wall, "results": results}

    def bench_once(self, srv: Server, wl, repeat_idx: int, sampling: dict, on_start=None, on_end=None) -> dict:
        warm = self._prompts(wl, wl.warmup_requests, offset=10_000)
        asyncio.run(self._run_load(srv, wl, warm, sampling))
        prompts = self._prompts(wl, wl.num_requests, offset=repeat_idx * 1000)
        if on_start:
            on_start()
        r = asyncio.run(self._run_load(srv, wl, prompts, sampling))
        if on_end:
            on_end()
        rs = r["results"]
        n_out = sum(x["n_out"] for x in rs)
        e2e = sorted(x["e2e_ms"] for x in rs)
        ttft = sorted(x["e2e_ms"] - x["decode_ms"] for x in rs)  # approx: queue + prefill (no streaming)
        pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]
        return {"output_tok_per_s": n_out / r["wall_s"], "req_per_s": len(rs) / r["wall_s"],
                "tpot_ms": sum(x["tpot_ms"] for x in rs) / len(rs), "ttft_ms_mean": sum(ttft) / len(ttft),
                "ttft_ms_p99": pct(ttft, 0.99), "e2e_ms_p50": pct(e2e, 0.5), "e2e_ms_p99": pct(e2e, 0.99),
                "total_output_tokens": n_out, "wall_s": r["wall_s"]}

    def greedy(self, srv: Server, prompts: list[str], n_predict: int = 48) -> list[list[int]]:
        out = []
        for p in prompts:
            d = requests.post(srv.url + "/completion", json={
                "prompt": p, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": False,
                "return_tokens": True}, timeout=300).json()
            out.append(d.get("tokens") or [])
        return out

    # ----------------------------------------------------------------- profile
    def profile(self, srv: Server, wl, sampling: dict, trace_out: Path) -> dict:
        if trace_out.exists():
            trace_out.unlink()
        marks = {}

        def start():
            marks["t0"] = time.time_ns()
            os.kill(srv.proc.pid, signal.SIGUSR1)
            time.sleep(0.2)

        def end():
            os.kill(srv.proc.pid, signal.SIGUSR2)
            marks["t1"] = time.time_ns()

        m = self.bench_once(srv, wl, repeat_idx=99, sampling=sampling, on_start=start, on_end=end)
        for _ in range(300):
            if trace_out.exists():
                break
            time.sleep(0.1)
        if not trace_out.exists():
            raise RuntimeError("trace file not produced (tracer not loaded?)")
        return {"bench": m, **self.parse_trace(trace_out, m), "trace": str(trace_out)}

    @staticmethod
    def parse_trace(path: Path, bench: dict) -> dict:
        names, kern, mem = {}, [], []
        dropped = 0
        with open(path) as f:
            for line in f:
                t = line[0]
                if t == "K":
                    _, nid, s, e, _g, _st = line.split()
                    kern.append((int(s), int(e), int(nid)))
                elif t == "M" or t == "S":
                    p = line.split()
                    mem.append((int(p[-2]), int(p[-1])))
                elif t == "N":
                    _, nid, name = line.rstrip("\n").split(" ", 2)
                    names[int(nid)] = name
                elif t == "E":
                    dropped = int(line.split()[1])
        ivs = sorted([(s, e) for s, e, _ in kern] + mem)
        if not ivs:
            raise RuntimeError("empty trace")
        busy, cs, ce = 0, ivs[0][0], ivs[0][1]
        for s, e in ivs[1:]:
            if s > ce:
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        window = max(e for _, e in ivs) - ivs[0][0]
        per_op = defaultdict(int)
        top = defaultdict(lambda: [0, 0])
        for s, e, nid in kern:
            per_op[classify_kernel(names.get(nid, ""))] += e - s
            top[nid][0] += 1
            top[nid][1] += e - s
        per_op["op.memcpy"] = sum(e - s for s, e in mem)
        raw_sum = sum(per_op.values()) or 1
        n_tok = bench["total_output_tokens"]
        to_ms_tok = lambda ns: ns / 1e6 / n_tok
        nodes = {k: to_ms_tok(v * busy / raw_sum) for k, v in per_op.items()}  # scale op sums to the busy union
        for k in ("op.gemm", "op.attention", "op.other", "op.memcpy"):
            nodes.setdefault(k, 0.0)
        nodes["step.gpu_busy"] = to_ms_tok(busy)
        nodes["step.gpu_idle"] = to_ms_tok(window - busy)
        nodes["step.gpu_window"] = to_ms_tok(window)
        nodes["request.time_per_token"] = 1000.0 / bench["output_tok_per_s"]
        nodes["request.outside_gpu_window"] = max(0.0, nodes["request.time_per_token"] - nodes["step.gpu_window"])
        top_k = sorted(top.items(), key=lambda x: -x[1][1])[:12]
        used = {names.get(n, "") for n in top}
        observed = {"flash_attn": "on" if any("flash_attn" in u for u in used) else "off",
                    "cuda_graphs_kernels_pct": None}
        return {"nodes": nodes, "observed_config": observed, "dropped_records": dropped, "n_kernels": len(kern),
                "top_kernels": [{"name": names.get(n, "?")[:120], "node": classify_kernel(names.get(n, "")),
                                 "count": c, "ms_per_tok": to_ms_tok(t)} for n, (c, t) in top_k]}

    # ---------------------------------------------------------------- deliver
    def launch_script(self, bin_dir: Path, model: Path, config: dict, port: int = 8080) -> str:
        cmd, env = self.command(bin_dir, model, config, port)
        cmd[cmd.index("--host") + 1] = "0.0.0.0"
        envs = "".join(f"export {k}={v}\n" for k, v in env.items())
        return "#!/usr/bin/env bash\nset -e\n" + envs + "exec " + " ".join(cmd) + ' "$@"\n'
