"""Minimal llama.cpp driver owned by the judge: build a deliverable, prepare the model, run the server.

The judge builds and launches deliverables itself instead of trusting whatever Sherpa built: the
only inputs it accepts are a base commit, an optional patch, build flags, server args and env.
Everything is content-addressed so that two scorings of the same deliverable use the same binary.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import machine
from .util import DATA, JUDGE_DATA, now_iso, read_json, run, sha256_bytes, sha256_file, stable_hash, write_json

REPO_URL = "https://github.com/ggml-org/llama.cpp"
BARE = DATA / "repos" / "llama.cpp.git"
BASE_CMAKE = ["-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF",
              "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"]
TARGETS = ["llama-server", "llama-bench", "llama-quantize", "llama-perplexity"]
# Server args the judge owns. A deliverable may not set them (model, network binding, caching to disk).
RESERVED_ARGS = {"-m", "--model", "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "--host", "--port",
                 "--path", "--api-key", "--api-key-file", "--slot-save-path", "-a", "--alias"}


# ----------------------------------------------------------------------------------------- source
def ensure_repo() -> Path:
    if not BARE.exists():
        BARE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--bare", REPO_URL, str(BARE)], timeout=3600)
    return BARE


def resolve(ref: str) -> str:
    ensure_repo()
    r = run(["git", "--git-dir", str(BARE), "rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    if r.returncode != 0:
        run(["git", "--git-dir", str(BARE), "fetch", "--tags", "origin", "+refs/heads/*:refs/heads/*"], timeout=1800)
        r = run(["git", "--git-dir", str(BARE), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    return r.stdout.strip()


def diff(parent: str, commit: str) -> str:
    return run(["git", "--git-dir", str(BARE), "diff", "--binary", parent, commit]).stdout


def commit_info(commit: str) -> dict:
    out = run(["git", "--git-dir", str(BARE), "show", "-s", "--format=%H%n%cI%n%s", commit]).stdout.splitlines()
    return {"sha": out[0], "date": out[1], "subject": out[2] if len(out) > 2 else ""}


# ------------------------------------------------------------------------------------------ build
@dataclass
class Build:
    key: str
    src: Path
    bin_dir: Path
    info: dict

    def exe(self, name: str) -> Path:
        return self.bin_dir / name


# Toolchain variables injected by conda / pip build environments. Leaking them into the build mixes
# a foreign binutils with the system glibc (observed: conda's ld failing on libcublas / GLIBC_2.34).
_FOREIGN_BUILD_VARS = {"CC", "CXX", "CPP", "LD", "AR", "NM", "RANLIB", "STRIP", "OBJCOPY", "OBJDUMP", "READELF",
                       "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "DEBUG_CFLAGS", "DEBUG_CXXFLAGS", "CMAKE_ARGS",
                       "CMAKE_PREFIX_PATH", "CONDA_BUILD_SYSROOT", "HOST", "BUILD", "GCC", "GXX", "CUDAHOSTCXX",
                       "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"}


def _hermetic_env() -> dict:
    """os.environ without foreign toolchains; PATH limited to system dirs + CUDA."""
    cu = machine.cuda()
    env = {k: v for k, v in os.environ.items() if k not in _FOREIGN_BUILD_VARS and not k.startswith("CONDA")}
    env["PATH"] = f"{Path(cu['nvcc']).parent}:/usr/local/bin:/usr/bin:/bin"
    env["CCACHE_BASEDIR"] = str(JUDGE_DATA / "src")
    env["CCACHE_NOHASHDIR"] = "true"
    return env


def _cmake_args() -> list[str]:
    cu = machine.cuda()
    if not cu:
        raise RuntimeError("no CUDA toolkit found (set SHERPA_CUDA_ROOT)")
    cc = machine.gpus()[0]["compute_cap"].replace(".", "")
    # Compilers come only from explicit overrides or the system: never from whatever env is active.
    c = os.environ.get("SHERPA_CC", "/usr/bin/gcc")
    cxx = os.environ.get("SHERPA_CXX", "/usr/bin/g++")
    args = [f"-DCMAKE_C_COMPILER={c}", f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
            f"-DCMAKE_CUDA_COMPILER={cu['nvcc']}", f"-DCUDAToolkit_ROOT={cu['root']}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cc}", "-DGGML_CCACHE=OFF", "-DLLAMA_CCACHE=OFF"]
    if machine.tool("ninja"):
        args.append(f"-DCMAKE_MAKE_PROGRAM={machine.tool('ninja')}")
    if machine.tool("ccache"):
        args += [f"-DCMAKE_{lang}_COMPILER_LAUNCHER={machine.tool('ccache')}" for lang in ("C", "CXX", "CUDA")]
    return args


def _toolchain_id() -> dict:
    """What the build cache key depends on besides the source: compilers and CUDA versions."""
    c = os.environ.get("SHERPA_CXX", "/usr/bin/g++")
    return {"cuda": machine.cuda()["version"], "cxx": run([c, "--version"]).stdout.splitlines()[0],
            "arch": machine.gpus()[0]["compute_cap"]}


def build(commit: str, patch: str = "", extra_cmake: list[str] | None = None, log_dir: Path | None = None) -> Build:
    """Build (commit + patch + flags). Cached by content; failures raise with the log tail."""
    commit = resolve(commit)
    extra_cmake = list(extra_cmake or [])
    arch_args, env = _cmake_args(), _hermetic_env()
    key_obj = {"commit": commit, "patch": sha256_bytes(patch.encode()) if patch else None,
               "cmake": BASE_CMAKE + extra_cmake, "toolchain": _toolchain_id(), "targets": TARGETS}
    key = stable_hash(key_obj)[:16]
    src = JUDGE_DATA / "src" / key
    marker = src / ".sherpa-build.json"
    if marker.exists():
        info = read_json(marker)
        return Build(key, src, Path(info["bin_dir"]), info)
    with machine.exclusive(f"build {key}"):
        if marker.exists():  # built by another process while we waited
            info = read_json(marker)
            return Build(key, src, Path(info["bin_dir"]), info)
        return _build_locked(key, key_obj, commit, patch, extra_cmake, arch_args, env, src, marker, log_dir)


def _build_locked(key, key_obj, commit, patch, extra_cmake, arch_args, env, src, marker, log_dir) -> Build:
    if src.exists():  # stale half-built tree
        run(["git", "--git-dir", str(BARE), "worktree", "remove", "--force", str(src)], check=False)
        shutil.rmtree(src, ignore_errors=True)
    run(["git", "--git-dir", str(BARE), "worktree", "prune"], check=False)
    src.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "--git-dir", str(BARE), "worktree", "add", "--detach", str(src), commit])
    log = (log_dir or src) / f"build-{key}.log"
    if patch:
        pf = src / ".sherpa-patch.diff"
        pf.write_text(patch)
        r = run(["git", "apply", "--whitespace=nowarn", str(pf)], cwd=src, log=log, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"patch does not apply on {commit[:12]} (see {log})")
    gen = ["-G", "Ninja"] if machine.tool("ninja") else []
    cmake = machine.tool("cmake") or "cmake"
    t0 = time.time()
    run([cmake, "-S", str(src), "-B", str(src / "build"), *gen, *BASE_CMAKE, *arch_args, *extra_cmake],
        base_env=env, log=log, timeout=1800)
    jobs = str(max(1, (os.cpu_count() or 4) - 2))
    run([cmake, "--build", str(src / "build"), "-j", jobs, "--target", "llama-server"],
        base_env=env, log=log, timeout=7200)
    # Older commits may lack some targets; build what exists.
    for t in TARGETS[1:]:
        run([cmake, "--build", str(src / "build"), "-j", jobs, "--target", t],
            base_env=env, log=log, timeout=7200, check=False)
    bin_dir = src / "build" / "bin"
    if not (bin_dir / "llama-server").exists():
        raise RuntimeError(f"llama-server not produced (see {log})")
    info = {**key_obj, "key": key, "bin_dir": str(bin_dir), "seconds": round(time.time() - t0, 1),
            "built_at": now_iso(), "server_sha256": sha256_file(bin_dir / "llama-server"), "log": str(log)}
    write_json(marker, info)
    return Build(key, src, bin_dir, info)


# ------------------------------------------------------------------------------------------ model
def convert_python() -> str:
    """Python used to run convert_hf_to_gguf.py (needs torch + transformers + sentencepiece)."""
    for py in [os.environ.get("SHERPA_CONVERT_PYTHON"), str(Path.home() / "miniconda3/envs/vllm/bin/python")]:
        if py and Path(py).exists():
            return py
    raise RuntimeError("set SHERPA_CONVERT_PYTHON to a python with torch + transformers")


def prepare_model(hf_dir: Path, slug: str, dtype: str, converter: Build, log: Path) -> dict:
    """HF snapshot -> GGUF with a pinned converter. Returns {path, sha256, dtype}."""
    out_dir = JUDGE_DATA / "models" / f"{slug}--conv-{converter.info['commit'][:10]}"
    out = out_dir / f"{dtype}.gguf"
    meta = out.with_suffix(".json")
    if meta.exists() and out.exists():
        return read_json(meta)
    with machine.exclusive(f"convert {slug} {dtype}"):   # CPU-heavy: must not overlap a measurement
        if meta.exists() and out.exists():
            return read_json(meta)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "bf16.gguf"
        if not base.exists():
            tmp = base.with_suffix(".part")
            run([convert_python(), str(converter.src / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16",
                 "--outfile", str(tmp)], log=log, timeout=3600)
            tmp.replace(base)
        if dtype != "bf16":
            tmp = out.with_suffix(".part")
            run([str(converter.exe("llama-quantize")), str(base), str(tmp), dtype.upper()], log=log, timeout=3600)
            tmp.replace(out)
        m = {"path": str(out), "sha256": sha256_file(out), "dtype": dtype, "hf_dir": str(hf_dir),
             "converter_commit": converter.info["commit"]}
        write_json(meta, m)
        return m


# ----------------------------------------------------------------------------------------- server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Server:
    proc: subprocess.Popen
    port: int
    log: Path
    cmd: list[str]
    lock: object = None          # machine.exclusive() held for the server's whole lifetime
    quiet: dict | None = None    # machine.wait_quiet() result taken before the server started

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if self.lock is not None:
            lk, self.lock = self.lock, None
            lk.__exit__(None, None, None)

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *a) -> None:
        self.stop()


def check_args(args: list[str]) -> None:
    for a in args:
        if a.split("=", 1)[0] in RESERVED_ARGS:
            raise ValueError(f"server arg '{a}' is reserved by the judge")


def start(b: Build, model_path: str, args: list[str], env: dict, log: Path, gpu: int = 0,
          timeout: float = 300) -> Server:
    check_args(args)
    lock = machine.exclusive(f"server {log.name}")
    lock.__enter__()
    try:
        quiet = machine.wait_quiet()
        port = free_port()
        cmd = [str(b.exe("llama-server")), "-m", model_path, "--host", "127.0.0.1", "--port", str(port), *args]
        full_env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}, "CUDA_VISIBLE_DEVICES": str(gpu)}
        log.parent.mkdir(parents=True, exist_ok=True)
        f = open(log, "w")
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env, start_new_session=True)
    except BaseException:
        lock.__exit__(None, None, None)
        raise
    srv = Server(proc, port, log, cmd, lock, quiet)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not srv.alive():
            srv.stop()
            raise RuntimeError(f"llama-server exited early:\n" + "\n".join(log.read_text().splitlines()[-25:]))
        try:
            with urllib.request.urlopen(srv.url + "/health", timeout=2) as r:
                if r.status == 200:
                    return srv
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.3)
    srv.stop()
    raise TimeoutError(f"llama-server not healthy after {timeout}s (see {log})")
