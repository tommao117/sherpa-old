"""Offline command drafts and log evidence for one inspected llama.cpp revision.

No subprocesses, file reads, device probes, or execution grants are produced here.
"""
from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
import re

from ..runtime.execution_contract import Checks, SHA256, _bounded
from ..runtime.models import Task
from ..runtime.store import digest

VERSION = "llama-completion-v1"
FRAMEWORK_URL = "https://github.com/ggml-org/llama.cpp"
SUPPORTED_COMMIT = "0fb8f157702702f33f4e018de8701b2569a1bc28"
ADAPTER = "llama-completion"
OFF_OPTIONS = (
    "LLAMA_BUILD_SERVER", "LLAMA_BUILD_APP", "LLAMA_BUILD_TESTS", "LLAMA_BUILD_EXAMPLES",
    "LLAMA_BUILD_UI", "LLAMA_USE_PREBUILT_UI", "LLAMA_CONNECT", "LLAMA_LLGUIDANCE",
    "LLAMA_OPENSSL", "LLAMA_BUILD_BORINGSSL", "LLAMA_BUILD_LIBRESSL", "LLAMA_SUBPROCESS",
    "LLAMA_TOOLS_INSTALL", "GGML_NATIVE", "GGML_CPU_KLEIDIAI",
)
RUNTIME_CHECKS = (
    "explicit_execution_grant_and_budget",
    "prepare_source_commit_and_clean_workspace",
    "registered_tool_versions_and_concrete_commands",
    "fresh_empty_build_scratch_home_and_caches",
    "system_config_absent_or_isolated:/etc/llama.cpp/config.ini",
    "selected_model_content_identity_and_compatibility",
    "successful_build_call_and_binary_identity",
    "current_gpu_uuid_backend_mapping_and_exclusive_availability",
    "memory_disk_output_and_process_supervision",
)


def _path(value):
    return (isinstance(value, str) and value.startswith("/") and not value.startswith("//") and len(value) <= 4096
            and not any(ch in value for ch in ("\x00", "\n", "\r", "\\", ";"))
            and ".." not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value)


def _inside(child, parent):
    return child != parent and PurePosixPath(parent) in PurePosixPath(child).parents


def _overlap(left, right):
    return left == right or _inside(left, right) or _inside(right, left)


def plan_completion(task, locations):
    """Convert declared inputs to non-executable configure/build/request drafts.

    The pure planner can run before E1/approval. Its output must not bypass E1
    or the runtime checks, even when every command argument is well formed.
    """
    checks = Checks()
    commands = []
    task_hash = None
    try:
        t = _bounded(task.snapshot() if isinstance(task, Task) else task)
        loc = _bounded(locations)
        if not isinstance(t, dict) or not isinstance(loc, dict):
            raise ValueError("task and locations must be JSON objects")
        task_hash = digest(t)
        target, model, resources = (t.get(key) for key in ("target", "model", "resources"))
        execution = t.get("contract", {}).get("execution", {})
        adapter = execution.get("adapter", {})
        limits = execution.get("limits", {})
        request = t.get("workload", {}).get("request", {})
        params = request.get("parameters", {})
        selected = {}

        def identity():
            obj = checks.object(target, "task.target", ("kind", "url"), ("revision",))
            if obj is not None:
                checks.require(obj.get("kind") == "git" and obj.get("url") == FRAMEWORK_URL,
                               "task.target", "unsupported_source", "This profile is for the inspected upstream only", "Keep the user-selected source; add a separately reviewed profile for another source")
            framework = checks.object(execution.get("framework"), "execution.framework", ("commit", "prepare_run", "prepare_call", "prepare_output_hash"))
            if framework is not None:
                checks.require(framework.get("commit") == SUPPORTED_COMMIT, "execution.framework.commit", "unsupported_revision",
                               "Command profile does not cover this revision", "Inspect that revision before selecting its CLI/options")
            checks.object(adapter, "execution.adapter", ("name", "version", "build_parameters"))
            checks.require(adapter.get("name") == ADAPTER and adapter.get("version") == "1", "execution.adapter", "unsupported_adapter",
                           "Select the versioned completion adapter", "Use llama-completion version 1 explicitly")
            build = checks.object(adapter.get("build_parameters"), "execution.adapter.build_parameters", ("cuda_arch",))
            if build is not None:
                value = build.get("cuda_arch")
                checks.require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{1,2}", value) is not None,
                               "execution.adapter.build_parameters.cuda_arch", "explicit_cuda_arch_required",
                               "A numeric CUDA architecture is required; native/auto are not allowed", "Supply the intended architecture without probing the GPU here")

        def paths():
            if checks.object(loc, "locations", ("run_root", "source", "build", "scratch", "cmake", "make", "cc", "cxx", "nvcc")) is None:
                return
            for key in ("run_root", "source", "build", "scratch", "cmake", "make", "cc", "cxx", "nvcc"):
                checks.require(_path(loc.get(key)), f"locations.{key}", "invalid_path", "Normalized absolute path required; no CMake list delimiters", "Use an explicit local path")
            if all(_path(loc.get(k)) for k in ("run_root", "source", "build", "scratch")):
                for key in ("source", "build", "scratch"):
                    checks.require(_inside(loc[key], loc["run_root"]), f"locations.{key}", "outside_run", "Workspace and output must be under the run root", "Use run-owned directories")
                for a, b in (("source", "build"), ("source", "scratch"), ("build", "scratch")):
                    checks.require(not _overlap(loc[a], loc[b]), f"locations.{a}", "overlapping_paths", "Source/build/scratch must be disjoint", "Use separate run-local directories")
                for key in ("cmake", "make", "cc", "cxx", "nvcc"):
                    if _path(loc.get(key)):
                        checks.require(not _overlap(loc[key], loc["run_root"]), f"locations.{key}", "mutable_tool_path", "Toolchain must not be in generated run outputs", "Use independently verified tool paths")

        def files():
            obj = checks.object(model, "task.model", ("kind", "path", "selected_set", "sets"))
            if obj is None:
                return
            checks.require(obj.get("kind") == "local" and _path(obj.get("path")), "task.model.path", "invalid_model_root", "A local absolute model directory is required", "Use the selected local model root")
            sets = obj.get("sets")
            if not checks.require(isinstance(sets, list) and bool(sets), "task.model.sets", "sets_required", "Explicit file sets required", "Keep alternative quantizations in separate sets"):
                return
            ids = [v.get("id") for v in sets if isinstance(v, dict)]
            choice = obj.get("selected_set")
            valid_ids = len(ids) == len(sets) and all(isinstance(v, str) and bool(v) for v in ids)
            valid_ids = valid_ids and len(set(ids)) == len(ids)
            if not checks.require(valid_ids and isinstance(choice, str) and choice in ids,
                                  "task.model.selected_set", "selection_required", "One unambiguous file set must be chosen", "Select a set explicitly"):
                return
            selected.update(sets[ids.index(choice)])
            checks.object(selected, "selected_set", ("id", "layout", "weights", "support"))
            weights = selected.get("weights")
            if not checks.require(selected.get("layout") == "single" and isinstance(weights, list) and len(weights) == 1 and selected.get("support") == [],
                                  "selected_set", "unsupported_layout", "Initial completion profile supports one self-contained GGUF only", "Keep shards/support files for a separately validated adapter"):
                return
            file = checks.object(weights[0], "selected_set.weights[0]", ("path", "sha256", "bytes"))
            if file is not None:
                p = file.get("path")
                valid = (isinstance(p, str) and bool(p) and len(p) <= 512 and not PurePosixPath(p).is_absolute()
                         and str(PurePosixPath(p)) == p and not any(ch in p for ch in ("\x00", "\n", "\r", "\\"))
                         and all(not x.startswith(".") for x in PurePosixPath(p).parts) and p.endswith(".gguf"))
                checks.require(valid, "selected_set.weights[0].path", "invalid_model_file", "Expected one normalized model-relative GGUF path", "Do not use a remote URL, traversal or a directory")
                checks.require(isinstance(file.get("sha256"), str) and SHA256.fullmatch(file["sha256"]) is not None,
                               "selected_set.weights[0].sha256", "content_identity_required", "Content digest declaration required", "Record it after explicit permission to inspect weights")
                checks.number(file.get("bytes"), "selected_set.weights[0].bytes", minimum=1, integer=True)
            if _path(obj.get("path")) and _path(loc.get("run_root")):
                checks.require(not _overlap(obj["path"], loc["run_root"]), "task.model.path", "input_output_overlap", "Model root and run root must not overlap", "Keep user inputs outside run outputs")

        def options():
            obj = checks.object(resources, "task.resources", ("device", "gpus"))
            if obj is not None:
                gpus = obj.get("gpus")
                checks.require(obj.get("device") == "gpu" and isinstance(gpus, list) and len(gpus) == 1
                               and isinstance(gpus[0], str) and re.fullmatch(r"GPU-[0-9a-fA-F-]{8,64}", gpus[0]) is not None,
                               "task.resources", "single_gpu_required", "This profile needs one explicitly selected GPU UUID", "Confirm the UUID separately; no device discovery occurs here")
            checks.object(request, "task.workload.request", ("interface", "input", "parameters"))
            checks.require(request.get("interface") == "cli", "task.workload.request.interface", "unsupported_interface", "Completion is a one-shot CLI", "Do not infer probability/HTTP behavior from text completion")
            checks.text(request.get("input"), "task.workload.request.input", 16000)
            checks.object(params, "task.workload.request.parameters", ("seed", "max_tokens", "ctx_size", "gpu_layers", "temperature"))
            for key, minimum, maximum in (("seed", 0, 4294967294), ("max_tokens", 2, 128), ("ctx_size", 32, 4096), ("gpu_layers", 1, 999)):
                value = params.get(key)
                checks.require(type(value) is int and minimum <= value <= maximum, f"request.parameters.{key}", "invalid_parameter",
                               f"Expected an integer in [{minimum}, {maximum}]", "Supply explicit bounded smoke parameters")
            checks.require(type(params.get("temperature")) in (int, float) and params["temperature"] == 0,
                           "request.parameters.temperature", "deterministic_sampling_required", "This smoke profile uses temperature 0", "Set temperature to 0")
            if type(params.get("ctx_size")) is int and type(params.get("max_tokens")) is int:
                checks.require(params["max_tokens"] < params["ctx_size"], "request.parameters", "context_conflict", "Generation must fit inside context", "Leave room for the prompt; actual tokenization must be checked at runtime")
            for key in ("cpu_threads", "memory_mib", "gpu_memory_mib", "disk_mib", "output_mib"):
                checks.number(limits.get(key), f"execution.limits.{key}", minimum=1, integer=True)
            checks.number(limits.get("step_seconds"), "execution.limits.step_seconds", minimum=0.001)
            acceptance = execution.get("acceptance")
            expected = {"scope": "smoke", "parser": {"name": ADAPTER, "version": "1"},
                        "verifier": {"name": "gpu-smoke", "version": "1"}, "reference": None, "tolerances": None}
            checks.require(acceptance == expected, "execution.acceptance", "unsupported_acceptance", "Only the explicit GPU smoke contract is supported", "Keep probability/score behavior as a separate validation")
            expected_permissions = {"network": "none", "write_scopes": ["build", "scratch"],
                                    "tools": ["cmake", "make", "cc", "cxx", "nvcc", "llama-completion"], "trusted_code": True}
            checks.require(execution.get("permissions") == expected_permissions, "execution.permissions", "unsupported_permissions", "Use the fixed local tool and output scope", "Declare this profile's exact permissions; execution still requires approval")

        checks.run("profile", [], identity)
        checks.run("locations", [], paths)
        checks.run("model", [], files)
        checks.run("parameters", [], options)
        if not checks.diagnostics:
            run_root, build, scratch = (loc[k] for k in ("run_root", "build", "scratch"))
            env = {"PATH": "/usr/bin:/bin", "HOME": scratch + "/home", "XDG_CONFIG_HOME": scratch + "/config",
                   "XDG_CACHE_HOME": scratch + "/cache", "TMPDIR": scratch + "/tmp", "CUDA_CACHE_PATH": scratch + "/cuda-cache",
                   "CUDA_VISIBLE_DEVICES": resources["gpus"][0], "OMP_NUM_THREADS": str(limits["cpu_threads"]),
                   "LC_ALL": "C", "GGML_LOG_LEVEL": "3"}
            configure = [loc["cmake"], "-S", loc["source"], "-B", build, "-G", "Unix Makefiles",
                         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MAKE_PROGRAM=" + loc["make"], "-DBUILD_SHARED_LIBS=OFF", "-DGGML_CUDA=ON",
                         "-DLLAMA_BUILD_COMMON=ON", "-DLLAMA_BUILD_TOOLS=ON", "-DGGML_BACKEND_DL=OFF",
                         "-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
                         "-DGGML_CUDA_CCCL_VERSION=", "-DCMAKE_C_COMPILER=" + loc["cc"],
                         "-DCMAKE_CXX_COMPILER=" + loc["cxx"], "-DCMAKE_CUDA_COMPILER=" + loc["nvcc"],
                         "-DCMAKE_CUDA_ARCHITECTURES=" + adapter["build_parameters"]["cuda_arch"]]
            configure += ["-D" + key + "=OFF" for key in OFF_OPTIONS]
            binary = build + "/bin/llama-completion"
            infer = [binary, "-m", str(PurePosixPath(model["path"]) / selected["weights"][0]["path"]),
                     "-p", request["input"], "-n", str(params["max_tokens"]), "--seed", str(params["seed"]),
                     "--temp", "0", "--ctx-size", str(params["ctx_size"]), "--threads", str(limits["cpu_threads"]),
                     "--threads-batch", str(limits["cpu_threads"]), "--device", "CUDA0", "--n-gpu-layers", str(params["gpu_layers"]),
                     "--split-mode", "none", "--fit", "off", "--offline", "--no-warmup", "--no-conversation",
                     "--no-display-prompt", "--no-context-shift", "--simple-io", "--load-mode", "none", "--verbosity", "3", "--log-colors", "off", "--color", "off"]
            for phase, argv in (("configure", configure), ("build", [loc["cmake"], "--build", build, "--config", "Release", "--target", "llama-completion", "--parallel", str(limits["cpu_threads"])]), ("run_once", infer)):
                commands.append({"phase": phase, "argv": argv, "cwd": run_root if phase != "run_once" else scratch,
                                 "env": copy.deepcopy(env), "inherit_environment": False, "stdin": "DEVNULL",
                                 "limits": copy.deepcopy(limits)})
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, RecursionError, OverflowError) as error:
        checks.issue("document", "invalid_adapter_input", str(error)[:300], "Supply an explicit task and this profile's location fields")
        commands = []
    result = {"version": VERSION, "status": "draft" if commands else "invalid", "task_hash": task_hash,
              "commands": commands, "diagnostics": checks.diagnostics, "executed": False, "execution_authorized": False,
              "readiness": "unverified", "required_runtime_checks": list(RUNTIME_CHECKS),
              "limits": ["Arguments were derived from declarations, not from verified file/device state.",
                         "Offline flags and an isolated environment are not a network or filesystem sandbox.",
                         "CUDA0 is the backend ordinal after single-UUID masking; the mapping must be verified at runtime."]}
    result["plan_hash"] = digest(result)
    return result


_NUMBER = r"(?:[0-9]{1,15}(?:\.[0-9]{0,9})?|\.[0-9]{1,9})"
_DEVICE = re.compile(r"llama_prepare_model_devices: using device (CUDA[0-9]{1,3}) \((.{1,300})\) \(([^()]{1,100})\) - ([0-9]{1,12}) MiB free")
_OFFLOAD = re.compile(r"load_tensors: offloaded ([0-9]{1,6})/([0-9]{1,6}) layers to GPU")
_PROMPT = re.compile(r"llama_perf_context_print:\s+prompt eval time =\s*(" + _NUMBER + r") ms /\s*([0-9]{1,9}) tokens.*")
_EVAL = re.compile(r"llama_perf_context_print:\s+eval time =\s*(" + _NUMBER + r") ms /\s*([0-9]{1,9}) runs.*")
_LOAD = re.compile(r"llama_perf_context_print:\s+load time =\s*(" + _NUMBER + r") ms")


def parse_completion_log(text, *, complete):
    """Extract version-specific evidence, not a GPU/behavior pass verdict."""
    result = {"version": VERSION, "complete": complete is True, "status": "inconclusive", "evidence": [],
              "device": None, "offloaded_layers": None, "total_layers": None, "prompt_tokens": None,
              "eval_runs": None, "load_ms": None, "generated_tokens": None, "issues": [],
              "runtime_verified": False, "behavior_verified": False}
    if not isinstance(text, str) or type(complete) is not bool or len(text) > 1024 * 1024:
        result["issues"].append("invalid_or_oversized_log")
        return result
    try:
        if len(text.encode("utf-8")) > 1024 * 1024:
            result["issues"].append("invalid_or_oversized_log")
            return result
    except UnicodeError:
        result["issues"].append("invalid_log_encoding")
        return result
    counts = {}
    for number, line in enumerate(text.splitlines(), 1):
        if len(line) > 2000:
            if "oversized_line" not in result["issues"]:
                result["issues"].append("oversized_line")
            continue
        clean = line.strip()
        kind, values = None, None
        if match := _DEVICE.fullmatch(clean):
            kind, values = "device", {"backend": match[1], "description": match[2], "bus_id": match[3]}
        elif match := _OFFLOAD.fullmatch(clean):
            kind, values = "offload", (int(match[1]), int(match[2]))
        elif match := _PROMPT.fullmatch(clean):
            kind, values = "prompt", int(match[2])
        elif match := _EVAL.fullmatch(clean):
            kind, values = "eval", int(match[2])
        elif match := _LOAD.fullmatch(clean):
            kind, values = "load", float(match[1])
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if counts[kind] > 1:
            if "ambiguous_repeated_" + kind not in result["issues"]:
                result["issues"].append("ambiguous_repeated_" + kind)
            continue
        result["evidence"].append({"line": number, "text": line[:2000], "kind": kind})
        if kind == "device":
            result["device"] = values
        elif kind == "offload":
            result["offloaded_layers"], result["total_layers"] = values
        else:
            result[{"prompt": "prompt_tokens", "eval": "eval_runs", "load": "load_ms"}[kind]] = values
    if not complete:
        result["issues"].append("partial_log")
    for kind in ("device", "offload", "prompt", "eval", "load"):
        if kind not in counts:
            result["issues"].append("missing_" + kind)
    if result["offloaded_layers"] is not None and not 0 < result["offloaded_layers"] <= result["total_layers"]:
        result["issues"].append("no_valid_gpu_offload")
    if result["prompt_tokens"] == 0 or result["eval_runs"] == 0:
        result["issues"].append("no_decoded_request_evidence")
    if result["load_ms"] is not None and not math.isfinite(result["load_ms"]):
        result["issues"].append("nonfinite_load_time")
    if not result["issues"]:
        result["status"] = "evidence_present"
    return result
