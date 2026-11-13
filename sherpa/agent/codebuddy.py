"""A replaceable CodeBuddy transport. No tools, MCP services, resume or automatic memory."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..runtime.calls import Calls, RunStopped
from ..runtime.store import atomic_json, process_identity
from .base import AgentError, BackendResponse

DEFAULT_OUTPUT_LIMIT_MIB = 16
RSS_LIMIT_KIB = 768 * 1024
HOST_RESERVE_KIB = 4 * 1024 * 1024
ISOLATION_ENV = {"CODEBUDDY_DISABLE_AUTO_MEMORY": "1", "CODEBUDDY_DISABLE_SYSTEM_REMINDER_MD": "1"}


def error_category(text):
    if re.search(r"\b429\b|rate.?limit|too many requests", text, re.I):
        return "rate_limited"
    if re.search(r"\b(?:401|403)\b|unauthorized|authentication failed|not logged in", text, re.I):
        return "authentication"
    if re.search(r"model.*(?:not found|unavailable|not supported|not available)|\b(?:502|503|504)\b", text, re.I):
        return "unavailable"
    if re.search(r"ECONNRESET|ECONNREFUSED|ENOTFOUND|ETIMEDOUT|network error", text, re.I):
        return "transport"
    return "provider_error"


def parse_response(raw: str, prompt: str, requested: str):
    try:
        messages = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise AgentError("protocol", "CLI did not return a JSON response") from error
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        raise AgentError("protocol", "unexpected CLI response envelope")
    result = next((item for item in reversed(messages) if isinstance(item, dict) and item.get("type") == "result"), None)
    if result is None:
        raise AgentError("protocol", "CLI returned no result record")
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    cost = result.get("total_cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
        cost = None
    reported = result.get("model")
    if not reported:
        reported = next((item.get("model") for item in reversed(messages)
                         if isinstance(item, dict) and item.get("role") == "assistant" and item.get("model")), None)
    def rejected(category, message):
        return BackendResponse("", reported, usage, cost, category, message)
    if reported and reported != requested:
        return rejected("model_mismatch", "provider reported a different model than requested")
    if result.get("is_error"):
        text = str(result.get("result", "provider error"))
        return rejected(error_category(text), text[:1000])
    user = next((item for item in messages if isinstance(item, dict)
                 and item.get("type") == "message" and item.get("role") == "user"), None)
    if user is None:
        return rejected("isolation", "CLI does not expose the user turn; input isolation cannot be checked")
    content = user.get("content")
    if isinstance(content, list):
        if any(not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}
               or not isinstance(part.get("text", ""), str) for part in content):
            return rejected("isolation", "unexpected injected user content")
        content = "".join(part.get("text", "") for part in content)
    if not isinstance(content, str):
        return rejected("isolation", "unexpected user-turn format")
    if content.strip() not in {prompt.strip(), "<user_query>" + prompt.strip() + "</user_query>"}:
        return rejected("isolation", "CLI user turn contains unexpected context; answer rejected")
    text = result.get("result")
    if not isinstance(text, str):
        return rejected("protocol", "CLI result is not text")
    return BackendResponse(text, reported, usage, cost, metadata={
        "user_turn_checked": True, "full_provider_context_verified": False,
        "cost_source": "CLI reported; not independently priced"})


def decode_exit(raw, stderr, returncode, diagnostics, prompt, model_id):
    """Separate CLI process failures from structured provider failures, preserving known charges."""
    if not returncode:
        return parse_response(raw, prompt, model_id)
    child_signal = diagnostics.get("child_signal")
    summary = (f"CLI exited: supervisor={returncode}, child={diagnostics.get('child_returncode')}, "
               f"signal={child_signal or 'none/unknown'}, stdout={diagnostics.get('stdout_bytes', len(raw.encode()))} bytes, "
               f"stderr={diagnostics.get('stderr_bytes', len(stderr.encode()))} bytes; "
               "see this call's process.json and provider.stdout/provider.stderr")
    if child_signal or returncode < 0 or "heap out of memory" in stderr.lower():
        category = ("output_limit" if child_signal == "SIGXFSZ" else "resource_limit"
                    if child_signal == "SIGXCPU" or "heap out of memory" in stderr.lower() else "process_signal")
        raise AgentError(category, summary)
    # Nonzero CLI exit can still carry a structured API error (and usage) in stdout.
    if raw.strip():
        try:
            response = parse_response(raw, prompt, model_id)
        except AgentError:
            response = None
        if response is not None:
            if response.error_category:
                return response
            return BackendResponse("", response.reported_model, response.usage, response.cost_usd,
                                   "cli_exit", summary, response.metadata)
    category = error_category(stderr)
    if category == "provider_error":
        category = "cli_exit"
    raise AgentError(category, summary)


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(value.split()[0])
    return values


def group_rss(group):
    rss = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE") // 1024
        except (OSError, ValueError, IndexError):
            continue
    return rss


def check_resources(group=None):
    memory = host_memory()
    swapped = memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)
    if memory.get("MemAvailable", 0) < HOST_RESERVE_KIB:
        raise RunStopped("budget_exhausted", "host available memory below 4 GiB; refusing agent process")
    if memory.get("SwapTotal", 0) and swapped > memory["SwapTotal"] / 2:
        raise RunStopped("budget_exhausted", "host swap usage exceeds half; refusing agent process")
    if group is not None and group_rss(group) > RSS_LIMIT_KIB:
        raise RunStopped("budget_exhausted", "agent process group exceeded 768 MiB RSS")


class CodeBuddy:
    capabilities = frozenset({"json"})
    version = "codebuddy-transport-v2"

    def __init__(self, config):
        self.command = config.get("command", "codebuddy")
        limit = config.get("max_output_mib", DEFAULT_OUTPUT_LIMIT_MIB)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise AgentError("configuration", "max_output_mib must be a positive integer")
        self.output_limit = limit * 1024 * 1024

    def check_output_size(self, stdout_bytes, stderr_bytes):
        if stdout_bytes + stderr_bytes > self.output_limit:
            raise AgentError("output_limit", f"CLI stdout/stderr exceeded {self.output_limit // (1024 * 1024)} MiB combined")

    def complete(self, model_id, request, context):
        executable = shutil.which(self.command)
        if not executable:
            raise AgentError("unavailable", "CodeBuddy executable not found")
        check_resources()
        prompt = request.prompt()
        if len(prompt.encode()) + len(request.instructions.encode()) > 48 * 1024:
            raise AgentError("input_limit", "request exceeds 48 KiB; narrow its evidence")
        cwd = context.directory / "session"
        cwd.mkdir()
        argv = [executable, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence", "--max-turns", "1",
                "--model", model_id, "--system-prompt", request.instructions, prompt]
        status_path = context.directory / "child-status.json"
        env = {**os.environ, **ISOLATION_ENV, "TMPDIR": str(cwd), "PYTHONDONTWRITEBYTECODE": "1",
               "NODE_OPTIONS": "--max-old-space-size=384 --max-semi-space-size=16",
               "SHERPA_AGENT_PROCESS_STATUS": str(status_path)}
        command = [sys.executable, "-I", str(Path(__file__).with_name("supervisor.py")), str(os.getpid()), *argv]
        diagnostics = {"requested_model": model_id, "supervisor_returncode": None, "child_returncode": None,
                       "child_signal": None, "sampled_peak_rss_kib": 0, "termination_reason": None,
                       "cleanup_complete": False, "artifact": "process.json",
                       "max_output_bytes": self.output_limit}
        try:
            with (context.directory / "provider.stdout").open("wb") as stdout, (context.directory / "provider.stderr").open("wb") as stderr:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    context.process_started(process_identity(process.pid))
                    while process.poll() is None:
                        context.check()
                        diagnostics["sampled_peak_rss_kib"] = max(diagnostics["sampled_peak_rss_kib"], group_rss(process.pid))
                        check_resources(process.pid)
                        if time.monotonic() >= context.deadline:
                            raise AgentError("timeout", "candidate request timed out")
                        self.check_output_size(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                        time.sleep(0.1)
                except BaseException as error:
                    diagnostics["termination_reason"] = getattr(error, "category", getattr(error, "status", type(error).__name__))
                    raise
                finally:
                    diagnostics["cleanup_complete"] = Calls.cleanup(process)
                    diagnostics["supervisor_returncode"] = process.returncode
                    diagnostics["stdout_bytes"] = os.fstat(stdout.fileno()).st_size
                    diagnostics["stderr_bytes"] = os.fstat(stderr.fileno()).st_size
                    if status_path.exists():
                        try:
                            diagnostics.update(json.loads(status_path.read_text()))
                        except (OSError, ValueError):
                            diagnostics["child_status_error"] = "unreadable child status"
                    atomic_json(context.directory / "process.json", diagnostics)
                    if not diagnostics["cleanup_complete"]:
                        raise RunStopped("cleanup_failed", "agent process group did not stop; no fallback allowed")
            context.check()
            out = context.directory / "provider.stdout"
            err = context.directory / "provider.stderr"
            self.check_output_size(out.stat().st_size, err.stat().st_size)
            with out.open("rb") as handle:
                data = handle.read(self.output_limit + 1)
            self.check_output_size(len(data), err.stat().st_size)
            raw = data.decode(errors="replace")
            with err.open() as handle:
                message = handle.read(8192)
            response = decode_exit(raw, message, process.returncode, diagnostics, prompt, model_id)
            response.metadata.update(diagnostics)
            return response
        except AgentError as error:
            error.diagnostics.update(diagnostics)
            raise
