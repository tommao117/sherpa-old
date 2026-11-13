from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sherpa.agent import AgentError, BackendResponse, Config, Request, Router
from sherpa.agent import codebuddy
from sherpa.cli import main
from sherpa.runtime.calls import Calls, RunStopped
from sherpa.runtime.models import Task
from sherpa.runtime.store import Store, read_json


def validate(data):
    if set(data) != {"answer"} or not isinstance(data["answer"], str):
        raise ValueError("expected an answer string")


class FakeBackend:
    capabilities = frozenset({"json"})
    version = "fake-v1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, model_id, request, context):
        self.calls.append((model_id, request, context))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(context)
        return response


def configuration():
    return {"version": 1, "backends": {"test": {"type": "fake"}},
            "pools": {"daily": ["a", "b"], "expert": ["c", "d"]},
            "models": {name: {"backend": "test", "model_id": name} for name in "abcd"}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-agent-test-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "input"
        source.mkdir()
        task = Task("test", {"kind": "local", "path": str(source)}, {"id": "mock"},
                    budget={"seconds": 30, "max_calls": 10, "model_cost_usd": 1})
        self.run = Store(root / "data").create_run(task)
        self.calls = Calls(self.run, seconds=30, max_calls=10)
        self.request = Request("daily", "read metadata", "Use only supplied evidence", {"files": []}, {"answer": "string"})
        self.good = BackendResponse('{"answer":"ok"}', usage={"input_tokens": 10}, cost_usd=0.01)

    def router(self, responses, data=None):
        backend = FakeBackend(responses)
        router = Router(Config(data or configuration()), {"fake": lambda cfg: backend})
        return router, backend

    def call(self, router, request=None):
        return router.call(request or self.request, self.calls, validate, allow_unpriced=True)

    def test_exact_user_names_and_verified_cli_ids_are_separate(self):
        config = Config.load()
        self.assertEqual(config.data["pools"]["daily"], ["deepseek-v4.1-flash", "GPT6 Luna", "hy4"])
        self.assertEqual(config.data["pools"]["expert"], ["Claude-Opus-5.5", "GPT-6-Astra", "Claude-Opus-5", "GPT-6.1-Sol"])
        self.assertTrue(config.data["models"]["hy4"]["enabled"])
        self.assertEqual(config.data["models"]["hy4"]["model_id"], "hy4std")
        self.assertEqual(config.data["models"]["GPT6 Luna"]["model_id"], "gpt-6-luna")
        self.assertFalse(config.describe()["verified_online"])

    def test_config_is_snapshotted_not_live_mutable_input(self):
        data = configuration()
        config = Config(data)
        before = config.version
        data["models"]["a"]["model_id"] = "changed"
        self.assertEqual(config.candidates("daily")[0][1]["model_id"], "a")
        self.assertEqual(config.version, before)
        data["models"]["a"]["model_id"] = "auto"
        with self.assertRaises(ValueError):
            Config(data)

    def test_ordered_failover_records_both_attempts_and_cost(self):
        router, backend = self.router([BackendResponse("", cost_usd=0.001, error_category="rate_limited", error_message="429"), self.good])
        answer = self.call(router)
        self.assertEqual([c[0] for c in backend.calls], ["a", "b"])
        self.assertTrue(answer.switched)
        records = self.run.records("calls")
        self.assertEqual([r["status"] for r in records], ["failed", "succeeded"])
        self.assertEqual(records[1]["caused_by"], records[0]["id"])
        self.assertEqual(records[1]["request_id"], records[0]["request_id"])
        self.assertEqual(records[1]["attempt"], 2)
        self.assertAlmostEqual(self.run.state()["cost"]["model_cost_usd"], 0.011)
        self.assertIsNone(records[1]["reported_model"])
        self.assertEqual(records[1]["requested_model"], "b")
        for record in records:
            self.assertFalse(record["synthetic"])
            self.assertFalse(record["large_model_intervention"])
            self.assertEqual(read_json(self.run.path / record["input"])["wire_prompt"], self.request.prompt())
            self.assertTrue((self.run.path / "calls" / record["id"] / "raw.json").exists())

    def test_api_unavailable_never_crosses_into_expert_pool(self):
        router, backend = self.router([AgentError("unavailable", "gone"), AgentError("unavailable", "gone")])
        with self.assertRaisesRegex(AgentError, "no cross-role"):
            self.call(router)
        self.assertEqual([c[0] for c in backend.calls], ["a", "b"])
        self.assertEqual(self.calls.count, 2)

    def test_expert_request_is_bounded_and_marked(self):
        with self.assertRaises(ValueError):
            Request("expert", "fix it", "do work", {}, {})
        request = Request("expert", "explain binary format", "Read evidence only", {}, {"answer": "string"},
                          intervention_reason="daily could not identify runtime", completion_criteria="list supported runtimes with evidence",
                          caused_by="daily-decision-1")
        router, backend = self.router([AgentError("transport", "network"), self.good])
        self.call(router, request)
        self.assertEqual([c[0] for c in backend.calls], ["c", "d"])
        self.assertTrue(all(r["large_model_intervention"] for r in self.run.records("calls")))
        self.assertEqual(self.run.records("calls")[0]["caused_by"], "daily-decision-1")

    def test_invalid_answer_does_not_trigger_model_escalation_or_failover(self):
        router, backend = self.router([BackendResponse('{"wrong":"field"}'), self.good])
        with self.assertRaises(AgentError) as error:
            self.call(router)
        self.assertEqual(error.exception.category, "invalid_response")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(self.run.records("calls")[0]["status"], "failed")

    def test_unknown_cost_is_not_zero_and_survives_later_tool_calls(self):
        router, _ = self.router([BackendResponse('{"answer":"ok"}')])
        self.call(router)
        self.assertIsNone(self.run.state()["cost"]["model_cost_usd"])
        self.calls.tool("inspect", {})
        cost = self.run.state()["cost"]
        self.assertIsNone(cost["model_cost_usd"])
        self.assertEqual(cost["unpriced_model_calls"], 1)

    def test_explicit_unpriced_opt_in_required(self):
        router, backend = self.router([self.good])
        with self.assertRaises(AgentError) as error:
            router.call(self.request, self.calls, validate)
        self.assertEqual(error.exception.category, "budget_policy")
        self.assertEqual(backend.calls, [])

    def test_zero_budget_does_not_call_provider(self):
        state = self.run.state()
        state["task"]["budget"]["model_cost_usd"] = 0
        self.run.update("test_budget", task=state["task"])
        router, backend = self.router([self.good])
        with self.assertRaises(RunStopped):
            self.call(router)
        self.assertEqual(backend.calls, [])

    def test_reported_cost_ceiling_prevents_next_attempt(self):
        router, backend = self.router([BackendResponse("", cost_usd=1.0, error_category="unavailable"), self.good])
        with self.assertRaises(RunStopped):
            self.call(router)
        self.assertEqual(len(backend.calls), 1)

    def test_attempt_budget_includes_failed_candidates(self):
        self.calls.max_calls = 1
        router, backend = self.router([AgentError("transport", "network"), self.good])
        with self.assertRaises(RunStopped):
            self.call(router)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(self.run.records("calls")), 1)

    def test_cancel_and_resource_error_never_fail_over(self):
        for category in ("cancelled", "cleanup_failed", "timed_out", "budget_exhausted"):
            with self.subTest(category=category):
                router, backend = self.router([RunStopped(category, "stop now"), self.good])
                with self.assertRaises(RunStopped):
                    self.call(router)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(self.run.records("calls")[-1]["status"], category)

    def test_cancel_after_response_keeps_raw_but_does_not_accept(self):
        def complete(context):
            self.run.cancel()
            return self.good
        router, backend = self.router([complete, self.good])
        with self.assertRaises(RunStopped):
            self.call(router)
        self.assertEqual(len(backend.calls), 1)
        record = self.run.records("calls")[0]
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue((self.run.path / "calls" / record["id"] / "raw.json").exists())

    def test_cancel_during_failed_attempt_cleanup_overrides_pool_exhaustion(self):
        data = configuration()
        data["pools"]["daily"] = ["a"]
        def failed_cleanup(context):
            self.run.cancel("cancelled during provider cleanup")
            raise AgentError("timeout", "candidate timed out")
        router, backend = self.router([failed_cleanup], data)
        with self.assertRaises(RunStopped) as stopped:
            self.call(router)
        self.assertEqual(stopped.exception.status, "cancelled")
        self.assertEqual(self.run.records("calls")[0]["status"], "cancelled")
        self.assertFalse(any(e["event"] == "agent_failover" for e in self.run.events()))

    def test_rejected_response_still_counts_reported_cost(self):
        router, backend = self.router([BackendResponse("", cost_usd=0.05, error_category="isolation", error_message="extra context")])
        with self.assertRaises(AgentError):
            self.call(router)
        self.assertEqual(self.run.state()["cost"]["known_model_cost_usd"], 0.05)
        self.assertEqual(self.run.records("calls")[0]["error_category"], "isolation")

    def test_model_mismatch_rejected_without_fallback(self):
        router, backend = self.router([BackendResponse('{"answer":"ok"}', reported_model="unexpected"), self.good])
        with self.assertRaises(AgentError) as error:
            self.call(router)
        self.assertEqual(error.exception.category, "model_mismatch")
        self.assertEqual(len(backend.calls), 1)

    def test_unresolved_candidate_skipped_with_reason(self):
        data = configuration()
        data["models"]["a"] = {"backend": "test", "enabled": False, "note": "await mapping"}
        router, backend = self.router([self.good], data)
        self.assertEqual(self.call(router).candidate, "b")
        skipped = next(e for e in self.run.events() if e["event"] == "agent_candidate_skipped")
        self.assertEqual(skipped["reason"], "await mapping")
        self.assertEqual(len(backend.calls), 1)

    def test_other_backend_can_be_registered_without_cli_calls(self):
        router, backend = self.router([self.good])
        with patch("subprocess.Popen", side_effect=AssertionError("no CLI for this backend")):
            result = self.call(router)
        self.assertEqual(result.output, {"answer": "ok"})

    def test_capability_rejected_before_transport(self):
        router, backend = self.router([self.good])
        request = Request("daily", "choose", "choose", {}, {}, capability="native-choice")
        with self.assertRaises(AgentError) as error:
            self.call(router, request)
        self.assertEqual(error.exception.category, "capability")
        self.assertFalse(backend.calls)

    def test_listing_does_not_invoke_cli_or_create_run(self):
        before = len(self.run.store.runs())
        buffer = io.StringIO()
        with redirect_stdout(buffer), patch("subprocess.Popen", side_effect=AssertionError("listing is local")):
            code = main(["--data-root", str(self.run.store.root), "agent", "list"])
        self.assertEqual(code, 0)
        self.assertEqual(len(self.run.store.runs()), before)
        self.assertFalse(json.loads(buffer.getvalue())["verified_online"])


class CodeBuddyTests(unittest.TestCase):
    def envelope(self, prompt="prompt", **fields):
        return json.dumps([{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<user_query>" + prompt + "</user_query>"}]},
            {"type": "result", "result": '{"answer":"ok"}', "usage": {"input_tokens": 3}, **fields}])

    def test_exact_echo_no_hidden_context_and_unknown_actual_model(self):
        response = codebuddy.parse_response(self.envelope(), "prompt", "requested")
        self.assertIsNone(response.reported_model)
        self.assertIsNone(response.cost_usd)
        self.assertEqual(response.usage["input_tokens"], 3)
        self.assertFalse(response.metadata["full_provider_context_verified"])

    def test_hidden_context_rejected(self):
        response = codebuddy.parse_response(self.envelope("extra memory\nprompt", total_cost_usd=0.02), "prompt", "m")
        self.assertEqual(response.error_category, "isolation")
        self.assertEqual(response.cost_usd, 0.02)
        self.assertEqual(response.usage["input_tokens"], 3)
        self.assertEqual(response.text, "")

    def test_missing_echo_fail_closed(self):
        response = codebuddy.parse_response('{"type":"result","result":"{}"}', "prompt", "m")
        self.assertEqual(response.error_category, "isolation")

    def test_api_fault_classification(self):
        for message, expected in (("429 Too many requests", "rate_limited"), ("Model is unavailable", "unavailable"),
                                  ("ECONNRESET", "transport"), ("403 Unauthorized", "authentication"),
                                  ("policy refused", "provider_error")):
            with self.subTest(message=message):
                response = codebuddy.parse_response(self.envelope(is_error=True, result=message), "prompt", "m")
                self.assertEqual(response.error_category, expected)

    def test_memory_guard_never_launches_under_pressure(self):
        with patch.object(codebuddy, "host_memory", return_value={"MemAvailable": 1024, "SwapTotal": 0}):
            with self.assertRaises(RunStopped):
                codebuddy.check_resources()

    def test_supervisor_kills_owned_children_when_parent_exits(self):
        with tempfile.TemporaryDirectory(prefix="sherpa-cli-supervisor-") as tmp:
            directory = Path(tmp)
            from sherpa.agent import supervisor
            helper = ("import os,subprocess,sys,time; from pathlib import Path; "
                      f"p=subprocess.Popen([sys.executable,'-I',{supervisor.__file__!r},str(os.getpid()),"
                      "sys.executable,'-I','-c','import time; time.sleep(60)'],start_new_session=True); "
                      f"Path({str(directory / 'pid')!r}).write_text(str(p.pid)); time.sleep(60)")
            parent = subprocess.Popen([sys.executable, "-I", "-c", helper], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child = None
            try:
                deadline = time.monotonic() + 4
                while not (directory / "pid").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                child = int((directory / "pid").read_text())
                time.sleep(0.1)
                parent.kill()
                parent.wait(timeout=3)
                deadline = time.monotonic() + 3
                while not Calls.group_gone(child) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(Calls.group_gone(child))
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=3)
                if child and not Calls.group_gone(child):
                    import signal
                    os.killpg(child, signal.SIGKILL)


class CLIExitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sherpa-cli-exit-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.request = Request("daily", "test transport", "Answer JSON only", {}, {"answer": "string"})

    def invoke(self, body, timeout=5, **config):
        from sherpa.agent import AttemptContext
        script = self.directory / "fake-cli"
        script.write_text(f"#!{sys.executable}\n" + body)
        script.chmod(0o700)
        attempt = self.directory / "attempt"
        attempt.mkdir()
        context = AttemptContext(attempt, time.monotonic() + timeout, lambda: None, lambda identity: None)
        with patch.object(codebuddy, "check_resources"):
            return codebuddy.CodeBuddy({"command": str(script), **config}).complete("model", self.request, context)

    def test_internal_file_larger_than_old_limit_is_allowed(self):
        body = ("import json,sys\n"
                "with open('internal-cache.bin','wb') as handle:\n"
                " for _ in range(80): handle.write(b'x'*65536)\n"
                "print(json.dumps([{'type':'message','role':'user','content':sys.argv[-1]},"
                "{'type':'result','result':'{}'}]))\n")
        response = self.invoke(body, max_output_mib=1)
        self.assertIsNone(response.error_category)
        self.assertEqual((self.directory / "attempt/session/internal-cache.bin").stat().st_size, 5 * 1024 * 1024)
        self.assertEqual(response.metadata["max_output_bytes"], 1024 * 1024)
        self.assertEqual(response.metadata["child_returncode"], 0)

    def test_default_accepts_captured_output_above_four_mib(self):
        body = ("import json,sys\n"
                "for _ in range(80): sys.stderr.write('x'*65536)\n"
                "print(json.dumps([{'type':'message','role':'user','content':sys.argv[-1]},"
                "{'type':'result','result':'{}'}]))\n")
        response = self.invoke(body)
        self.assertIsNone(response.error_category)
        self.assertEqual(response.metadata["stderr_bytes"], 5 * 1024 * 1024)
        self.assertEqual(response.metadata["max_output_bytes"], 16 * 1024 * 1024)

    def test_combined_output_checked_even_when_child_exits_fast(self):
        body = "import sys; sys.stdout.write('x'*600000); sys.stderr.write('x'*600000)\n"
        with self.assertRaises(AgentError) as error:
            self.invoke(body, max_output_mib=1)
        self.assertEqual(error.exception.category, "output_limit")
        self.assertIn("1 MiB combined", str(error.exception))
        self.assertTrue(error.exception.diagnostics["cleanup_complete"])

    def test_excess_output_stops_live_child_and_cleans_group(self):
        body = "import sys,time; sys.stdout.write('x'*2097152); sys.stdout.flush(); time.sleep(10)\n"
        with self.assertRaises(AgentError) as error:
            self.invoke(body, max_output_mib=1)
        self.assertEqual(error.exception.category, "output_limit")
        metadata = read_json(self.directory / "attempt/process.json")
        self.assertEqual(metadata["termination_reason"], "output_limit")
        self.assertTrue(metadata["cleanup_complete"])

    def test_output_limit_configuration_is_validated(self):
        for value in (0, -1, True, "16", 1.5):
            with self.subTest(value=value), self.assertRaises(AgentError):
                codebuddy.CodeBuddy({"max_output_mib": value})
        self.assertEqual(codebuddy.CodeBuddy({"max_output_mib": 64}).output_limit, 64 * 1024 * 1024)

    def test_empty_nonzero_cli_exit_has_diagnostic_not_api_guess(self):
        with self.assertRaises(AgentError) as error:
            self.invoke("import sys; sys.exit(7)\n")
        self.assertEqual(error.exception.category, "cli_exit")
        self.assertIn("child=7", str(error.exception))
        metadata = read_json(self.directory / "attempt/process.json")
        self.assertEqual(metadata["child_returncode"], 7)
        self.assertEqual(metadata["stdout_bytes"], 0)
        self.assertEqual(metadata["stderr_bytes"], 0)
        self.assertTrue(metadata["cleanup_complete"])
        self.assertEqual(error.exception.diagnostics["child_returncode"], 7)

    def test_signal_exit_is_not_mislabeled_as_memory_failure(self):
        with self.assertRaises(AgentError) as error:
            self.invoke("import os,signal; os.kill(os.getpid(), signal.SIGTERM)\n")
        self.assertEqual(error.exception.category, "process_signal")
        metadata = read_json(self.directory / "attempt/process.json")
        self.assertEqual(metadata["child_returncode"], -15)
        self.assertEqual(metadata["child_signal"], "SIGTERM")
        self.assertEqual(metadata["supervisor_returncode"], 143)

    def test_structured_api_error_on_stdout_keeps_cost_when_exit_nonzero(self):
        body = "import json,sys; print(json.dumps({'type':'result','is_error':True,'result':'429 Too many requests','total_cost_usd':0.02,'usage':{'input_tokens':9}})); sys.exit(1)\n"
        response = self.invoke(body)
        self.assertEqual(response.error_category, "rate_limited")
        self.assertEqual(response.cost_usd, 0.02)
        self.assertEqual(response.usage["input_tokens"], 9)
        self.assertEqual(response.metadata["child_returncode"], 1)

    def test_success_output_does_not_override_failed_exit(self):
        body = "import json,sys; print(json.dumps([{'type':'message','role':'user','content':sys.argv[-1]},{'type':'result','result':'{}','total_cost_usd':0.03}])); sys.exit(2)\n"
        response = self.invoke(body)
        self.assertEqual(response.error_category, "cli_exit")
        self.assertEqual(response.cost_usd, 0.03)
        self.assertEqual(response.text, "")

    def test_timeout_keeps_termination_reason_and_cleanup_status(self):
        with self.assertRaises(AgentError) as error:
            self.invoke("import time; time.sleep(10)\n", timeout=0.2)
        self.assertEqual(error.exception.category, "timeout")
        metadata = read_json(self.directory / "attempt/process.json")
        self.assertEqual(metadata["termination_reason"], "timeout")
        self.assertTrue(metadata["cleanup_complete"])


if __name__ == "__main__":
    unittest.main()
