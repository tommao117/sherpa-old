"""Serial, role-local failover with one trace Call per candidate attempt."""
from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable

from ..runtime.calls import RunStopped
from ..runtime.store import atomic_json
from .base import AgentError, Answer, AttemptContext, FAILOVER_ERRORS, Request
from .codebuddy import CodeBuddy
from .config import Config


class Router:
    def __init__(self, config: Config | None = None, backend_factories=None):
        self.config = config or Config.load()
        self.factories = {"codebuddy_cli": CodeBuddy, **(backend_factories or {})}
        self.instances = {}

    def backend(self, name):
        if name not in self.instances:
            entry = self.config.data["backends"][name]
            factory = self.factories.get(entry["type"])
            if factory is None:
                raise AgentError("configuration", f"no adapter registered for {entry['type']}")
            self.instances[name] = factory(entry)
        return self.instances[name]

    def call(self, request: Request, calls, validate: Callable[[dict], None], *,
             timeout_s=90, allow_unpriced=False, request_timeout_s=None, max_attempts=None) -> Answer:
        if max_attempts is not None and (isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1):
            raise ValueError("max_attempts must be a positive integer")
        if request_timeout_s is not None and (isinstance(request_timeout_s, bool)
                or not isinstance(request_timeout_s, (int, float)) or not math.isfinite(request_timeout_s)
                or not 0 < request_timeout_s <= 600):
            raise ValueError("request timeout must be finite and in (0, 600]")
        request_deadline = min(calls.deadline, time.monotonic() + request_timeout_s) if request_timeout_s else calls.deadline
        def check_request():
            calls.check_interrupt()
            if time.monotonic() >= request_deadline:
                raise AgentError("request_timeout", "bounded request deadline exceeded; return control to caller")
        if not callable(validate):
            raise ValueError("caller must provide semantic output validation")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("candidate timeout must be in (0, 300] seconds")
        # CLI accounting is not a hard pre-paid spending limit. Require explicit opt-in,
        # then bound wall time and number of attempts; never interpret absent costs as zero.
        if not allow_unpriced:
            raise AgentError("budget_policy", "provider charges cannot be pre-bounded; explicit allow_unpriced is required")
        task = calls.run.state().get("task") or {}
        cap = task.get("budget", {}).get("model_cost_usd", 0)
        if cap <= 0:
            raise RunStopped("budget_exhausted", "task has zero model-call budget")
        group_id = "request-" + uuid.uuid4().hex[:12]
        calls.run.event("agent_request_started", request_id=group_id, role=request.role,
                        config_version=self.config.version, config=self.config.data,
                        pricing_policy="unknown charges permitted; only time/attempts are hard-bounded")
        previous, attempted = request.caused_by, []
        for name, entry in self.config.candidates(request.role):
            if not entry.get("enabled", True):
                calls.run.event("agent_candidate_skipped", request_id=group_id, candidate=name,
                                role=request.role, reason=entry.get("note", "disabled"))
                continue
            calls.check()
            check_request()
            if max_attempts is not None and len(attempted) >= max_attempts:
                raise AgentError("attempt_limit", "bounded request attempt limit reached")
            cost = calls.cost_summary()
            if cost["known_model_cost_usd"] >= cap:
                raise RunStopped("budget_exhausted", "reported model costs reached task budget")
            backend = self.backend(entry["backend"])
            if request.capability not in backend.capabilities:
                raise AgentError("capability", f"{entry['backend']} does not support {request.capability}")
            inputs = {"role": request.role, "purpose": request.purpose, "instructions": request.instructions,
                      "context": request.context, "output_contract": request.output_contract,
                      "wire_prompt": request.prompt(), "config_version": self.config.version,
                      "candidate": name, "requested_model": entry["model_id"], "backend": entry["backend"],
                      "capability": request.capability, "completion_criteria": request.completion_criteria}
            started = time.monotonic()
            record, directory = calls.new("agent", entry["backend"], inputs, caused_by=previous,
                                          large_model=request.role == "expert", reason=request.intervention_reason)
            record.update(request_id=group_id, attempt=len(attempted) + 1, role=request.role, candidate=name,
                          backend=entry["backend"], backend_version=backend.version,
                          requested_model=entry["model_id"], reported_model=None,
                          config_version=self.config.version, synthetic=False, usage={}, cost_usd=None)
            calls.run.save("calls", record["id"], record, "agent_attempt_started")
            def process_started(identity):
                record["process"] = identity
                calls.run.save("calls", record["id"], record, "agent_process_started")
            attempt_context = AttemptContext(directory, min(request_deadline, started + timeout_s),
                                             check_request, process_started)
            attempted.append(name)
            try:
                response = backend.complete(entry["model_id"], request, attempt_context)
                record.update(reported_model=response.reported_model, usage=response.usage,
                              cost_usd=response.cost_usd, provider_metadata=response.metadata)
                atomic_json(directory / "raw.json", {"text": response.text, "reported_model": response.reported_model,
                                                     "error_category": response.error_category,
                                                     "error_message": response.error_message})
                check_request()
                if response.reported_model and response.reported_model != entry["model_id"]:
                    raise AgentError("model_mismatch", "actual model differs from requested model")
                if response.error_category:
                    raise AgentError(response.error_category, response.error_message or "provider failed")
                try:
                    parsed = json.loads(response.text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    if not isinstance(parsed, dict):
                        raise ValueError("reply must be an object")
                    validate(parsed)
                except (ValueError, TypeError, KeyError) as error:
                    raise AgentError("invalid_response", str(error)) from error
                check_request()
                calls.finish(record, directory, "succeeded", parsed, started)
                calls.run.event("agent_request_finished", request_id=group_id, role=request.role,
                                candidate=name, call_id=record["id"], switched=len(attempted) > 1)
                return Answer(parsed, record["id"], name, entry["model_id"], response.reported_model,
                              switched=len(attempted) > 1)
            except AgentError as error:
                record["error_category"] = error.category
                if error.diagnostics:
                    record["provider_metadata"] = {**record.get("provider_metadata", {}), **error.diagnostics}
                try:
                    calls.check_interrupt()
                except RunStopped as stop:
                    calls.finish(record, directory, stop.status, {"error": stop.reason,
                                                                 "candidate_error": error.category}, started)
                    raise
                calls.finish(record, directory, "failed", {"error": str(error), "category": error.category}, started)
                if error.category not in FAILOVER_ERRORS:
                    raise
                calls.run.event("agent_failover", request_id=group_id, role=request.role, candidate=name,
                                failed_call=record["id"], reason=error.category)
                previous = record["id"]
            except (RunStopped, KeyboardInterrupt) as error:
                state = error.status if isinstance(error, RunStopped) else "cancelled"
                calls.finish(record, directory, state, {"error": str(error)}, started)
                raise
            except Exception as error:
                calls.finish(record, directory, "failed", {"category": "internal", "error": str(error)}, started)
                raise
        raise AgentError("pool_exhausted", f"{request.role} candidates exhausted: {attempted}; no cross-role fallback")
