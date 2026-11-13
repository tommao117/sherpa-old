from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

from sherpa.runtime import report_drafts
from sherpa.runtime.explore import ACTION_FORMS
from sherpa.runtime.store import atomic_json, digest, read_json
import test_inputs as input_fixtures
from test_inputs import dual_finish
from test_explore import tool


def missing_files(request, context):
    candidate = dual_finish(request, context)
    candidate["routes"][0].pop("model_files")
    return candidate


def repair(request, changes):
    return {"action": "repair_finish", "draft_id": request.context["report_draft"]["id"], "changes": changes}


def file_repair(request, context):
    return repair(request, [{"route": 0, "field": "model_files", "value": ["weights.gguf"]}])


class DraftUnitTests(unittest.TestCase):
    def setUp(self):
        self.form = copy.deepcopy(ACTION_FORMS["finish"])
        self.form["routes"][0].pop("runtime")
        self.form["routes"][0].pop("framework_source")
        self.form["routes"][0]["model_files"] = []
        self.candidate = {"action": "finish", "summary": "unchanged", "routes": [{"name": "candidate", "steps": ["check"]}],
                          "selected_route": None, "unknowns": ["not verified"]}
        self.draft = report_drafts.make(self.candidate, "run", "call", "rejected_draft", "missing model_files")

    def test_patch_is_copy_only_and_does_not_guess_missing_fields(self):
        before = copy.deepcopy(self.draft)
        action = {"action": "repair_finish", "draft_id": self.draft["id"], "changes": [
            {"route": None, "field": "summary", "value": "new summary"}]}
        new = report_drafts.expand(action, self.draft, self.form)
        self.assertEqual(before, self.draft)
        self.assertEqual(new["summary"], "new summary")
        self.assertNotIn("model_files", new["routes"][0])
        self.assertEqual(new["routes"], self.candidate["routes"])

    def test_rejects_old_id_duplicate_targets_identity_indexes_and_generic_patches(self):
        actions = [
            {"route": True, "field": "model_files", "value": []},
            {"route": -1, "field": "model_files", "value": []},
            {"route": 1, "field": "model_files", "value": []},
            {"route": 0, "field": "framework_commit", "value": "other"},
            {"route": None, "field": "routes", "value": []},
            {"route": None, "field": "action", "value": "tool"},
            {"path": "/routes/0/model_files", "value": []},
        ]
        for change in actions:
            with self.subTest(change=change), self.assertRaises(ValueError):
                report_drafts.expand({"action": "repair_finish", "draft_id": self.draft["id"], "changes": [change]}, self.draft, self.form)
        action = {"action": "repair_finish", "draft_id": self.draft["id"], "changes": [{"route": 0, "field": "model_files", "value": []}] * 2}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report_drafts.expand(action, self.draft, self.form)
        action.update(draft_id="wrong", changes=[])
        with self.assertRaisesRegex(ValueError, "current"):
            report_drafts.expand(action, self.draft, self.form)

    def test_full_candidate_survives_display_truncation(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["routes"][0]["steps"] = ["large text " * 100] * 12
        draft = report_drafts.make(candidate, "run", "call", "rejected_draft", "missing")
        view = report_drafts.view(draft, self.form)
        self.assertTrue(view["candidate"]["truncated"])
        self.assertLessEqual(len(json.dumps(view, ensure_ascii=False).encode()), 4500)
        new = report_drafts.expand({"action": "repair_finish", "draft_id": draft["id"], "changes": [
            {"route": 0, "field": "model_files", "value": []}]}, draft, self.form)
        self.assertEqual(new["routes"][0]["steps"], candidate["routes"][0]["steps"])
        with self.assertRaises(ValueError):
            report_drafts.make({**candidate, "summary": "x" * 48000}, "run", "call", "rejected_draft", "large")

    def test_small_view_budget_counts_wrapping_escaping_and_unicode(self):
        from sherpa.platform.exploration import bounded_view
        for limit in (64, 150, 200, 300, 1800, 11000):
            for text in ('"\\\n' * 9000, "文" * 9000, "a" * (limit - 1)):
                with self.subTest(limit=limit):
                    value = bounded_view(text, limit)
                    self.assertLessEqual(len(json.dumps(value, ensure_ascii=False).encode()), limit)
                    self.assertTrue(value["truncated"])
        self.assertEqual(bounded_view("short", 64), {"data": "short", "truncated": False})

    def test_four_unicode_routes_still_fit_summary_budget(self):
        self.candidate["routes"] = [{"name": "文" * 120, "evidence": ["据" * 500], "model_files": ["文" * 800],
                                     "字段" * 300: "bad"} for _ in range(4)]
        draft = report_drafts.make(self.candidate, "run", "call", "rejected_draft", "错" * 1000)
        self.assertLessEqual(len(json.dumps(report_drafts.view(draft, self.form), ensure_ascii=False).encode()), 4500)

    def test_draft_source_windows_ignore_expired_or_sparse_references(self):
        refs = [{"id": "read", "start": 41, "end": 42}, {"id": "missing", "start": 1, "end": 1},
                {"id": "read", "start": 42, "end": 43}]
        self.candidate["routes"][0]["evidence"] = refs
        draft = report_drafts.make(self.candidate, "run", "call", "rejected_draft", "missing files")
        evidence = {"read": {"id": "read", "tool": "search", "parameters": {"scope": "model", "path": "guide.md"},
            "view": {"data": {"lines": [{"line": 41, "text": "actual model source"}, {"line": 42, "text": "next actual line"}]}, "truncated": False}}}
        views, missing = report_drafts.evidence_views(draft, evidence)
        self.assertEqual(len(views), 1)
        self.assertEqual(len(missing), 2)
        self.assertEqual(views[0]["displayed_ranges"], [[41, 42]])
        self.assertEqual(views[0]["requested_by_draft"], draft["id"])
        self.assertEqual(draft["candidate"]["routes"][0]["evidence"], refs)

    def test_old_schema_is_not_claimed_repairable(self):
        self.candidate["routes"][0]["framework_commit"] = "old"
        draft = report_drafts.make(self.candidate, "old", "call", "rejected_draft", "old schema")
        self.assertFalse(report_drafts.view(draft, self.form)["repairable"])
        with self.assertRaises(ValueError):
            report_drafts.finite_json('{"action":"finish","unknown":NaN}')


class DraftIntegrationTests(unittest.TestCase):
    setUp = input_fixtures.InputTests.setUp
    router = input_fixtures.InputTests.router
    run_inputs = input_fixtures.InputTests.run_inputs
    actions = input_fixtures.InputTests.actions

    def test_missing_field_repaired_without_rewriting_original_call(self):
        snapshot = {}
        def check(request, context):
            draft = request.context["report_draft"]
            self.assertEqual(draft["routes"][0]["missing_fields"], ["model_files"])
            run = self.store.open_run(draft["id"].split("/")[0])
            directory = run.path / "calls" / draft["id"].split("/")[1]
            snapshot.update({p: p.read_bytes() for p in directory.iterdir() if p.is_file()})
            return file_repair(request, context)
        run, proposal, backend = self.run_inputs(self.actions()[:-1] + [missing_files, check], max_rounds=5)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        self.assertEqual(snapshot, {p: p.read_bytes() for p in snapshot})
        record = run.records("calls")[-1]
        output = read_json(run.path / record["output"])
        self.assertEqual(output["action"], "repair_finish")
        self.assertNotIn("routes", output)
        candidate = read_json(run.path / record["report_draft"]["path"])
        self.assertEqual(candidate["candidate"]["routes"][0]["model_files"], ["weights.gguf"])
        self.assertEqual(proposal["plan"]["action"], "finish")
        self.assertEqual(proposal["assembly"]["candidate_hash"], candidate["candidate_hash"])
        self.assertTrue(all(model == "small" for model, _, _ in backend.requests))

    def test_full_revalidation_rejects_unchanged_bad_citation(self):
        def bad(request, context):
            value = missing_files(request, context)
            value["routes"][0]["evidence"][0]["end"] = 8
            return value
        ids = []
        def correct(request, context):
            self.assertIn("not currently displayed", request.context["report_draft"]["error"])
            ids.append(request.context["report_draft"]["id"])
            refs = dual_finish(request, context)["routes"][0]["evidence"]
            return repair(request, [{"route": 0, "field": "evidence", "value": refs}])
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [bad, file_repair, correct], max_rounds=6)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failed = [c for c in run.records("calls") if c.get("error_category") == "invalid_response"]
        self.assertEqual(len(failed), 2)
        self.assertEqual(ids[0].split("/")[1], failed[-1]["id"])
        latest = read_json(run.path / run.records("calls")[-1]["report_draft"]["path"])
        self.assertEqual(latest["base"]["id"], ids[0])

    def test_invalid_patch_and_non_json_leave_previous_draft_intact(self):
        ids = []
        def invalid(request, context):
            ids.append(request.context["report_draft"]["id"])
            return repair(request, [{"route": 0, "field": "runtime", "value": "other"}])
        def recover(request, context):
            self.assertEqual(request.context["report_draft"]["id"], ids[0])
            return file_repair(request, context)
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing_files, invalid, "not json", recover], max_rounds=7)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failed = [c for c in run.records("calls") if c.get("error_category") == "invalid_response"]
        self.assertEqual(len(failed), 3)
        self.assertFalse(any("report_draft" in r for r in failed[1:]))

    def test_multisegment_repair_recovers_latest_candidate_without_replaying(self):
        def incomplete(request, context):
            return repair(request, [{"route": None, "field": "summary", "value": "Retained updated summary"}])
        old, _, _ = self.run_inputs(self.actions()[:-1] + [missing_files, incomplete], max_rounds=5)
        before = {p: p.read_bytes() for p in old.path.rglob("*") if p.is_file()}
        middle, _, _ = self.run_inputs([], resume_from=old.id, prepare_only=True)
        new, proposal, backend = self.run_inputs([tool("machine"), file_repair], resume_from=middle.id, max_rounds=2)
        self.assertIsNotNone(proposal, new.state()["stop_reason"])
        self.assertEqual(proposal["plan"]["summary"], "Retained updated summary")
        self.assertEqual(before, {p: p.read_bytes() for p in old.path.rglob("*") if p.is_file()})
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(new.state()["cumulative_cost"]["calls"], old.state()["cost"]["calls"] + middle.state()["cost"]["calls"] + new.state()["cost"]["calls"])

    def test_cited_lines_near_tail_are_resurfaced_for_repair_without_recall(self):
        (self.model / "README.md").write_text("\n".join(["intro " * 12] * 60 + ["Specific weights documentation."]))
        def bad(request, context):
            value = missing_files(request, context)
            value["routes"][0]["evidence"][1].update(start=61, end=61)
            return value
        old, _, _ = self.run_inputs([tool("read", scope="framework", path="README.md"),
            tool("read", scope="model", path="README.md", lines=80), tool("machine"), bad], max_rounds=4)
        read_record = [c for c in old.records("calls") if c["name"] == "environment.read"][-1]
        self.assertEqual(read_json(old.path / read_record["output"])["lines"][-1]["line"], 61)
        def check(request, context):
            requested = [e for e in request.context["observations"] if e.get("requested_by_draft")]
            self.assertEqual({e["parameters"]["scope"] for e in requested}, {"model", "framework"})
            self.assertTrue(any([61, 61] in e["displayed_ranges"] for e in requested))
            self.assertEqual(request.context["draft_evidence_unavailable"], [])
            return file_repair(request, context)
        new, proposal, backend = self.run_inputs([tool("machine"), check], resume_from=old.id, max_rounds=2)
        self.assertIsNotNone(proposal, new.state()["stop_reason"])
        self.assertEqual(proposal["plan"]["routes"][0]["evidence"][1]["quote"], "Specific weights documentation.")
        self.assertEqual(len(backend.requests), 2)

    def test_accepted_repair_before_proposal_write_failure_recovers_as_draft(self):
        def fail_write(path, value):
            if path.name == "proposal.json":
                raise OSError("disk failure")
            atomic_json(path, value)
        with patch("sherpa.runtime.explore.atomic_json", side_effect=fail_write):
            old, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing_files, file_repair], max_rounds=5)
        self.assertIsNone(proposal)
        self.assertEqual(old.state()["status"], "failed")
        self.assertEqual(old.records("calls")[-1]["status"], "succeeded")
        def retry(request, context):
            self.assertEqual(request.context["report_draft"]["status"], "uncommitted_draft")
            return repair(request, [{"route": None, "field": "summary", "value": "Retry with current evidence"}])
        new, proposal, _ = self.run_inputs([tool("machine"), retry], resume_from=old.id, max_rounds=2)
        self.assertIsNotNone(proposal, new.state()["stop_reason"])
        self.assertEqual(proposal["plan"]["summary"], "Retry with current evidence")

    def test_changed_model_evidence_and_expired_machine_still_block_repair(self):
        old, _, _ = self.run_inputs(self.actions()[:-1] + [missing_files], max_rounds=4)
        new, proposal, _ = self.run_inputs([file_repair], resume_from=old.id, max_rounds=1)
        self.assertIsNone(proposal)
        record = new.records("calls")[-1]
        self.assertIn("query machine", read_json(new.path / record["output"])["error"])
        (self.model / "README.md").write_text("Changed model content.\n")
        later, proposal, _ = self.run_inputs([tool("machine"), file_repair], resume_from=new.id, max_rounds=2)
        self.assertIsNone(proposal)
        self.assertIn("not currently displayed", read_json(later.path / later.records("calls")[-1]["output"])["error"])

    def test_no_weight_selection_is_inferred_and_nonexistent_weight_rejected(self):
        def no_selection(request, context):
            return repair(request, [{"route": None, "field": "summary", "value": "Still not selecting a file"}])
        def wrong(request, context):
            return repair(request, [{"route": 0, "field": "model_files", "value": ["missing.gguf"]}])
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing_files, no_selection, wrong, file_repair], max_rounds=7)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        errors = [read_json(run.path / r["output"])["error"] for r in run.records("calls") if r.get("error_category") == "invalid_response"]
        self.assertIn("model_files", errors[1])
        self.assertIn("missing.gguf", errors[2])

    def test_large_finish_returns_to_model_and_cannot_reuse_previous_candidate_hash(self):
        ids = []
        def oversized(request, context):
            ids.append(request.context["report_draft"]["id"])
            value = dual_finish(request, context)
            for field in ("prerequisites", "steps", "verification", "limitations"):
                value["routes"][0][field] = ["x" * 1400] * 12
            return value
        def recover(request, context):
            self.assertEqual(request.context["report_draft"]["id"], ids[0])
            return file_repair(request, context)
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing_files, oversized, recover], max_rounds=6)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])
        failed = [c for c in run.records("calls") if c.get("error_category") == "invalid_response"]
        self.assertIn("48000", read_json(run.path / failed[-1]["output"])["error"])
        self.assertNotIn("report_draft", failed[-1])
        self.assertEqual(proposal["assembly"]["candidate_id"].split("/")[1], run.records("calls")[-1]["id"])

    def test_cancel_after_successful_repair_before_capture_is_recoverable(self):
        from sherpa.runtime.calls import RunStopped
        real_capture = report_drafts.capture
        def cancel(inv, record):
            if record["status"] == "succeeded":
                raise RunStopped("cancelled", "after successful call before draft write")
            real_capture(inv, record)
        with patch.object(report_drafts, "capture", side_effect=cancel):
            old, proposal, _ = self.run_inputs(self.actions()[:-1] + [missing_files, file_repair], max_rounds=5)
        self.assertIsNone(proposal)
        self.assertEqual(old.state()["status"], "cancelled")
        self.assertNotIn("report_draft", old.records("calls")[-1])
        def check(request, context):
            self.assertEqual(request.context["report_draft"]["status"], "uncommitted_draft")
            self.assertNotIn("model_files", request.context["report_draft"]["routes"][0]["missing_fields"])
            return repair(request, [{"route": None, "field": "summary", "value": "Recovered after cancellation"}])
        new, proposal, _ = self.run_inputs([tool("machine"), check], resume_from=old.id, max_rounds=2)
        self.assertIsNotNone(proposal, new.state()["stop_reason"])
        self.assertEqual(proposal["plan"]["routes"][0]["model_files"], ["weights.gguf"])

    def test_huge_invalid_field_still_produces_a_bounded_repairable_view(self):
        def invalid(request, context):
            candidate = dual_finish(request, context)
            candidate["routes"][0]["model_files"] = ["长" * 8000]
            return candidate
        def corrected(request, context):
            self.assertLess(len(json.dumps(request.context, ensure_ascii=False).encode()), 24000)
            self.assertLessEqual(len(json.dumps(request.context["report_draft"], ensure_ascii=False).encode()), 4500)
            return file_repair(request, context)
        run, proposal, _ = self.run_inputs(self.actions()[:-1] + [invalid, corrected], max_rounds=5)
        self.assertIsNotNone(proposal, run.state()["stop_reason"])

    def test_corrupt_derived_artifact_stops_before_provider(self):
        old, _, _ = self.run_inputs(self.actions()[:-1] + [missing_files], max_rounds=4)
        record = old.records("calls")[-1]
        path = old.path / record["report_draft"]["path"]
        value = read_json(path)
        value["candidate"]["summary"] = "corruption"
        atomic_json(path, value)
        new, proposal, backend = self.run_inputs([], resume_from=old.id)
        self.assertIsNone(proposal)
        self.assertEqual(new.state()["status"], "failed")
        self.assertIn("draft artifact checksum", new.state()["stop_reason"])
        self.assertEqual(backend.requests, [])

    def test_recomputed_artifact_hash_cannot_detach_candidate_from_raw(self):
        old, _, _ = self.run_inputs(self.actions()[:-1] + [missing_files], max_rounds=4)
        record = old.records("calls")[-1]
        value = read_json(old.path / record["report_draft"]["path"])
        value["candidate"]["summary"] = "different action"
        value["candidate_hash"] = digest(value["candidate"])
        atomic_json(old.path / record["report_draft"]["path"], value)
        record["report_draft"]["hash"] = digest(value)
        old.save("calls", record["id"], record, "test_corruption")
        new, proposal, backend = self.run_inputs([], resume_from=old.id)
        self.assertIsNone(proposal)
        self.assertIn("verified action/base", new.state()["stop_reason"])
        self.assertEqual(backend.requests, [])


if __name__ == "__main__":
    unittest.main()
