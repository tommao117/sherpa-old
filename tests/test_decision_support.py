from __future__ import annotations

import copy
import json
import time
import unittest
from types import SimpleNamespace

from sherpa.runtime.decision_support import assemble, context, read_view, task_facts


def document(call_id, scope, texts):
    return {"id": call_id, "tool": "read", "parameters": {"scope": scope, "path": "README.md"},
            "view": {"data": {"path": "README.md", "sha256": "file-hash", "lines": [
                {"line": i + 10, "text": text} for i, text in enumerate(texts)]}, "truncated": False}}


class DecisionSupportTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"source": {"kind": "git", "url": "https://example.invalid/framework", "revision": "release"}, "commit": "fixed"}
        self.inv = SimpleNamespace(
            run=SimpleNamespace(id="new-run"), framework_ref=self.binding, device="gpu",
            tools=SimpleNamespace(binding={"framework": self.binding, "model": {"path": "/model"}}, catalog={}),
            evidence={"code": document("code", "framework", ["A framework fact.", "Second code line."]),
                      "model": document("model", "model", ["A model fact.", "Second model line."])},
            calls=SimpleNamespace(max_calls=8, count=2, deadline=time.monotonic() + 60),
            max_rounds=8, max_delegations=1, delegations=0, history=[], advice=[], continuation=None,
            catalog_offset=0, focus="model", seen=set(), visible_lines={})
        self.action = {"action": "finish", "summary": "Proposed next check", "selected_route": None, "unknowns": ["untested"],
                       "routes": [{"name": "candidate", "evidence": [{"id": "code", "start": 10, "end": 11},
                                                                     {"id": "model", "start": 10, "end": 10}]}]}

    def test_assembly_preserves_claims_and_original_action(self):
        context(self.inv, 1)
        original = copy.deepcopy(self.action)
        result = assemble(self.action, self.inv)
        self.assertEqual(self.action, original)
        route = result["routes"][0]
        self.assertEqual(route["framework_commit"], "fixed")
        self.assertEqual(route["framework_source"], self.binding["source"]["url"])
        self.assertEqual(route["evidence"][0]["quote"], "A framework fact.\nSecond code line.")
        self.assertEqual(result["summary"], self.action["summary"])
        self.assertEqual(result["unknowns"], self.action["unknowns"])
        self.assertIsNone(result["selected_route"])

    def test_line_selection_rejects_invisible_reversed_boolean_or_missing_lines(self):
        context(self.inv, 1)
        for ref in ({"id": "unknown", "start": 10, "end": 10}, {"id": "code", "start": 11, "end": 10},
                    {"id": "code", "start": True, "end": 10}, {"id": "code", "start": 10, "end": 12},
                    {"id": "code", "start": 10, "end": 19}, {"id": "code", "quote": "A framework fact."}):
            with self.subTest(ref=ref):
                action = copy.deepcopy(self.action)
                action["routes"][0]["evidence"] = [ref]
                with self.assertRaises(ValueError):
                    assemble(action, self.inv)

    def test_copy_identity_cannot_override_bound_inputs(self):
        context(self.inv, 1)
        for key in ("framework_commit", "framework_source", "runtime"):
            action = copy.deepcopy(self.action)
            action["routes"][0][key] = "other"
            with self.assertRaisesRegex(ValueError, "identity"):
                assemble(action, self.inv)

    def test_partial_view_cannot_cite_hidden_line_and_recall_reveals_it(self):
        texts = ["line evidence " + "a" * 150 for _ in range(24)]
        self.inv.evidence["code"] = document("code", "framework", texts)
        result = context(self.inv, 1)
        view = next(e for e in result["observations"] if e["id"] == "code")
        self.assertTrue(view["view"]["truncated"])
        action = copy.deepcopy(self.action)
        action["routes"][0]["evidence"][0] = {"id": "code", "start": 30, "end": 30}
        with self.assertRaisesRegex(ValueError, "not currently displayed"):
            assemble(action, self.inv)
        self.inv.focus = "code"
        context(self.inv, 2)
        result = assemble(action, self.inv)
        self.assertEqual(result["routes"][0]["evidence"][0]["quote"], texts[20])

    def test_truncated_original_tool_data_is_never_citable(self):
        self.inv.evidence["code"]["view"] = {"truncated": True, "preview": "A framework fact."}
        self.inv.focus = "code"
        result = context(self.inv, 1)
        self.assertTrue(next(e for e in result["observations"] if e["id"] == "code")["view"]["truncated"])
        self.assertNotIn("code", self.inv.visible_lines)
        with self.assertRaisesRegex(ValueError, "not currently displayed"):
            assemble(self.action, self.inv)

    def test_metadata_and_sparse_range_cannot_be_cited(self):
        context(self.inv, 1)
        self.inv.visible_lines["code"].pop(11)
        self.inv.visible_lines["code"][12] = "Later line"
        action = copy.deepcopy(self.action)
        action["routes"][0]["evidence"][0]["end"] = 12
        with self.assertRaisesRegex(ValueError, "not currently displayed"):
            assemble(action, self.inv)
        self.inv.visible_lines["code"] = {}
        with self.assertRaises(ValueError):
            assemble(self.action, self.inv)

    def test_old_machine_is_not_current_and_failed_refresh_creates_no_fact(self):
        self.inv.evidence["old-machine"] = {"id": "old-machine", "tool": "machine", "parameters": {},
            "origin_run": "old-run", "view": {"data": {"gpus": [{"occupancy": "free"}]}, "truncated": False}}
        self.assertIsNone(task_facts(self.inv)["machine"])
        self.inv.history = [{"action": "tool", "result": {"tool": "machine", "status": "failed"}}]
        self.assertTrue(context(self.inv, 1)["completion"]["machine_refresh_required"])
        self.inv.evidence["current"] = {"id": "current", "tool": "machine", "parameters": {},
            "view": {"data": {"cuda_runtime_verified": False, "reserved": False, "gpus": [{"occupancy": "busy"}],
                              "tools": {"nvcc": False}, "tool_detection": {"false_means": "not found on current PATH"}}, "truncated": False}}
        result = context(self.inv, 2)
        self.assertFalse(result["completion"]["machine_refresh_required"])
        self.assertEqual(result["facts"]["machine"]["evidence_id"], "current")
        self.assertFalse(result["facts"]["machine"]["observation"]["cuda_runtime_verified"])
        self.assertIn("PATH", result["facts"]["machine"]["observation"]["tool_detection"]["false_means"])

    def test_large_machine_fact_is_complete_not_marked_seen_from_truncated_preview(self):
        machine = {"gpus": [{"name": "large fact " + "x" * 7000}], "cuda_runtime_verified": False}
        self.inv.evidence["machine"] = {"id": "machine", "tool": "machine", "parameters": {},
                                       "view": {"data": machine, "truncated": False}}
        result = context(self.inv, 1)
        self.assertEqual(result["facts"]["machine"]["observation"], machine)
        self.assertIn("machine", self.inv.seen)
        self.assertLessEqual(len(json.dumps(result).encode()), 24000)
        self.inv.evidence["machine-new"] = {"id": "machine-new", "tool": "machine", "parameters": {},
                                           "view": {"preview": "partial", "truncated": True}}
        result = context(self.inv, 2)
        self.assertIsNone(result["facts"]["machine"])
        self.assertNotIn("machine-new", self.inv.seen)
        self.assertTrue(result["completion"]["machine_refresh_required"])

    def test_refreshed_first_page_invalidates_old_trailing_pages(self):
        for call_id, offset, name, next_offset in (("first", 0, "first.gguf", 40), ("tail", 40, "removed.gguf", None),
                                                 ("refresh", 0, "remaining.gguf", None)):
            self.inv.evidence[call_id] = {"id": call_id, "tool": "list", "parameters": {"scope": "model", "path": ".", "offset": offset},
                "view": {"data": {"path": ".", "next_offset": next_offset, "entries": [{"name": name, "kind": "file"}]}, "truncated": False}}
        facts = task_facts(self.inv)
        self.assertEqual([f["path"] for f in facts["model_inventory"]], ["remaining.gguf"])
        self.assertEqual(len(facts["inventory_pages"]), 1)
        self.assertIn("tail", self.inv.evidence)

    def test_inventory_pages_keep_paths_and_freshest_observation(self):
        for call_id, path, name in (("a", ".", "old.gguf"), ("b", ".", "new.gguf"), ("c", "nested", "nested.gguf")):
            self.inv.evidence[call_id] = {"id": call_id, "tool": "list", "parameters": {"scope": "model", "path": path},
                "view": {"data": {"path": path, "next_offset": 40, "entries": [{"name": name, "kind": "file", "bytes": 10}]}, "truncated": False}}
        facts = task_facts(self.inv)
        self.assertEqual({f["path"] for f in facts["model_inventory"]}, {"new.gguf", "nested/nested.gguf"})
        self.assertEqual({f["evidence_id"] for f in facts["model_inventory"]}, {"b", "c"})
        self.assertTrue(all(p["next_offset"] == 40 for p in facts["inventory_pages"]))

    def test_large_context_remains_bounded_and_never_mutates_trace(self):
        for i in range(200):
            self.inv.evidence[f"code{i}"] = document(f"code{i}", "framework", ["内容" * 60] * 50)
        before = copy.deepcopy(self.inv.evidence)
        result = context(self.inv, 1)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False).encode()), 24000)
        self.assertEqual(before, self.inv.evidence)
        self.assertTrue(result["omitted_evidence_ids"])
        self.assertIsNotNone(result["catalog_next_offset"])
        self.assertIn("code", self.inv.visible_lines)
        self.assertIn("model", self.inv.visible_lines)

    def test_empty_navigation_and_all_skipped_are_visible_but_not_citable(self):
        for matched, skipped in (([], []), ([10], [10])):
            with self.subTest(matched=matched):
                self.inv.evidence["search"] = {"id": "search", "tool": "search", "parameters": {"scope": "framework", "path": "large.md", "query": "target"},
                    "view": {"data": {"path": "large.md", "lines": [], "matched_lines": matched, "skipped_lines": skipped}, "truncated": False}}
                self.inv.focus = "search"
                result = context(self.inv, 1)
                view = next(e for e in result["observations"] if e["id"] == "search")
                self.assertEqual(view["navigation"]["matched_lines"], matched)
                self.assertEqual(view["navigation"]["skipped_lines"], skipped)
                self.assertNotIn("search", self.inv.visible_lines)

    def test_focused_navigation_tail_is_visible_and_partial_view_has_working_guidance(self):
        entry = document("search", "framework", ["a" * 200 for _ in range(32)])
        entry["tool"] = "search"
        self.inv.evidence["search"] = entry
        self.inv.focus = "search"
        result = context(self.inv, 1)
        view = next(e for e in result["observations"] if e["id"] == "search")
        self.assertEqual(len(view["view"]["data"]["lines"]), 32)
        self.assertIsNone(view["more"])
        self.assertIn(41, self.inv.visible_lines["search"])
        short = read_view(entry, 1000, focused=True)
        self.assertIn("do not repeatedly recall", short["more"])
        self.assertEqual(short["first_omitted_line"], 14)

    def test_read_view_preserves_complete_lines_only(self):
        entry = document("id", "model", ["短句", "长" * 1000, "after"])
        view = read_view(entry, 200)
        self.assertEqual(view["view"]["data"]["lines"], [{"line": 10, "text": "短句"}])
        self.assertTrue(view["view"]["truncated"])
        self.assertEqual(view["displayed_ranges"], [[10, 10]])


if __name__ == "__main__":
    unittest.main()
