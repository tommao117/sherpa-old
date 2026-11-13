from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sherpa.platform.exploration import EvidenceTools, bounded_view
from sherpa.runtime.decision_support import assemble, context
from sherpa.runtime.evidence_windows import select_lines, window_views


class NavigationToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sherpa-navigation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.tools = EvidenceTools(self.source, self.root, lambda: None)
        self.addCleanup(self.tools.close)

    def write(self, text, path="guide.md"):
        (self.source / path).write_text(text)
        return path

    def test_literal_case_insensitive_search_keeps_true_lines_and_pagination(self):
        self.write("preamble\nA.*B literal\na different b\na.*b again\ntrailer\n")
        result = self.tools.execute("search", {"path": "guide.md", "query": "a.*b", "around": 0, "max_matches": 1})
        self.assertEqual(result["matched_lines"], [2])
        self.assertEqual(result["lines"], [{"line": 2, "text": "A.*B literal"}])
        self.assertEqual(result["next_line"], 4)
        next_page = self.tools.execute("search", {"path": "guide.md", "query": "a.*b", "start": result["next_line"], "around": 0})
        self.assertEqual(next_page["matched_lines"], [4])
        self.assertIsNone(next_page["next_line"])
        self.assertEqual(result["sha256"], next_page["sha256"])

    def test_context_windows_deduplicate_and_no_match_is_not_failure(self):
        self.write("begin\nneedle one\nneedle two\nend\n")
        result = self.tools.execute("search", {"path": "guide.md", "query": "needle", "around": 2})
        self.assertEqual([l["line"] for l in result["lines"]], [1, 2, 3, 4])
        no_match = self.tools.execute("search", {"path": "guide.md", "query": "absent"})
        self.assertEqual(no_match["lines"], [])
        self.assertEqual(no_match["matched_lines"], [])
        self.assertIsNone(no_match["next_line"])

    def test_outline_ignores_fenced_code_and_only_returns_actual_headings(self):
        text = "# First\n```python\n## Not a heading\n```\nBody\n  ## Second\n~~~text\n# Not either\n~~~\n### Third\nsetext\n======\n"
        self.write(text)
        result = self.tools.execute("outline", {"path": "guide.md"})
        self.assertEqual(result["lines"], [{"line": 1, "text": "# First"}, {"line": 6, "text": "  ## Second"}, {"line": 10, "text": "### Third"}])
        tail = self.tools.execute("outline", {"path": "guide.md", "start": 3})
        self.assertEqual([l["line"] for l in tail["lines"]], [6, 10])
        self.assertIsNone(result["next_line"])

    def test_outline_pagination_and_unicode_long_lines_stay_bounded(self):
        self.write("\n".join(f"## Section {i}" for i in range(35)))
        result = self.tools.execute("outline", {"path": "guide.md"})
        self.assertEqual(len(result["lines"]), 30)
        self.assertEqual(result["next_line"], 31)
        self.write("key " + "长" * 3000 + "\nkey short\n")
        result = self.tools.execute("search", {"path": "guide.md", "query": "key", "around": 0})
        self.assertEqual(result["skipped_lines"], [1])
        self.assertEqual(result["lines"], [{"line": 2, "text": "key short"}])
        self.assertFalse(bounded_view(result)["truncated"])

    def test_same_path_boundaries_for_all_navigation_tools(self):
        self.write("secret", "weights.gguf")
        outside = self.root / "outside.md"
        outside.write_text("private")
        (self.source / "linked.md").symlink_to(outside)
        import os
        os.mkfifo(self.source / "fifo.md")
        for tool in ("search", "outline"):
            for path in ("weights.gguf", "../outside.md", "linked.md", "fifo.md", ".env"):
                args = {"path": path, **({"query": "x"} if tool == "search" else {})}
                with self.subTest(tool=tool, path=path), self.assertRaises((ValueError, OSError)):
                    self.tools.execute(tool, args)
        for bad in ("", "\n", "x" * 121, 123):
            with self.assertRaises(ValueError):
                self.tools.execute("search", {"path": "guide.md", "query": bad})


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.entry = {"id": "code", "tool": "search", "parameters": {"scope": "framework", "path": "guide.md"},
                      "view": {"data": {"path": "guide.md", "lines": [{"line": 50, "text": "CUDA backend requirements"},
                       {"line": 51, "text": "Build the selected backend"}, {"line": 90, "text": "Other heading"}]}, "truncated": False}}
        self.evidence = {"code": self.entry}
        self.visible = {"code": {50: "CUDA backend requirements", 51: "Build the selected backend"}}
        self.ref = {"id": "code", "start": 50, "end": 51}

    def test_pin_requires_displayed_complete_contiguous_lines_and_budget(self):
        self.assertEqual(len(window_views([self.ref], self.evidence, self.visible)), 1)
        for ref in ({"id": "code", "start": 90, "end": 90}, {"id": "code", "start": 51, "end": 90},
                    {"id": "code", "start": 52, "end": 51}, {"id": "code", "start": True, "end": 51}):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                window_views([ref], self.evidence, self.visible)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            window_views([self.ref, self.ref], self.evidence, self.visible)
        self.assertEqual(window_views([], self.evidence, self.visible), [])
        with self.assertRaises(ValueError):
            window_views([self.ref] * 7, self.evidence, self.visible)

    def test_pinned_evidence_survives_focus_changes_and_is_citable(self):
        model = copy.deepcopy(self.entry)
        model.update(id="model", parameters={"scope": "model", "path": "model.md"})
        self.evidence["model"] = model
        for i in range(10):
            self.evidence[f"noise-{i}"] = {"id": f"noise-{i}", "tool": "read", "parameters": {"scope": "framework", "path": f"doc{i}.md"},
                "view": {"data": {"lines": [{"line": n, "text": "unrelated context " * 10} for n in range(1, 40)]}, "truncated": False}}
        binding = {"source": {"kind": "local", "path": "/framework"}, "commit": "fixed"}
        inv = SimpleNamespace(run=SimpleNamespace(id="run"), framework_ref=binding, device="gpu",
            tools=SimpleNamespace(binding={"framework": binding, "model": {"path": "/model"}}, catalog={}),
            evidence=self.evidence, calls=SimpleNamespace(max_calls=20, count=0, deadline=__import__("time").monotonic() + 60),
            max_rounds=12, max_delegations=1, delegations=0, history=[], advice=[], continuation=None, catalog_offset=0,
            focus="noise-9", seen=set(), visible_lines={}, pinned_windows=[self.ref, {"id": "model", "start": 90, "end": 90}])
        before = copy.deepcopy(inv.evidence)
        c = context(inv, 1)
        self.assertEqual(c["observations"][0]["id"], "code")
        self.assertEqual(inv.visible_lines["code"][51], "Build the selected backend")
        result = assemble({"action": "finish", "routes": [{"evidence": [self.ref]}]}, inv)
        self.assertIn("CUDA backend", result["routes"][0]["evidence"][0]["quote"])
        self.assertEqual(before, inv.evidence)
        self.assertLessEqual(len(json.dumps(c).encode()), 24000)
        inv.pinned_windows = []
        self.assertEqual(context(inv, 2)["retained_windows"], [])
        self.assertIn("code", inv.evidence)

    def test_expert_and_legacy_quote_cannot_join_sparse_lines(self):
        from sherpa.runtime.explore import Investigation
        inv = SimpleNamespace(evidence=self.evidence, seen={"code"}, bound_inputs=True)
        Investigation.citations(inv, [{"id": "code", "quote": "CUDA backend requirements\nBuild the selected backend"}])
        for tool in ("search", "outline"):
            self.entry["tool"] = tool
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, "quote absent"):
                Investigation.citations(inv, [{"id": "code", "quote": "Build the selected backend\nOther heading"}])
            Investigation.citations(inv, [{"id": "code", "quote": "Other heading"}])

    def test_sparse_search_and_outline_do_not_create_contiguous_citations(self):
        with self.assertRaises(ValueError):
            select_lines({"id": "code", "start": 50, "end": 52}, self.evidence)
        self.entry["tool"] = "outline"
        with self.assertRaises(ValueError):
            select_lines({"id": "code", "start": 51, "end": 53}, self.evidence)
        self.assertEqual(select_lines({"id": "code", "start": 90, "end": 90}, self.evidence)[0]["line"], 90)


if __name__ == "__main__":
    unittest.main()
