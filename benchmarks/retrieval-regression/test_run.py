"""Tests for the retrieval regression probe.

Scoring, fixture-contract and stub-binary probe tests are pure and run
everywhere. The end-to-end test needs a built binary at
ENTIRE_BENCHMARK_BRAIN_BINARY and is skipped otherwise.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as probe  # noqa: E402


def measurement(**overrides):
    base = {
        "tasks": 22,
        "mean_recall": 1.0,
        "top_rank_hit_rate": 1.0,
        "recall_failures": 0,
        "recall_failure_sample": None,
        "eval_disagreements": [],
        "mean_precision": 0.2,
        "mean_useful_per_1k": 6.0,
        "brief_hit_rate": 1.0,
        "brief_max_prefix_fraction": 0.03,
        "brief_max_prefix_bytes": 120,
        "brief_failures": 0,
        "brief_failure_sample": None,
        "brief_mean_bytes": 4000.0,
    }
    base.update(overrides)
    return base


class ScoreTests(unittest.TestCase):
    def test_clean_branch_is_100_without_markers(self):
        monitor = probe.score(measurement(), measurement())
        self.assertEqual(monitor["value"], 100)
        self.assertNotIn("REGRESSION", monitor["rationale"])
        self.assertFalse(monitor["regressed"])

    def test_value_is_the_weakest_of_recall_top_rank_and_brief(self):
        monitor = probe.score(measurement(mean_recall=0.95, top_rank_hit_rate=0.7, brief_hit_rate=0.9), None)
        self.assertEqual(monitor["value"], 70)

    def test_regression_against_base_caps_the_value(self):
        monitor = probe.score(measurement(mean_recall=0.95), measurement())
        self.assertEqual(monitor["value"], probe.REGRESSION_CAP)
        self.assertIn("fewer labeled facts", monitor["rationale"])

    def test_every_regression_flag_is_reported_not_only_the_first(self):
        monitor = probe.score(
            measurement(mean_recall=0.9, top_rank_hit_rate=0.8, mean_useful_per_1k=3.0, brief_mean_bytes=9000.0),
            measurement(),
        )
        for marker in ("fewer labeled facts", "rank lower", "useful-per-1k fell", "grew x2.2"):
            self.assertIn(marker, monitor["rationale"])

    def test_status_block_before_facts_zeroes_by_fraction(self):
        monitor = probe.score(measurement(brief_max_prefix_fraction=0.8, brief_max_prefix_bytes=3000), measurement())
        self.assertEqual(monitor["value"], 0)
        self.assertIn("3000 bytes (80%) before its facts", monitor["rationale"])

    def test_status_block_before_facts_zeroes_by_absolute_bytes(self):
        # 12.8 KB ahead of a 30 KB evidence block is only 30%, still over budget.
        monitor = probe.score(measurement(brief_max_prefix_fraction=0.3, brief_max_prefix_bytes=12800), measurement())
        self.assertEqual(monitor["value"], 0)

    def test_float_noise_is_not_a_regression(self):
        monitor = probe.score(measurement(mean_recall=1.0 - 1e-12), measurement())
        self.assertFalse(monitor["regressed"])

    def test_partial_brief_failures_are_named_and_cap(self):
        monitor = probe.score(
            measurement(brief_hit_rate=0.5, brief_failures=11, brief_failure_sample="unknown flag: --branch"),
            None,
        )
        self.assertEqual(monitor["value"], probe.REGRESSION_CAP)
        self.assertIn("brief failed on 11 of 22 tasks (unknown flag: --branch)", monitor["rationale"])
        self.assertNotIn("before its facts", monitor["rationale"])

    def test_eval_disagreement_is_flagged(self):
        monitor = probe.score(measurement(eval_disagreements=["t1", "t2"]), None)
        self.assertEqual(monitor["value"], probe.REGRESSION_CAP)
        self.assertIn("different recall than the ids it returns on 2 tasks", monitor["rationale"])

    def test_probe_error_is_zero_with_reason(self):
        monitor = probe.score(None, None, "go build failed")
        self.assertEqual(monitor["value"], 0)
        self.assertIn("could not run: go build failed", monitor["rationale"])

    def test_missing_base_scores_branch_alone(self):
        monitor = probe.score(measurement(mean_recall=0.5), None)
        self.assertEqual(monitor["value"], 50)
        self.assertNotIn("vs base", monitor["rationale"])


class PrefixTests(unittest.TestCase):
    def test_status_first_is_measured_even_when_status_nests_a_facts_key(self):
        packet = {"status": {"facts": {"count": 3}, "padding": "x" * 5000}, "task": "q", "facts": [{"id": "fact:a"}]}
        fraction, nbytes = probe.facts_prefix(packet)
        self.assertGreater(fraction, probe.BRIEF_PREFIX_LIMIT)
        self.assertGreater(nbytes, 5000)

    def test_facts_early_is_small(self):
        packet = {"generated_at": "t", "task": "q", "facts": [{"id": "fact:a"}], "status": {"padding": "x" * 5000}}
        fraction, nbytes = probe.facts_prefix(packet)
        self.assertLess(fraction, 0.05)
        self.assertLess(nbytes, 100)

    def test_missing_facts_section_is_none(self):
        self.assertIsNone(probe.facts_prefix({"status": {"facts": {"count": 3}}}))


class StubBinaryProbeTests(unittest.TestCase):
    """Drive the probes with a fake entire-brain that prints canned packets."""

    TASKS = [
        {"id": "a", "task": "question a", "k": 3, "relevant": ["fact:a"]},
        {"id": "b", "task": "question b", "k": 3, "relevant": ["fact:b1", "fact:b2"]},
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="brain-retrieval-stub-")
        self.addCleanup(self.tmp.cleanup)
        self.store = probe.FixtureStore(Path(self.tmp.name) / "s", "https://example.invalid/o/r", "main")
        self.store.create()

    def stub(self, body: str) -> Path:
        path = Path(self.tmp.name) / "entire-brain"
        path.write_text("#!/usr/bin/env python3\nimport json, sys\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_recall_probe_computes_its_own_recall_and_top_rank(self):
        binary = self.stub(
            """
            q = sys.argv[2]
            order = {"question a": ["fact:x", "fact:a"], "question b": ["fact:b1", "fact:b2", "fact:x"]}
            print(json.dumps({"facts": [{"id": i} for i in order[q]]}))
            """
        )
        out = probe.recall_probe(self.store, binary, self.TASKS)
        self.assertEqual(out["mean_recall"], 1.0)
        self.assertEqual(out["top_rank_hit_rate"], 0.5)  # task a has a distractor at rank 1
        self.assertEqual(out["failures"], 0)

    def test_brief_probe_catches_status_first_with_nested_facts_key(self):
        binary = self.stub(
            """
            print(json.dumps({"status": {"facts": {"count": 2}, "pad": "x" * 20000},
                              "facts": [{"id": "fact:a"}, {"id": "fact:b1"}, {"id": "fact:b2"}]}))
            """
        )
        out = probe.brief_probe(self.store, binary, self.TASKS)
        self.assertEqual(out["hit_rate"], 1.0)
        self.assertGreater(out["max_prefix_fraction"], probe.BRIEF_PREFIX_LIMIT)
        self.assertGreater(out["max_prefix_bytes"], probe.BRIEF_PREFIX_BYTES_LIMIT)

    def test_brief_probe_reports_failures_as_failures(self):
        binary = self.stub(
            """
            if sys.argv[2] == "question a":
                sys.stderr.write("unknown flag\\n"); sys.exit(1)
            print(json.dumps({"status": {}}))  # no facts section
            """
        )
        out = probe.brief_probe(self.store, binary, self.TASKS)
        self.assertEqual(out["failures"], 2)
        self.assertEqual(out["hit_rate"], 0.0)
        self.assertEqual(out["max_prefix_fraction"], 0.0)
        self.assertIn("unknown flag", out["failure_sample"])

    def test_store_env_is_isolated(self):
        os.environ["ENTIRE_BRAIN_WEBHOOK_URL"] = "https://example.invalid/hook"
        os.environ["GIT_DIR"] = "/nonexistent"
        self.addCleanup(os.environ.pop, "ENTIRE_BRAIN_WEBHOOK_URL", None)
        self.addCleanup(os.environ.pop, "GIT_DIR", None)
        env = self.store.env()
        self.assertNotIn("ENTIRE_BRAIN_WEBHOOK_URL", env)
        self.assertNotIn("GIT_DIR", env)
        self.assertEqual(env["ENTIRE_BRAIN_NO_EGRESS"], "1")
        self.assertEqual(env["ENTIRE_BRAIN_LOCAL_ONLY"], "1")
        self.assertTrue(env["ENTIRE_PLUGIN_DATA_DIR"].startswith(self.tmp.name))


class FixtureContractTests(unittest.TestCase):
    def test_every_question_names_a_fixture_fact(self):
        facts = {f["key"] for f in probe.load_json(probe.FACTS_PATH)["facts"]}
        for q in probe.load_json(probe.QUESTIONS_PATH):
            self.assertTrue(q["relevant_keys"], f"question {q['id']} has no labels")
            for key in q["relevant_keys"]:
                self.assertIn(key, facts, f"question {q['id']} names unknown fact {key}")

    def test_frozen_tasks_match_questions(self):
        questions = {q["id"]: q for q in probe.load_json(probe.QUESTIONS_PATH)}
        tasks = probe.load_json(probe.TASKS_PATH)
        self.assertEqual(set(questions), {t["id"] for t in tasks})
        for task in tasks:
            # A reworded question without a re-freeze would otherwise go unnoticed.
            self.assertEqual(task["task"], questions[task["id"]]["task"], f"{task['id']}: run `run.py freeze`")
            self.assertEqual(task["k"], questions[task["id"]].get("k", 5))
            self.assertEqual(len(task["relevant"]), len(questions[task["id"]]["relevant_keys"]))
            self.assertTrue(all(fid.startswith("fact:") for fid in task["relevant"]))
            self.assertEqual(task["label_source"], "human")
        probe.check_fixture_contract(probe.load_json(probe.FACTS_PATH), tasks)

    def test_contract_rejects_unlabeled_task(self):
        with self.assertRaises(probe.ProbeError):
            probe.check_fixture_contract({"facts": [{}]}, [{"id": "x", "k": 5, "relevant": []}])


@unittest.skipUnless(os.environ.get("ENTIRE_BENCHMARK_BRAIN_BINARY"), "set ENTIRE_BENCHMARK_BRAIN_BINARY to run")
class EndToEndTests(unittest.TestCase):
    def test_measure_known_good_binary_is_clean(self):
        binary = Path(os.environ["ENTIRE_BENCHMARK_BRAIN_BINARY"]).resolve()
        with tempfile.TemporaryDirectory(prefix="brain-retrieval-test-") as tmp:
            report = probe.measure(binary, "branch", Path(tmp))
        self.assertEqual(report["facts_seeded"], len(probe.load_json(probe.FACTS_PATH)["facts"]))
        self.assertEqual(report["tasks"], len(probe.load_json(probe.TASKS_PATH)))
        monitor = probe.score(report, None)
        self.assertEqual(monitor["value"], 100, json.dumps(report["recall_rows"], indent=1))
        self.assertFalse(monitor["regressed"], monitor["rationale"])
        self.assertEqual(report["eval_disagreements"], [])
        self.assertLess(report["brief_max_prefix_bytes"], probe.BRIEF_PREFIX_BYTES_LIMIT)


if __name__ == "__main__":
    unittest.main()
