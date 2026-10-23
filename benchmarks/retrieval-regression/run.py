#!/usr/bin/env python3
"""Retrieval regression probe for Entire Brain.

Builds an isolated brain store from a committed fixture, reads the labeled
memories back through `recall`, `memories eval` and `brief`, and compares the
branch binary with the base-branch binary. The result is a 0-100 trail
monitor value plus a one-line rationale.

This is a regression signal, not evidence. See README.md for what it does
and does not claim, and for how the value is defined.

Subcommands:
  freeze   seed the fixture with one binary and write fixture/tasks.json
  measure  seed + probes for one binary, print the report
  compare  build the working tree and a base ref, measure both, print the
           report and then the monitor object as the last stdout line
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixture"
FACTS_PATH = FIXTURE_DIR / "facts.json"
QUESTIONS_PATH = FIXTURE_DIR / "questions.json"
TASKS_PATH = FIXTURE_DIR / "tasks.json"

# The #335 shape: a status block ahead of the evidence. Agents truncate on an
# absolute budget, so both an absolute and a relative limit apply.
BRIEF_PREFIX_LIMIT = 0.5
BRIEF_PREFIX_BYTES_LIMIT = 8192
# A brief packet that grows past this multiple of base is flagged.
BRIEF_BYTES_GROWTH_LIMIT = 2.0
# A flagged regression against base caps the value here so the dip is visible
# on the trail, not only inside the rationale.
REGRESSION_CAP = 50

# Means over a few dozen tasks are rationals with small denominators;
# anything smaller than this is float noise, not a change.
EPS = 1e-9

# Environment the fixture store and the brain binary may see. Everything else,
# ENTIRE_* and GIT_* included, is dropped so a developer's webhook, hosted
# token, embedder or GIT_DIR cannot leak into or out of the measurement.
ENV_PASSTHROUGH = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "SystemRoot")


class ProbeError(RuntimeError):
    """A step that makes the measurement meaningless (build, seed, contract)."""


# --------------------------------------------------------------------------
# helpers


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def run(argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a process; a missing executable or a timeout is a ProbeError, not a traceback."""
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProbeError(f"{argv[0]} is not available: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"{' '.join(argv[:3])} exceeded {timeout}s") from exc
    except OSError as exc:
        raise ProbeError(f"{argv[0]} could not start: {exc}") from exc


def isolated_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def repo_root() -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, env=os.environ.copy())
    if proc.returncode != 0:
        raise ProbeError(f"not inside a git repository: {proc.stderr.strip()[:200]}")
    return Path(proc.stdout.strip())


# --------------------------------------------------------------------------
# isolated store


class FixtureStore:
    """A throwaway repository plus plugin dirs that one binary writes into."""

    def __init__(self, root: Path, origin: str, branch: str) -> None:
        self.root = root
        self.repo = root / "repo"
        self.xdg = root / "xdg"
        self.origin = origin
        self.branch = branch

    def create(self) -> None:
        self.repo.mkdir(parents=True)
        for name in ("config", "data", "state", "cache"):
            (self.xdg / name).mkdir(parents=True)
        env = isolated_env()
        steps = [
            ["git", "init", "-q", "-b", self.branch],
            ["git", "remote", "add", "origin", self.origin],
            [
                "git",
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "retrieval regression fixture",
            ],
        ]
        for argv in steps:
            proc = run(argv, cwd=self.repo, env=env)
            if proc.returncode != 0:
                raise ProbeError(f"fixture git step failed: {' '.join(argv)}: {proc.stderr.strip()[:200]}")

    def env(self) -> dict[str, str]:
        env = isolated_env()
        env["ENTIRE_REPO_ROOT"] = str(self.repo)
        env["ENTIRE_PLUGIN_CONFIG_DIR"] = str(self.xdg / "config")
        env["ENTIRE_PLUGIN_DATA_DIR"] = str(self.xdg / "data")
        env["ENTIRE_PLUGIN_STATE_DIR"] = str(self.xdg / "state")
        env["ENTIRE_PLUGIN_CACHE_DIR"] = str(self.xdg / "cache")
        env["ENTIRE_BRAIN_NO_EGRESS"] = "1"
        env["ENTIRE_BRAIN_LOCAL_ONLY"] = "1"
        return env

    def brain(self, binary: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return run([str(binary), *args], cwd=self.repo, env=self.env(), timeout=timeout)


def seed(store: FixtureStore, binary: Path, facts: dict[str, Any]) -> dict[str, str]:
    """Record every fixture fact; return key -> fact id."""
    ids: dict[str, str] = {}
    for fact in facts["facts"]:
        proc = store.brain(
            binary,
            "remember",
            fact["text"],
            "--path",
            fact["path"],
            "--kind",
            fact["kind"],
            "--branch",
            store.branch,
            "--agent",
            "none",
            "--json",
        )
        if proc.returncode != 0:
            raise ProbeError(f"remember failed for {fact['key']}: {proc.stderr.strip()[:300]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"remember emitted no JSON for {fact['key']}: {exc}") from exc
        fact_id = str(payload.get("id") or "").strip() if isinstance(payload, dict) else ""
        if not fact_id:
            raise ProbeError(f"remember returned no id for {fact['key']}")
        ids[fact["key"]] = fact_id
    return ids


# --------------------------------------------------------------------------
# probes


def json_document(proc: subprocess.CompletedProcess[str], what: str) -> dict[str, Any] | None:
    """One JSON object from stdout, or None with no exception."""
    if proc.returncode != 0:
        return None
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def surfaced_ids(packet: dict[str, Any]) -> list[str]:
    return [str(f.get("id")) for f in packet.get("facts") or [] if isinstance(f, dict)]


def recall_probe(store: FixtureStore, binary: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall@k and top-rank computed here from the ids `recall` returns.

    This never trusts a metric the binary under test computed about itself:
    the ids come from the binary, the arithmetic does not.
    """
    rows = []
    for task in tasks:
        relevant = list(task["relevant"])
        k = int(task["k"])
        proc = store.brain(binary, "recall", task["task"], "--branch", store.branch, "--k", str(k), "--json")
        packet = json_document(proc, "recall")
        row: dict[str, Any] = {"id": task["id"], "ok": packet is not None, "recall": 0.0, "top_rank": False}
        if packet is None:
            row["error"] = (proc.stderr.strip() or proc.stdout.strip())[:200] or f"exit {proc.returncode}"
            rows.append(row)
            continue
        got = surfaced_ids(packet)
        top_k = set(got[:k])
        row["recall"] = sum(1 for fid in relevant if fid in top_k) / len(relevant)
        row["top_rank"] = all(fid in set(got[: len(relevant)]) for fid in relevant)
        rows.append(row)
    failures = [r for r in rows if not r["ok"]]
    return {
        "tasks": len(rows),
        "mean_recall": (sum(r["recall"] for r in rows) / len(rows)) if rows else 0.0,
        "top_rank_hit_rate": (sum(1 for r in rows if r["top_rank"]) / len(rows)) if rows else 0.0,
        "failures": len(failures),
        "failure_sample": failures[0]["error"] if failures else None,
        "rows": rows,
    }


def eval_facts(store: FixtureStore, binary: Path, tasks_path: Path) -> dict[str, Any]:
    """The product's own eval, kept for its token accounting (useful per 1k)."""
    proc = store.brain(
        binary,
        "memories",
        "eval",
        "--tasks",
        str(tasks_path),
        "--retriever",
        "facts",
        "--branch",
        store.branch,
        "--agent",
        "none",
        "--json",
        timeout=300,
    )
    if proc.returncode != 0:
        raise ProbeError(f"memories eval failed: {proc.stderr.strip()[:300]}")
    doc = json_document(proc, "eval")
    if doc is None:
        raise ProbeError("memories eval emitted no JSON object")
    return doc


def facts_prefix(packet: dict[str, Any]) -> tuple[float, int] | None:
    """(fraction, bytes) of the packet that precedes the top-level `facts` section.

    Measured on the re-serialized top-level sections in document order, not by
    searching the raw bytes: the status section nests its own "facts" key, so a
    byte search would find that first and hide the #335 layout. None when the
    packet has no top-level facts section.
    """
    keys = list(packet.keys())
    if "facts" not in keys:
        return None
    before = {k: packet[k] for k in keys[: keys.index("facts")]}
    total = len(json.dumps(packet).encode("utf-8"))
    before_bytes = len(json.dumps(before).encode("utf-8"))
    return ((before_bytes / total) if total else 1.0, before_bytes)


def brief_probe(store: FixtureStore, binary: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the brief packet carry the labeled facts, and how far in do they sit?

    A brief that fails or emits no facts section is a failure row: it counts
    against the hit rate and is reported as a failure, never as an offset.
    """
    rows = []
    for task in tasks:
        proc = store.brain(binary, "brief", task["task"], "--json")
        packet = json_document(proc, "brief")
        row: dict[str, Any] = {
            "id": task["id"],
            "ok": False,
            "hit": False,
            "prefix_fraction": None,
            "prefix_bytes": None,
            "bytes": len(proc.stdout.encode("utf-8")),
        }
        if packet is None:
            row["error"] = (proc.stderr.strip() or proc.stdout.strip())[:200] or f"exit {proc.returncode}"
            rows.append(row)
            continue
        prefix = facts_prefix(packet)
        if prefix is None:
            row["error"] = "brief packet has no top-level facts section"
            rows.append(row)
            continue
        surfaced = set(surfaced_ids(packet))
        row["ok"] = True
        row["hit"] = all(fact_id in surfaced for fact_id in task["relevant"])
        row["prefix_fraction"], row["prefix_bytes"] = prefix
        rows.append(row)
    failures = [r for r in rows if not r["ok"]]
    ok_rows = [r for r in rows if r["ok"]]
    return {
        "tasks": len(rows),
        "hit_rate": (sum(1 for r in rows if r["hit"]) / len(rows)) if rows else 0.0,
        "max_prefix_fraction": max((r["prefix_fraction"] for r in ok_rows), default=0.0),
        "max_prefix_bytes": max((r["prefix_bytes"] for r in ok_rows), default=0),
        "failures": len(failures),
        "failure_sample": failures[0]["error"] if failures else None,
        "mean_bytes": (sum(r["bytes"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0.0,
        "rows": rows,
    }


def check_fixture_contract(facts: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    if not facts.get("facts"):
        raise ProbeError("fixture/facts.json has no facts")
    if not tasks:
        raise ProbeError("fixture/tasks.json has no tasks")
    for task in tasks:
        if not task.get("relevant"):
            raise ProbeError(f"task {task.get('id')} has no relevant ids; an unlabeled task would always count as a hit")
        if int(task.get("k", 0)) < len(task["relevant"]):
            raise ProbeError(f"task {task.get('id')} has k smaller than its label count")


def measure(binary: Path, label: str, workdir: Path) -> dict[str, Any]:
    """Seed a fresh store with `binary` and measure it. Raises ProbeError on contract breaks."""
    facts = load_json(FACTS_PATH)
    tasks = load_json(TASKS_PATH)
    check_fixture_contract(facts, tasks)
    if not binary.is_file():
        raise ProbeError(f"{label}: binary not found at {binary}")
    store = FixtureStore(workdir / label, facts["origin"], facts["branch"])
    store.create()
    ids = seed(store, binary, facts)

    expected = {fact_id for task in tasks for fact_id in task["relevant"]}
    missing = sorted(expected - set(ids.values()))
    if missing:
        raise ProbeError(
            f"{label}: frozen task labels reference {len(missing)} fact id(s) the seeded store does not contain "
            f"(first: {missing[0]}); fact id derivation or the fixture changed, re-run `run.py freeze` deliberately"
        )

    recall = recall_probe(store, binary, tasks)
    if recall["failures"] == recall["tasks"]:
        raise ProbeError(f"{label}: recall failed on every task ({recall['failure_sample']})")
    brief = brief_probe(store, binary, tasks)
    if brief["failures"] == brief["tasks"]:
        raise ProbeError(f"{label}: brief failed on every task ({brief['failure_sample']})")

    summary = eval_facts(store, binary, TASKS_PATH)
    results = summary.get("results") or []
    labeled = [r for r in results if r.get("labeled")]
    if len(labeled) != len(tasks):
        raise ProbeError(f"{label}: eval labeled {len(labeled)} of {len(tasks)} tasks; the label contract changed")
    eval_recall_by_id = {r.get("id"): float(r.get("recall", 0.0)) for r in results}
    disagreements = [
        r["id"] for r in recall["rows"] if r["ok"] and abs(r["recall"] - eval_recall_by_id.get(r["id"], -1.0)) > EPS
    ]

    return {
        "label": label,
        "binary": str(binary),
        "facts_seeded": len(ids),
        "tasks": len(tasks),
        "mean_recall": recall["mean_recall"],
        "top_rank_hit_rate": recall["top_rank_hit_rate"],
        "recall_failures": recall["failures"],
        "recall_failure_sample": recall["failure_sample"],
        "eval_mean_recall": (sum(float(r.get("recall", 0.0)) for r in labeled) / len(labeled)) if labeled else 0.0,
        "eval_disagreements": disagreements,
        "mean_precision": float(summary.get("mean_precision", 0.0)),
        "mean_useful_per_1k": float(summary.get("mean_useful_per_1k", 0.0)),
        "mean_tokens": float(summary.get("mean_tokens", 0.0)),
        "brief_hit_rate": brief["hit_rate"],
        "brief_max_prefix_fraction": brief["max_prefix_fraction"],
        "brief_max_prefix_bytes": brief["max_prefix_bytes"],
        "brief_failures": brief["failures"],
        "brief_failure_sample": brief["failure_sample"],
        "brief_mean_bytes": brief["mean_bytes"],
        "recall_rows": recall["rows"],
        "brief_rows": brief["rows"],
        "eval_results": [
            {k: r.get(k) for k in ("id", "recall", "precision", "useful_per_1k", "tokens", "labeled")} for r in results
        ],
    }


# --------------------------------------------------------------------------
# scoring


def score(branch: dict[str, Any] | None, base: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    """Turn measurements into the trail monitor value and rationale.

    value = 100 x min(recall@k, top-rank hit rate, brief hit rate), measured on
    the branch binary; 0 when the brief packet puts its facts past
    BRIEF_PREFIX_LIMIT of its bytes or BRIEF_PREFIX_BYTES_LIMIT bytes, or when
    the probe could not run; capped at REGRESSION_CAP when anything regressed
    against base. The rationale carries the numbers and every flag.
    """
    if error or branch is None:
        return {
            "value": 0,
            "rationale": f"Retrieval probe could not run: {error or 'no branch measurement'}",
            "regressed": False,
        }
    if branch["tasks"] <= 0:
        return {"value": 0, "rationale": "Retrieval probe could not run: zero tasks", "regressed": False}

    recall = branch["mean_recall"]
    top = branch["top_rank_hit_rate"]
    hit = branch["brief_hit_rate"]
    prefix = branch["brief_max_prefix_fraction"]
    prefix_bytes = int(branch.get("brief_max_prefix_bytes") or 0)
    value = int(round(100 * min(recall, top, hit)))
    zero: list[str] = []
    flags: list[str] = []

    if prefix > BRIEF_PREFIX_LIMIT or prefix_bytes > BRIEF_PREFIX_BYTES_LIMIT:
        zero.append(
            f"brief packet puts {prefix_bytes} bytes ({prefix:.0%}) before its facts "
            f"(limits {BRIEF_PREFIX_BYTES_LIMIT} bytes, {BRIEF_PREFIX_LIMIT:.0%})"
        )
    for name in ("recall", "brief"):
        failures = int(branch.get(f"{name}_failures") or 0)
        if failures:
            flags.append(f"{name} failed on {failures} of {branch['tasks']} tasks ({branch.get(f'{name}_failure_sample')})")
    if branch.get("eval_disagreements"):
        flags.append(
            f"memories eval reports a different recall than the ids it returns on "
            f"{len(branch['eval_disagreements'])} tasks (first: {branch['eval_disagreements'][0]})"
        )

    parts = [
        f"recall@k {recall:.0%}, top-rank {top:.0%}, brief hit rate {hit:.0%}, "
        f"useful/1k {branch['mean_useful_per_1k']:.1f}, brief {branch['brief_mean_bytes']:.0f} B "
        f"on {branch['tasks']} frozen tasks"
    ]
    if base is not None:
        d_recall = recall - base["mean_recall"]
        d_top = top - base["top_rank_hit_rate"]
        d_hit = hit - base["brief_hit_rate"]
        d_useful = branch["mean_useful_per_1k"] - base["mean_useful_per_1k"]
        base_bytes = float(base.get("brief_mean_bytes") or 0.0)
        growth = (branch["brief_mean_bytes"] / base_bytes) if base_bytes > 0 else 1.0
        parts.append(
            f"vs base: recall {d_recall:+.0%}, top-rank {d_top:+.0%}, brief hit {d_hit:+.0%}, "
            f"useful/1k {d_useful:+.1f}, brief bytes x{growth:.2f}"
        )
        if d_recall < -EPS or d_hit < -EPS:
            flags.append("branch recovers fewer labeled facts than base")
        if d_top < -EPS:
            flags.append("labeled facts rank lower than on base")
        if d_useful < -0.05 * max(base["mean_useful_per_1k"], EPS):
            flags.append("useful-per-1k fell more than 5% against base")
        if growth > BRIEF_BYTES_GROWTH_LIMIT:
            flags.append(f"brief packet grew x{growth:.1f} against base")

    if zero:
        value = 0
    elif flags:
        value = min(value, REGRESSION_CAP)
    if zero or flags:
        parts.append("REGRESSION: " + "; ".join(zero + flags))
    return {"value": value, "rationale": ". ".join(parts) + ".", "regressed": bool(zero or flags)}


# --------------------------------------------------------------------------
# building and git on the real checkout (keeps the user's git config)


def build(source: Path, output: Path) -> None:
    proc = run(["go", "build", "-o", str(output), "./cmd/entire-brain"], cwd=source, env=os.environ.copy(), timeout=900)
    if proc.returncode != 0:
        raise ProbeError(f"go build in {source} failed: {proc.stderr.strip()[-400:]}")


def validate_branch_name(root: Path, name: str) -> str:
    name = name.strip()
    proc = run(["git", "check-ref-format", "--branch", name], cwd=root, env=os.environ.copy())
    if proc.returncode != 0 or not name or name.startswith("-"):
        raise ProbeError(f"base branch name is not a valid branch name: {name!r}")
    return name


def fetch_base(root: Path, base_branch: str) -> str | None:
    """Make origin/<base_branch> exist locally; returns an error string on failure."""
    refspec = f"+refs/heads/{base_branch}:refs/remotes/origin/{base_branch}"
    proc = run(["git", "fetch", "--quiet", "origin", refspec], cwd=root, env=os.environ.copy(), timeout=300)
    return None if proc.returncode == 0 else (proc.stderr.strip()[:200] or f"exit {proc.returncode}")


def build_base(root: Path, base_ref: str, workdir: Path) -> Path:
    worktree = workdir / "base-src"
    env = os.environ.copy()
    run(["git", "worktree", "prune"], cwd=root, env=env)  # entries left by a killed earlier run
    proc = run(["git", "worktree", "add", "-q", "--detach", str(worktree), base_ref], cwd=root, env=env)
    if proc.returncode != 0:
        raise ProbeError(f"git worktree add {base_ref} failed: {proc.stderr.strip()[:300]}")
    try:
        binary = workdir / "entire-brain-base"
        build(worktree, binary)
        return binary
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root, env=env)


def probe_changed_since(root: Path, base_ref: str) -> bool | None:
    """True when this directory differs from base_ref, None when git cannot tell.

    A branch could edit the fixture or the scorer to make itself read 100; the
    rationale then says so, which is the most an in-repo probe can do.
    """
    try:
        rel = str(HERE.relative_to(root))
    except ValueError:
        rel = str(HERE)
    env = os.environ.copy()
    # The build uses the working tree, so compare base against the working
    # tree, and count untracked files too: `git diff` alone ignores those.
    untracked = run(["git", "ls-files", "--others", "--exclude-standard", "--", rel], cwd=root, env=env)
    if untracked.returncode != 0:
        return None
    if untracked.stdout.strip():
        return True
    proc = run(["git", "diff", "--quiet", base_ref, "--", rel], cwd=root, env=env)
    if proc.returncode == 0:
        return False
    if proc.returncode == 1:
        return True
    return None


def fixture_stamp() -> str:
    """Short hashes a reviewer can match against the diff."""
    return (
        f"fixture {sha256_file(FACTS_PATH)[:8]}/{sha256_file(TASKS_PATH)[:8]} "
        f"probe {sha256_file(Path(__file__).resolve())[:8]}"
    )


# --------------------------------------------------------------------------
# commands


def cmd_freeze(args: argparse.Namespace) -> int:
    facts = load_json(FACTS_PATH)
    questions = load_json(QUESTIONS_PATH)
    with tempfile.TemporaryDirectory(prefix="brain-retrieval-freeze-") as tmp:
        store = FixtureStore(Path(tmp) / "freeze", facts["origin"], facts["branch"])
        store.create()
        ids = seed(store, Path(args.bin).resolve(), facts)
    tasks = []
    for q in questions:
        unknown = [k for k in q["relevant_keys"] if k not in ids]
        if unknown:
            raise ProbeError(f"question {q['id']} names unknown fact key(s): {unknown}")
        tasks.append(
            {
                "id": q["id"],
                "task": q["task"],
                "branch": facts["branch"],
                "k": int(q.get("k", 5)),
                "relevant": [ids[k] for k in q["relevant_keys"]],
                "label_source": "human",
            }
        )
    check_fixture_contract(facts, tasks)
    TASKS_PATH.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TASKS_PATH} ({len(tasks)} tasks, {len(ids)} facts, sha256:{sha256_file(TASKS_PATH)})")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="brain-retrieval-measure-") as tmp:
        try:
            report = measure(Path(args.bin).resolve(), "branch", Path(tmp))
            monitor = score(report, None)
        except ProbeError as exc:
            report, monitor = {"error": str(exc)}, score(None, None, str(exc))
    monitor["rationale"] += f" {fixture_stamp()}."
    emit(report, monitor, args.json)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    root = repo_root()
    report: dict[str, Any] = {
        "fixture": {"facts": sha256_file(FACTS_PATH), "tasks": sha256_file(TASKS_PATH), "probe": sha256_file(Path(__file__).resolve())},
    }
    branch_m = base_m = None
    error = None
    base_ref = args.base_ref
    base_note = None
    if base_ref is None:
        try:
            base_branch = validate_branch_name(root, args.base_branch)
            base_ref = f"origin/{base_branch}"
            if not args.no_fetch:
                fetch_error = fetch_base(root, base_branch)
                if fetch_error:
                    base_note = f"fetch of {base_branch} failed: {fetch_error}"
        except ProbeError as exc:
            base_ref = None
            base_note = str(exc)
    report["base_ref"] = base_ref

    with tempfile.TemporaryDirectory(prefix="brain-retrieval-compare-") as tmp:
        work = Path(tmp)
        try:
            if args.branch_bin:
                branch_bin = Path(args.branch_bin).resolve()
            else:
                branch_bin = work / "entire-brain-branch"
                build(root, branch_bin)
            branch_m = measure(branch_bin, "branch", work)
            report["branch"] = branch_m
        except ProbeError as exc:
            error = str(exc)
            report["branch"] = {"error": error}
        if error is None:
            try:
                if args.base_bin:
                    base_bin = Path(args.base_bin).resolve()
                elif base_ref is None:
                    raise ProbeError(base_note or "no base ref")
                else:
                    base_bin = build_base(root, base_ref, work)
                base_m = measure(base_bin, "base", work)
                report["base"] = base_m
            except ProbeError as exc:
                # A missing base is a weaker measurement, not a failed one.
                report["base"] = {"error": str(exc)}

    monitor = score(branch_m, base_m, error)
    if base_m is None and error is None:
        monitor["rationale"] += f" Base comparison unavailable: {report['base']['error'][:160]}."
    changed = probe_changed_since(root, base_ref) if base_ref else None
    report["probe_changed_vs_base"] = changed
    if changed:
        monitor["rationale"] += " NOTE: the probe or its fixture differs from base on this branch; a probe cannot vouch for its own change."
    monitor["rationale"] += f" {fixture_stamp()}."
    emit(report, monitor, args.json)
    return 0


def emit(report: dict[str, Any], monitor: dict[str, Any], compact: bool) -> None:
    """Report first, monitor object as the very last stdout line."""
    if not compact:
        print(json.dumps(report, indent=2))
    print(json.dumps({"value": monitor["value"], "rationale": monitor["rationale"]}))


class JSONLineParser(argparse.ArgumentParser):
    """Argument errors still end in the JSON line the runner reads."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise ProbeError(f"bad arguments: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = JSONLineParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze", help="seed the fixture and write fixture/tasks.json with resolved fact ids")
    p.add_argument("--bin", required=True, help="entire-brain binary used to resolve fact ids")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("measure", help="measure one binary against the frozen fixture")
    p.add_argument("--bin", required=True)
    p.add_argument("--json", action="store_true", help="print only the monitor line")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("compare", help="build the working tree and a base ref, measure both, print the monitor line last")
    p.add_argument("--base-branch", default="main", help="base branch name; fetched and used as origin/<name>")
    p.add_argument("--base-ref", help="explicit base ref (skips validation and fetch)")
    p.add_argument("--no-fetch", action="store_true", help="do not fetch the base branch first")
    p.add_argument("--branch-bin", help="use this binary for the working tree instead of building")
    p.add_argument("--base-bin", help="use this binary for base instead of building")
    p.add_argument("--json", action="store_true", help="print only the monitor line")
    p.set_defaults(func=cmd_compare)

    try:
        args = parser.parse_args(argv)
        return int(args.func(args))
    except ProbeError as exc:
        print(json.dumps({"value": 0, "rationale": f"Retrieval probe could not run: {exc}"}))
        return 2
    except Exception as exc:  # noqa: BLE001 - the runner needs a JSON line, not a traceback
        print(json.dumps({"value": 0, "rationale": f"Retrieval probe could not run: {type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    if shutil.which("git") is None:
        print(json.dumps({"value": 0, "rationale": "Retrieval probe could not run: git is not on PATH"}))
        sys.exit(2)
    sys.exit(main())
