# Retrieval regression probe

> **Status: regression signal, not evidence.** This probe tells a trail whether
> the branch still recovers a frozen set of labeled memories as well as the base
> branch does. It is deterministic, unpaid, and runs in under a minute. It makes
> no claim about Brain's value to an agent; that lane is `../agent-brain/` and
> its claim gate.

## Why it exists

Issue #335 put a 12.8 KB status block ahead of facts in `brief --json`. Agents
truncated the packet, Brain went from useful to 0/3 recovered in the memory
ladder, and nothing on the trail noticed before merge. Every retrieval surface
had an offline evaluator by then; none ran per trail. This probe is the
missing per-trail check.

## What it measures

For one `entire-brain` binary:

1. Creates a throwaway repository with a fixed origin so the brain store key is
   deterministic, and points the binary at throwaway plugin directories through
   `ENTIRE_PLUGIN_*_DIR`. The binary runs with a whitelisted environment: no
   `ENTIRE_*` or `GIT_*` variables from the developer's shell reach it, and
   `ENTIRE_BRAIN_NO_EGRESS` and `ENTIRE_BRAIN_LOCAL_ONLY` are set, so a
   configured webhook or hosted token can neither leak in nor be posted to.
2. Records every memory in `fixture/facts.json` with `remember --agent none`.
   Fact ids are content-derived, so the ids match `fixture/tasks.json` exactly;
   a mismatch is reported as a contract break, never scored as recall zero.
3. Runs `recall <task> --k <k> --json` for every task and computes recall@k and
   top-rank in the probe from the ids returned. Top-rank is recall at k equal
   to the number of labels: 1 only when every labeled fact outranks every
   distractor. The arithmetic never comes from the binary under test.
4. Runs `brief <task> --json` for every task and checks that the labeled facts
   are in the packet (brief's default limit is three facts, so this is a
   top-three check), plus how many bytes and what share of the packet precede
   the top-level `facts` section. That offset is measured on the top-level
   sections in document order, not by searching bytes, because the status
   section nests its own `facts` key and a byte search would miss the #335
   layout.
5. Runs `memories eval --retriever facts` over the same tasks for its token
   accounting (useful per 1k) and checks that its per-task recall agrees with
   the recall computed in step 3; a disagreement is flagged.

`compare` does this for a working-tree build and a base-ref build and reports
both. It validates the base branch name with `git check-ref-format`, fetches
`refs/heads/<base>` into `refs/remotes/origin/<base>` itself, and builds base in
a temporary worktree that it removes afterwards.

## The value

| Value | Meaning |
| --- | --- |
| 100 | every labeled fact is recovered at k, ranks above every distractor, and is in the brief packet |
| 51 to 99 | the branch misses some of that on its own, with no base regression flagged |
| 50 or below | something regressed against base (fewer facts recovered, lower rank, useful-per-1k down more than 5%, brief packet more than twice the size, a probe surface failing on some tasks, or eval disagreeing with the ids it returns); the value is capped at 50 so the dip is visible on the trail |
| 0 | the brief packet puts more than 8 KB or more than half of its bytes before the facts (the #335 shape), or the probe could not run |

The rationale carries the numbers, every flag after a `REGRESSION:` marker,
and short hashes of the fixture, task set and probe so a reviewer can match
them to the diff. A 0 whose rationale starts with "Retrieval probe could not
run:" is a tooling gap, not a regression; read the rationale before acting.

## What it does not measure

- History, docs, issues or query retrievers (the fixture has no sessions).
- Whether an agent uses what it is given.
- Anything at corpus scale; the fixture is 24 memories and 22 questions.
- Latency, which is noise at this size.

A branch that scores 100 here can still be a retrieval regression on a real
corpus. A branch that scores below base here has a regression a human should
look at before merge.

The probe runs from the branch under test, so a branch can change the scorer
or the fixture, or re-freeze the task set so the base binary no longer matches
it. When `benchmarks/retrieval-regression/` differs from the base ref, the
rationale carries a NOTE saying so; review that diff before trusting the
number. That is the most an in-repo probe can do about itself.

## Run it

```sh
python3 benchmarks/retrieval-regression/run.py compare --base-branch main
```

Builds the working tree (including uncommitted edits) and `origin/main` with
`go build`, measures both, prints a JSON report and then the monitor object as
the last line. Use `--branch-bin` and `--base-bin` to skip the builds,
`--no-fetch` to skip the fetch, `--json` for the monitor line only.

```sh
python3 benchmarks/retrieval-regression/run.py measure --bin ./entire-brain
```

One binary, no base comparison.

## Change the fixture

Edit `fixture/facts.json` or `fixture/questions.json`, then re-freeze the task
ids with a known-good binary and commit all three files together:

```sh
python3 benchmarks/retrieval-regression/run.py freeze --bin ./entire-brain
```

`tasks.json` is generated; do not hand-edit it. The runner compares the
seeded ids against it on every run, so a change to fact id derivation shows up
as a contract break with a clear message rather than a silent score drop.
Every question must name at least one memory; the contract check refuses an
unlabeled task because it would always count as a hit.

Keep distractors: several memories share vocabulary on purpose (three on
`brief`, three on ranking, two on scoping) so that a ranking change moves the
top-rank hit rate, not only recall at k.

## The trail runner

`.entire/runners/trail-retrieval-regression.json` runs `compare` on every push
and emits the monitor as `retrieval` (percent, higher is better). The sandbox
needs `go`, `git` and `python3`, and the base branch fetchable from `origin`.
When the base cannot be built the rationale says "Base comparison unavailable"
and the value is the branch alone; when the probe cannot run at all it reports
0 with the reason.

## Tests

```sh
python3 -B -m unittest discover -s benchmarks/retrieval-regression -p 'test_*.py'
```

`mise run test:benchmarks` runs the same. Scoring, fixture-contract and
stub-binary probe tests run everywhere. The end-to-end test runs only when
`ENTIRE_BENCHMARK_BRAIN_BINARY` points at a built binary.
