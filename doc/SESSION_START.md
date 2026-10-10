# CONSILIUM — Session Start Prompt (paste at the beginning of every new session)

You are the coding agent for the CONSILIUM project (clinical glaucoma screening
pipeline). Work under the project contracts. TOKEN DISCIPLINE IS MANDATORY in
this session: we operate under a 5-hour rolling usage window, so minimize
context size and avoid re-reading anything already summarized here.

## 1. Resume position — read FIRST, in this exact order, nothing else yet

1. `Kimi.md` — read ONLY:
(a) the file header (STANDING CONVENTION + "Current position" line), and
(b) the **most recent** section entry (the last `### Section X.Y` block).
Do NOT read older phase logs — the header + last entry are the checkpoint.
CURRENT POSITION (as of last checkpoint): Phase 2 IN PROGRESS — Sections
2.1 + 2.2 done. NEXT TASK: Section 2.3 — in-repo interpretation:
calibration, tiers, quality, CDR (awaiting go-ahead).
2. `SKILLS.md` — read ONLY the index table + the ONE skill matching the
current task. For Section 2.3 that is skill 15 `calibration-and-thresholds`
AND skill 14 `cdr-and-mask-validation`. Read no other skills unless the
task later requires them.
3. `AGENTS.md` — read ONLY: §2 (Invariants table, all 14) and the section(s)
matching the current task (for 2.3: §8 Vision subsystem + §21 Code
standards). Skim §5 hard boundaries. Do NOT read §9–§20, §22–§26 now.
4. `ARCHITECTURE.md` — read ONLY §5.4 (Vision subsystem design) and §16.1/16.2
(CDR + calibration algorithms) for this task. Read other sections only
when a later task requires them.

If anything in these files appears to conflict, AGENTS.md wins — stop and
tell me instead of silently resolving it.

## 2. Task for this session

Section 2.3 of the implementation plan as recorded in `Kimi.md` / §22 of
AGENTS.md: build the in-repo interpretation layer — `src/consilium/vision/`
modules: `rle.py` (RLE decode), `postprocess.py` (largest component, fill
holes, clip cup⊆disc with recorded clip), `cdr.py` (vertical primary +
area_based secondary, nested-mask convention, all guards), `quality.py`
(client-side prechecks + post-inference quality + mask QC), `calibration.py`
(temperature scaling on raw logits), `pipeline.py` (`run(image) -> CADResult`),
plus `configs/thresholds.yaml` and the `configs/models.yaml` calibrator
params seam. Wire the FakeVisionClient fixtures keyed by sha256. Follow
skills 14 + 15 exactly: tests first (synthetic masks at known ratios 0.3/
0.5/0.7/0.9, cup=0, cup==disc, tilted/elliptical, two blobs, cup-outside-
disc), tolerances in config not code, `# CLINICAL-REVIEW:` markers on
threshold choices. Keep `training/segmentation/metrics.py` mirrored to
`vision/cdr.py` — reconcile per the 2.2 note (vision/cdr.py becomes
canonical). Do not start until I say go.

## 3. Token-saving working rules (apply for the WHOLE session)

1. NEVER read the full repository, never `ls -R`/glob-scan for exploration.
Read only files you are about to create or modify, and only the sections
you need (use targeted reads, not whole-file dumps of large files).
2. Do NOT re-read the four project docs in full (read full docs only if you lost more then 70% if the memory of the project). If you need a detail, read
only the named section. This prompt + `Kimi.md`'s last entry are your
context; treat them as authoritative for cross-session state.
3. Run SCOPED tests only: `uv run pytest tests/unit/test_<module>.py -q`
during development. Run the full offline battery (`pytest -m "not live"`,
ruff, mypy, lint-imports, schema-check) ONLY once, at section end, before
writing the `Kimi.md` entry.
4. Note: `make` is NOT installed on this machine — run the `uv run ...`
equivalents directly (see Kimi.md Section 0.3).
5. Keep your chat responses short: status lines, file lists, test counts.
No prose summaries of what I already know — spend output budget on code
and test results only.
6. When context usage feels heavy (long file reads accumulate), tell me and
I will run `/compact` — before that, ensure `Kimi.md` is up to date so
nothing is lost (files on disk + the log are the memory, not the chat).
7. New session per section/phase is our convention: at section end, update
`Kimi.md` with the full section entry per its log template (goal, files
created/changed, key decisions, verified commands + counts, deferred,
CLINICAL-REVIEW items, MANUAL STEPS list), then stop. I will start a fresh
session for the next section and paste this prompt again.
8. No network, no real LLM calls, no real endpoint calls — offline rule
(AGENTS.md §20). Synthetic fixtures only.

## 4. Standing conventions you must follow

- Every section report (chat + `Kimi.md`) ends with **MANUAL STEPS FOR THE
HUMAN** (empty list if none).
- Invariants I-1…I-14 are non-negotiable; if a task seems to require breaking
one, stop and ask.PHI 
- Fail closed, append-only audit thinking, no PHI anywhere (filenames
included), strict Pydantic schemas, mypy --strict, ruff clean.
- Report honestly: if you could not verify something, say so in the entry.

Acknowledge briefly (a few lines: resume position + what you will read),
then wait for my go-ahead on Section 2.3.

## 5. SHELL COMMAND EXECUTION — RTK (MANDATORY)

### 5.1 Mandatory RTK Usage

- Before executing any shell command, use `rtk` as the command prefix wherever RTK supports the command.
- Examples:
  - `rtk git status`
  - `rtk git diff`
  - `rtk uv run pytest`
  - `rtk uv run ruff check src/`
  - `rtk uv run mypy --strict src/`
  - `rtk docker compose ps`
  - `rtk ls src/consilium/`
- For command chains, prefix each command individually where required. Do not assume that prefixing the first command automatically applies RTK to subsequent commands.
- Use Windows-compatible commands and paths. Do not assume Unix utilities such as `ls`, `cat`, `grep`, or `tail` are installed directly in Windows CMD.

### 5.2 RTK Output Handling

- Treat RTK's condensed output as the default result.
- Use `rtk proxy <command>` only when the condensed output is missing expected information, is garbled, or contradicts the command's exit status.
- Never disable RTK merely to obtain verbose output unless there is a concrete debugging reason.
- Preserve the original command's exit status and investigate failures instead of assuming a command succeeded.
- RTK is an output-filtering tool, not a replacement for the underlying command.

### 5.3 Startup Verification

At the beginning of each new Kimi Code session:

1. Read this entire `SESSION_START.md` file before making changes.
2. Check that RTK is available using `rtk --version`.
3. Verify the current repository using `rtk git status`.
4. If RTK reports that hooks are not installed, inspect the situation and recommend `rtk init -g` if appropriate. Do not silently assume automatic hooks are active.
5. If a command fails because RTK cannot execute it, diagnose the issue. Use the underlying command without RTK only when necessary, and report why.

### 5.4 Project Safety and Development Rules

- Follow `AGENTS.md` as the primary repository instruction contract and respect its precedence rules.
- Do not modify application code, configuration, schemas, tests, or documentation merely to verify RTK.
- Never claim RTK is active or functioning correctly without checking the actual command output.
- Do not run destructive Git commands or commit changes unless explicitly requested.
- Continue normal development after verification; RTK must not interfere with the project's testing, linting, type checking, or CI requirements.

### 5.5 Session Continuity

These instructions apply to every fresh Kimi Code session. At session startup, confirm that you have read `SESSION_START.md` and are following its RTK requirements. If any instruction conflicts with `AGENTS.md`, follow the repository's documented precedence rules and explain the conflict before proceeding.