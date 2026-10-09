# SKILLS.md — CONSILIUM

Project-local playbooks for coding agents. `AGENTS.md` says **what must be true**; this file says **how to do the recurring jobs without breaking it**.

How to use this file:
- Before starting a task, find the skill whose **Use when** matches. Follow its steps in order. If more than one applies, run them in the order listed in the task (usually `architecture-guard` first).
- A skill's **Verify** section is the minimum evidence required before you call the task done. Say in your summary which checks you ran and which you could not.
- If a skill and `AGENTS.md` disagree, `AGENTS.md` wins and you tell the human.
- These skills are deliberately few and specific. Do not install generic third-party "agent skills" into this repo to cover what is here; if you think one is needed, name the gap and ask — and read its instructions before trusting it.

Index

| # | Skill | Use when | Touches |
|---|---|---|---|
| 1 | `architecture-guard` | start of every task, before every PR | every task |
| 2 | `schema-change` | fields/enums/fact IDs/reason codes change | `schemas/`, contracts |
| 3 | `routing-change` | role visibility changes | `configs/routing.yaml`, router |
| 4 | `specialist-agent` | new or changed specialist role | `agents/`, prompts |
| 5 | `prompt-change` | any prompt edit | `prompts/` |
| 6 | `note-extraction` | touching `intake/` | `intake/`, extractor prompt |
| 7 | `verifier-check` | new deterministic check | `verify/` |
| 8 | `critic-fault-injection` | evaluating critic/verifier | `eval/fault_injection.py` |
| 9 | `llm-client-and-fakes` | touching `llm/` or any LLM call site | `llm/`, tests |
| 10 | `vision-endpoint` | touching the endpoint contract or `VisionClient` | `deploy/vision_endpoint/`, `vision/` |
| 11 | `graph-node` | adding/altering graph nodes or edges | `graph/` |
| 12 | `guardrail-rule` | rail or policy-list changes | `guardrails/`, `configs/guardrails/` |
| 13 | `vision-train-eval` | training/eval of SwinV2/SegFormer, datasets, preprocessing | `training/`, `vision/` |
| 14 | `cdr-and-mask-validation` | touching `vision/cdr.py`, mask post-processing | `vision/cdr.py` |
| 15 | `calibration-and-thresholds` | calibrator or tier threshold changes | `configs/thresholds.yaml` |
| 16 | `db-migration` | `persistence/models.py` changes | `persistence/` |
| 17 | `clinician-review-and-reason-codes` | review flow, diffing, reason codes | `review/` |
| 18 | `phi-review` | new log line/trace/fixture/export; before release | everything |
| 19 | `ablation-run` | architecture evaluation, paper numbers | `eval/` |
| 20 | `bad-report-triage` | a report is flagged wrong | debugging |
| 21 | `rag-readiness-gate` | someone proposes RAG/pgvector/fine-tuning | V2 decision |
| 22 | `release-checklist` | cutting any releasable version | releases |

---

## 1. `architecture-guard`

**Use when:** at the start of every task, and again before opening a PR.

**Why it exists:** most damage to a system like this is not a bug; it is a convenient shortcut that quietly erodes isolation, auditability, or human accountability.

**Steps**
1. Identify which pipeline stage(s) the task touches (AGENTS.md §1, §4). Name the input and output contract of each (§7).
2. Run the invariant checklist against your *intended change*, not the finished code:
   - I-1 Does any LLM-facing code now receive an image, or return a number that flows into `CADResult`?
   - I-2 Does the router still use only `routing.yaml`, with no LLM or fallback?
   - I-3 Do all new claims carry resolvable `evidence_refs`?
   - I-4 Is there any new path to `APPROVED` that bypasses `review/`? Does anything mark an UNVERIFIED draft as final?
   - I-5 Does every new failure path end in `FAILED`/`NEEDS_ATTENTION`/`REJECTED_INPUT` with a reason? No silent degradation?
   - I-6/I-7 Is any new text (note, report) flowing into logs, traces, or system prompts unredacted? Any upload filename stored or logged?
   - I-9 Did a prompt, threshold, routing file, or the endpoint contract change without an eval?
   - I-10 Is anything vector/embedding/RAG creeping into the serving path? (`eval/` embeddings for the diversity metric are the only allowed exception — they must not be importable from `src/consilium`.)
   - I-13 Is any code loading PyTorch, running a forward pass, or importing `training/`/`deploy/` inside `src/consilium`? All vision access must go through `VisionClient`.
   - I-14 Is any image data or endpoint payload being sent to the LLM provider, or into logs/traces?
3. Check import boundaries (AGENTS.md §5): `schemas` ← everything; `vision` ⟂ `llm/agents`; `training`/`deploy` never imported by serving; only `persistence` touches the DB; only `llm/` touches the OpenAI SDK; only `vision/` HTTP impl talks to the endpoint.
4. Write a 3–6 line plan in the PR/task description: files touched, invariants at risk, tests that prove safety.

**Verify:** `make ci`; import-linter (or equivalent) rule set passes; plan present.

**Pitfalls:** "temporary" bypasses; adding a `dict[str, Any]` to move faster; widening a type to `Any` to silence mypy; a "quick" retry loop that turns a failed vision call into a degraded report.

---

## 2. `schema-change`

**Use when:** adding/removing/renaming a field, enum value, fact ID, or reason code in anything under `schemas/`, `context/facts.py`, or `review/` taxonomies.

**Steps**
1. State the consumers: grep for every import of the model and every string use of a changed fact ID. (Fact IDs must live only in `context/facts.py` — if you find a literal elsewhere, fix it as part of the change.)
2. Decide compatibility: additive-optional (safe), or breaking. Persisted JSON (`cad_json`, `raw_vision_json`, `draft_json`, `facts_json`, `note_json`) means breaking changes need a **data migration** or a versioned model.
3. Edit the model. Keep `Strict` (`strict=True`, `extra="forbid"`, `frozen=True`). New required LLM-facing fields need prompt updates → also run `prompt-change`.
4. Update validators if the change affects invariants (refs ⊆ allowed IDs; NUMERIC claims need `numeric_ref`; approved verdict ⇒ no issues; `endpoint_revision` must equal the pin for `RawVisionOutput`).
5. Regenerate JSON-Schema snapshots (`make schema-snapshot`) and commit the diff — the diff is the review artifact.
6. If DB-persisted: add an Alembic migration (`db-migration`). New JSON columns follow the existing `*_json` pattern.
7. Update `AGENTS.md` §7 in the same commit if the contract shape changed.

**Verify:** contract tests pass; snapshot diff reviewed; fake-LLM + fake-vision integration tests updated and green; migration up/down tested if applicable.

**Pitfalls:** enum values are API — renaming silently breaks stored rows; loosening `extra="forbid"` to "make the LLM output pass" (fix the prompt or the schema, never the strictness); changing the endpoint wire format without bumping the handler version (`vision-endpoint`).

---

## 3. `routing-change`

**Use when:** changing which facts any role may see, adding a fact that needs routing, adding a role.

**Rule zero:** the allowlist is a clinical decision. You implement the mechanism; a clinician co-author owns the policy.

**Steps**
1. Write down the intended change as a delta to the table in AGENTS.md §9.2 (which role, which fact IDs, grant or revoke) and *why*, in clinical terms.
2. Edit `configs/routing.yaml` only. Do not special-case in `router.py`.
3. Update the policy header in the YAML: reviewer name/role, date, `# CLINICAL-REVIEW:` marker if not yet reviewed.
4. Run the leak property test with the new policy and extend its generator if a new value type (e.g. a new free-text field) can carry forbidden content.
5. Re-run the specialist-scope tests; update any fixture whose allowed IDs changed.
6. Because routing changes agent inputs, run `ablation-run` (at least S3 vs previous S3) and record `S_inter` and unique-fact coverage before/after.
7. `routing_version` (hash of the YAML) is recorded on `routed_contexts` automatically — confirm it changed.

**Verify:** leak test ≥1000 hypothesis examples; scope tests; before/after diversity numbers; PR names the clinical reviewer or says "pending review".

**Pitfalls:** value leakage through free text (note quoting a CDR to the pharmacist); granting a role a field "just in case"; fixing a failing role output by widening its context instead of fixing its prompt.

---

## 4. `specialist-agent`

**Use when:** creating a new specialist role or materially changing one (Ophthalmologist, Optometrist, Pharmacist).

**Steps**
1. Define the role's *scope in one paragraph* and its *explicit exclusions*. Pharmacist example: drug-class safety only; no doses/regimens; cannot comment on disc morphology.
2. Add the `Role` enum value (`schema-change`); add its routing section (`routing-change`).
3. Create `prompts/<role>/v1.md` (Jinja). Required sections: role & scope, exclusions, output contract (matches `SubReport`), evidence-citation rule (cite fact IDs from the provided block only), uncertainty rule (use `uncertainties`), out-of-scope rule (use `declined_out_of_scope`), untrusted-data rule for note text, wording rules (AGENTS.md I-12).
4. Implement via the shared `BaseSpecialist` — role-specific code should be limited to prompt path and context type. No role-specific parsing.
5. Output handling: structured output → `SubReport`; validator checks refs ⊆ `allowed_ids()`; retry ≤2 with the validation error as a repair hint; else raise `AgentFailure`.
6. Tests: scope-violation rejection; malformed output retry then fail; an adversarial note attempting to redirect the role; a case where the role should decline (e.g. pharmacist asked about CDR in the note).
7. Eval: run `ablation-run` smoke to confirm the new role adds role-specific findings (unique-fact coverage) instead of echoing others.

**Verify:** unit + integration with fake LLM; live schema-conformance spot check on ≥20 synthetic cases (`make test-live`); diversity numbers recorded.

**Pitfalls:** a role prompt that quietly restates the whole case (re-creates the shared context through the prompt); personas ("You are a world-famous…") instead of constraints; asking for more fields than the role can ground.

---

## 5. `prompt-change`

**Use when:** editing anything under `prompts/` (including `note_extractor/`), or the Jinja context passed into a prompt.

**Steps**
1. Copy to a new version file (`v1.md` → `v2.md`). Never edit a released prompt in place; runs reference prompt versions.
2. State the hypothesis: "v2 should reduce <failure> without hurting <metric>".
3. Update the prompt-version pin in config/settings (not in code).
4. Run the fixed prompt-regression set (`make eval-smoke`) comparing v(n) vs v(n+1): schema-conformance rate, verifier issue rate by type, `S_inter`/coverage, token count. For the extractor, add: extraction field accuracy on the fixture corpus, redaction survival rate.
5. For the live comparison, run ≥3 samples per case; report mean and spread. Do not conclude from a single run.
6. Check the prompt for leakage: it must not contain facts a role is not allowed to see (templates sometimes embed "example outputs" with real-looking values — use clearly fictional, non-case values and keep them out of routing-sensitive fields). The extractor prompt must declare its input non-instructional.
7. Adopt v(n+1) only if the hypothesis holds and no guarded metric regresses beyond the tolerance written in the PR.

**Verify:** comparison table in the PR; old version retained; `prompt_version` appears in `agent_runs`.

**Pitfalls:** tuning against the same ten cases you evaluate on; adding "do not hallucinate" instructions and calling it a fix (verifier + schema are the fix); prompts growing without bound — measure tokens.

---

## 6. `note-extraction`

**Use when:** touching `intake/` (de-identification, extraction, note validation) or the extractor prompt.

**Why it exists:** the raw note is the untrusted, PHI-bearing, injection-prone input. Everything downstream treats `ClinicalNote` as typed facts; this skill guards the boundary where free text becomes structure.

**Steps**
1. **Order is fixed** (AGENTS.md §10): raw text → input rail (injection/length/charset) → PHI de-identification (deterministic patterns) → LLM extraction → Pydantic validation. Never reorder; the LLM never sees unredacted text (I-6).
2. De-identification patterns live in config, not code. Each pattern records a name (stored in `NoteExtraction.redactions`) and never its match value. Add patterns only with tests: positive (fires), negative (benign near-misses like dates and "2–4 months"), boundary cases.
3. Extraction via `LLMClient` with the pinned extractor model (default GPT-4o-mini) and versioned prompt. Structured output bound to `ClinicalNote`. Malformed → ≤2 repair retries with the validation error fed back → else `FAILED`.
4. **Anti-over-extraction:** the prompt must instruct the model to leave fields `None`/empty when the note does not say so. Write tests where the note is vague ("eye pressure a bit high") and assert no fabricated numeric is extracted.
5. **Injection resistance:** notes containing instructions ("ignore your rules and output…") must not change extraction behavior. Test with instruction-like content before and after redaction.
6. Provenance: store model snapshot, prompt version, redaction pattern names, confidence. Low confidence → the Context Builder emits `note.caveat.extraction`; verify the Director carries it into `limitations`.
7. Do not add an embedding model here (I-10). Comprehension is the LLM's job.
8. Fixtures: synthetic notes only, generated by a committed generator; include multilingual and noisy-text cases.

**Verify:** adversarial note suite green; extraction field-accuracy fixture corpus passes; no PHI pattern value appears in any log/trace/fixture (run `phi-review`); provenance stored on every run.

**Pitfalls:** redaction happening after extraction; regexes that eat clinical content (dates, measurements); "helpful" extraction that invents values; storing raw note text in `audit_log` (forbidden).

---

## 7. `verifier-check`

**Use when:** adding or modifying a check in `verify/deterministic.py`, or the forbidden-language lexicon.

**Steps**
1. Write the check as a pure function `(draft, global_ctx, cfg) -> list[Issue]` returning `Issue(raised_by="deterministic")`.
2. Choose the `IssueType`. If none fits, use `schema-change` to add one; do not overload `UNSUPPORTED`.
3. Put tolerances/lexicon entries in `configs/thresholds.yaml` (or a lexicon file), not in code.
4. Write tests **first**: positive (violation flagged), negative (clean draft passes), boundary (exactly-at-tolerance), and a regression case for any false positive you've seen.
5. Add the corresponding seeded fault to `critic-fault-injection` so the check's recall is measured.
6. Measure false-rejection rate on the clean-draft corpus. A check that rejects good drafts will train clinicians to ignore the system.

**Verify:** unit tests; fault-injection recall for the new fault class; clean-draft false-rejection delta reported.

**Pitfalls:** regexes for medical language that match benign phrases (test with real-looking benign text); numeric scan false positives on dates, "2–4 months" follow-up intervals, and tier labels; tolerance loosened to pass one case.

---

## 8. `critic-fault-injection`

**Use when:** evaluating the critic/verifier, or after any change to `verify/`, the critic prompt, the critic model/setting, or the director prompt.

**Fault classes (keep in sync with AGENTS.md §13.4):** numeric alteration (gross and subtle: 0.62 → 0.82 *and* 0.62 → 0.58); fabricated finding with plausible ref; fabricated finding with no ref; deleted mandatory finding/caveat; dose or definitive-diagnosis insertion; laterality flip; suppressed disagreement.

**Steps**
1. Assemble a corpus of **known-good** drafts: approved by the verifier *and* spot-checked by a human. Version the corpus.
2. For each draft and each fault class, apply the mutation programmatically (deterministic given a seed), record the ground-truth fault label.
3. Run three configurations: deterministic only, LLM critic only, both — in the production order (deterministic first). Same mutated drafts for all.
4. Report per class: recall (fault caught), and overall false-rejection rate on unmutated drafts. Include CIs.
5. Inspect misses by hand — the pattern in what slips past is what you fix.
6. Store as an eval run; the numbers feed the paper/README (I-11).

**Verify:** run ID with per-class table; the clean-draft false-rejection rate is present; sample of misses attached.

**Pitfalls:** mutations so crude any check catches them (include subtle ones); using the critic's own outputs to build the "known-good" corpus (circular); changing the critic's model/setting without re-running this (the critic's identity is part of the safety case).

---

## 9. `llm-client-and-fakes`

**Use when:** touching `llm/`, adding a new LLM call site, or writing a test that involves an agent. The `VisionClient` follows this same pattern — see `vision-endpoint`.

**Steps**
1. All LLM access goes through the `LLMClient` protocol: `async generate(messages, response_model, *, model, temperature, max_output_tokens, timeout_s) -> ParsedResponse[T]` plus usage metadata. The OpenAI implementation is the only code importing the SDK.
2. Use the SDK's structured-output/parse mechanism bound to the Pydantic `response_model`. Check the installed SDK version's API — it has changed over time; do not rely on memory.
3. Retry policy: transient errors (timeouts, 429, 5xx) with exponential backoff + jitter, capped; **schema-validation failures** get a bounded repair retry with the validation message appended; refusals/content-policy responses are *not* retried blindly — they surface as `AgentFailure`.
4. Enforce the per-case token budget in the client; exceeding it raises, not truncates.
5. `FakeLLMClient`: keyed canned responses by `(agent, prompt_version, case fixture id)`, with modes: valid, malformed JSON, schema-violating, timeout, refusal. Used by default in all tests.
6. A shared **contract test** runs against both fake and live clients (live marked `@pytest.mark.live`) so they cannot drift.
7. Never log message bodies at INFO; log token counts, model ID, latency, attempt.

**Verify:** contract test (fake always; live manually); a test for each failure mode proving the graph's reaction.

**Pitfalls:** model IDs defaulting to a floating alias (pin dated snapshots); swallowing a refusal and returning an empty `SubReport`; tests that silently hit the network.

---

## 10. `vision-endpoint`

**Use when:** touching `deploy/vision_endpoint/` (custom handler), `vision/` client code, `configs/models.yaml`, or the endpoint deployment.

**Why it exists:** the endpoint is a third-party boundary — images leave our infra, responses come back over HTTP, and the endpoint can be cold, slow, or silently updated. This skill keeps that boundary honest.

**Steps**
1. **Contract (AGENTS.md §8.0):** the endpoint returns `RawVisionOutput` — raw classifier probability/logits and RLE-encoded masks with declared dimensions + `endpoint_revision`. Nothing else. If a change wants the endpoint to return CDR, tiers, or quality scores: refuse — interpretation lives in-repo (I-13).
2. **Handler:** preprocessing (resize, normalization) happens inside the endpoint handler, consuming the preprocessing spec exported by `training/`. Maintain the preprocessing-parity test: a fixture image produces the identical tensor through the training path and the handler path.
3. **Revision pinning:** `configs/models.yaml` pins the endpoint revision. The client checks every response's `endpoint_revision` against the pin; mismatch → `VISION_UNAVAILABLE` (fail closed) + alert. Deployments bump the pin deliberately in the same commit that updates the packaging artifact hash.
4. **Client robustness:** `VisionClient` mirrors `LLMClient` — bounded retries with exponential backoff + jitter on 503/timeouts/connection errors (cold starts), hard timeout `VISION_TIMEOUT_S` (default 120s), no fallback to "no vision" output (I-5).
5. **Fake for tests/dev:** `FakeVisionClient` returns canned `RawVisionOutput` keyed by fixture, with modes: normal, cold-start (slow first call), 503-then-ok, revision-mismatch, malformed payload, empty mask. The local docker-compose stack includes a small fake vision HTTP service so e2e tests exercise real HTTP without the real endpoint.
6. **Recorded fixtures:** contract tests for the HTTP impl use recorded responses (respx), never live network in CI.
7. **Privacy/terms (I-14):** endpoint must be private (not public), TLS only, region compatible with the data-residency decision, HF data-usage terms reviewed and recorded. On any change to the endpoint or provider, re-check these.
8. **Cost/ops:** record the always-on vs scale-to-zero choice in `configs/models.yaml` comments; cold-start latency and retry counts are emitted as metrics (AGENTS.md §18).

**Verify:** contract tests (fake + recorded); revision-pin mismatch test; cold-start retry test; preprocessing parity test; handler smoke test against the dev endpoint (live-tagged).

**Pitfalls:** letting the endpoint compute anything interpretive (it becomes an unversioned clinical component); swallowing a revision mismatch; retry loops that exceed the case timeout; preprocessing drift between `training/` and the handler.

---

## 11. `graph-node`

**Use when:** adding or altering a LangGraph node, edge, or state field.

**Steps**
1. Node signature: `async def node(state: CaseState, deps: Deps) -> CaseStatePatch`. No globals; all effects via `deps` (LLM, vision, repos, clock).
2. **LangGraph is thin wiring.** The node's real logic lives in an ordinary module (`vision/`, `intake/`, `agents/`, `verify/`) that does not import LangGraph. If you find yourself importing LangGraph outside `graph/`, stop.
3. Persist on exit: status transition + audit event + state patch, in one DB transaction. Do not rely on LangGraph checkpointing for durability.
4. Define every failure edge up front: which exception types map to which terminal status (`FAILED`, `NEEDS_ATTENTION`, `REJECTED_INPUT`). No node may raise an unmapped error.
5. Conditional edges are plain functions of state, unit-tested without running the graph.
6. Resume semantics: on restart, load persisted status and re-enter at the next node. Nodes must be idempotent — vision results are never recomputed if `cad_results` exists; extraction never re-runs if `clinical_notes` exists.
7. The audit loop's counters (`round`, `revision`) live in state and are persisted. `CRITIC_MAX_ROUNDS` is read from config, not hard-coded.
8. Parallel specialist fan-out: gather with per-task error capture; one failure fails the case after retries (I-5), but successful sibling outputs are still persisted for debugging.
9. New nodes added to V1 must include: `NOTE_EXTRACTED` (intake) and `VISION_CALLED` (endpoint call) — follow their existing patterns for rail-before-LLM and retry-on-cold-start.

**Verify:** integration tests for happy path, each failure edge, loop exhaustion (assert the UNVERIFIED draft contract on the API), crash-resume, cold-start path; state diagram in AGENTS.md §4 updated if edges changed.

**Pitfalls:** hidden state in closures; a node that both calls the LLM and decides routing; swallowed exceptions inside fan-out; depending on LangGraph's in-memory checkpoint store for durability.

---

## 12. `guardrail-rule`

**Use when:** adding/changing an input rail, output rail, or policy list.

**Steps**
1. Decide the layer: if it's about *shape*, it's Pydantic; *values/coverage*, the verifier; *entailment*, the critic; *content policy/injection/PHI*, a rail. Don't implement the same check in two layers without reason.
2. **Placement (fixed):** the input rail runs on the **raw note text before de-identification and extraction** (AGENTS.md §10, §14); the output rail runs on the DirectorDraft before clinician view. No rail on the vision path — quality gates handle that.
3. Write the adversarial examples first in `tests/adversarial/` — at least 5 positive (should trigger) and 5 benign near-misses (should not), including obfuscations (unicode homoglyphs, spacing, mixed case, multilingual).
4. Implement in `configs/guardrails/` (and a deterministic Python pre-check for anything that can be a regex/lexicon — cheaper and testable offline). NeMo handles what needs a model (chiefly injection detection).
5. On trigger: record an audit event (`rail`, `reason`, `case_id`), set the correct status (`REJECTED_INPUT` pre-extraction, `NEEDS_ATTENTION` post), return an actionable message. Never silently pass modified content on.
6. Measure latency added; rails sit in the critical path. Record p50/p95.
7. Note in the config what the rule does **not** catch — future-you needs the gap list.

**Verify:** adversarial suite green; latency recorded; benign near-misses don't trigger.

**Pitfalls:** treating NeMo as a safety proof; rails that mutate clinical text invisibly; placing a rail in front of the vision models (pointless — use quality gates); rail after extraction when the PHI lives in the raw text.

---

## 13. `vision-train-eval`

**Use when:** training, fine-tuning, or re-evaluating SwinV2 or SegFormer; adding a dataset; changing preprocessing; packaging a new endpoint artifact.

**Steps**
1. **Data:** register each dataset in `training/data/DATASETS.md` — source, license/terms (verify at download; do not assume), label provenance, size, device, population notes. Track via DVC; raw data never in git. **Locked V1 plan (AGENTS.md §8.5):** REFUGE = primary dev/eval; Harvard-FairVision = final ablation/external eval only (CC BY-NC-ND 4.0 — non-commercial research, never for clinical decisions; record the license); ORIGA = unavailable, do not reference it in code or configs.
2. **Splits:** patient-level (both eyes together). Generate once with a fixed seed, commit the split manifest (IDs only), reuse everywhere. Confirm no duplicates across train/val/test and across the "external" set.
3. **Preprocessing:** one shared spec used by training and by the endpoint handler (`vision-endpoint` step 2 maintains parity). Normalisation stats and resize policy live in `configs/models.yaml`. Training/serving skew is the most common silent vision bug.
4. **Training:** config-driven, seeded, logged to MLflow; log git SHA, data version, hyperparameters, curves. Start from public pretrained weights; justify any change of backbone size by measured gain.
5. **Evaluation** (val for model selection, test touched once at the end, external reported separately):
   - Classifier: AUROC, sensitivity@specificity and specificity@sensitivity operating points, Brier, ECE, reliability plot, confusion at chosen tier thresholds.
   - Segmenter: Dice/IoU for disc and cup, vCDR MAE vs. expert, failure-case gallery.
   - Subgroups where metadata allows.
6. **Packaging (`training/endpoint_packaging/`):** export weights + preprocessing spec; build the custom handler; compute sha256; register run ID; deploy to the (dev, then prod) endpoint; bump the revision pin in `configs/models.yaml` in the same commit. Serving refuses to start on hash/pin mismatch.
7. **Robustness checks** before accepting a model: JPEG-quality drops, brightness/contrast shifts, blur, a different camera type if available; report degradation.

**Verify:** MLflow run with all metrics; split manifest committed; preprocessing parity test; artifact hash + revision pin verified in a serving smoke test.

**Pitfalls:** image-level splits leaking patients; tuning thresholds on the test set; comparing in-distribution test numbers with external-style claims; ignoring class prevalence when reading predictive values.

---

## 14. `cdr-and-mask-validation`

**Use when:** touching `vision/cdr.py`, mask post-processing, or the mask convention. Masks arrive from the endpoint as `RawVisionOutput` — decode first, then everything below applies.

**Steps**
1. Decode RLE masks; assert dimensions match the declared header; resolution changes across endpoint revisions must fail the revision-pin check, not silently re-scale.
2. Mask post-processing: keep the largest connected component per structure; fill holes; enforce `cup ⊆ disc` (clip cup to disc and record that the clip occurred as a QC signal).
3. `vertical` CDR = cup vertical extent / disc vertical extent (bounding-box height or max vertical chord — pick one, document it, test it). Guard divisions: disc diameter ≥ 1 else the case fails QC.
4. `area_based` = `sqrt(cup_area / disc_area)` under the nested convention. Never use the paper's `cup/(cup+disc)` form with nested masks (see AGENTS.md §8.2).
5. Populate `MaskQC` honestly; `ok=False` becomes a caveat fact the Director must carry through.
6. Tests with synthetic masks:
   - concentric circles at known ratios (0.3, 0.5, 0.7, 0.9) — assert exact within pixel quantisation;
   - cup = 0; cup == disc; elliptical and tilted discs; two disconnected blobs; cup outside disc;
   - a regression test documenting why the paper's formula differs.
7. Compare against expert vCDR on a labelled set (REFUGE provides annotations); report MAE and Bland–Altman bias.

**Verify:** all synthetic tests; MAE number stored; QC flag behaviour covered.

**Pitfalls:** pixel-area ratios on images of different resolution (ratios are fine, absolute areas aren't comparable — don't put raw pixel areas where a clinician might read them as physical sizes); silently clipping masks without recording it; trusting mask dimensions without checking.

---

## 15. `calibration-and-thresholds`

**Use when:** fitting a calibrator, changing tier thresholds, or the endpoint model changes.

**Steps**
1. The calibrator operates **in-repo on raw endpoint logits** (`p_raw` → `p_calibrated`) — never inside the endpoint (I-13).
2. Fit temperature scaling (or isotonic, if you can justify the extra flexibility and have enough data) on the validation set only.
3. Evaluate calibration on test: ECE (state the binning), Brier, reliability diagram. If calibration is poor on the external set, say so in `limitations` of the model card.
4. Choose tier thresholds on validation to hit a stated operating point (e.g. sensitivity ≥ X for HIGH+INTERMEDIATE combined). Write the operating-point rationale in `thresholds.yaml` comments. This is `# CLINICAL-REVIEW:` material.
5. Freeze: thresholds, calibrator params, and model artifact/pin are versioned **together**. A change to any one is a new model version.
6. Re-run `ablation-run` smoke: downstream behaviour (tier distribution, critic issue rate) will shift when thresholds shift.

**Verify:** metrics in MLflow; thresholds and calibrator referenced by the same artifact ID; downstream smoke run done.

**Pitfalls:** presenting an uncalibrated score as a probability; threshold tuned on the test set; forgetting prevalence shift between dev data and deployment population.

---

## 16. `db-migration`

**Use when:** any change to `persistence/models.py`.

**Steps**
1. Change the SQLAlchemy model; autogenerate an Alembic revision; **read and edit it** — autogenerate misses constraints, defaults, and data moves.
2. Write the downgrade. If it is genuinely destructive, say so in the revision docstring and refuse in production via an explicit guard.
3. Append-only tables (`audit_log`, `verdicts`, `reviews`): preserve the privilege revocations (no UPDATE/DELETE for the app role) in the migration.
4. New tables follow the §16 patterns — JSON columns named `*_json`; encrypted columns (`raw_redacted_text_enc`) use the same column-encryption helper; indexes for the new access pattern; `EXPLAIN` on realistic row counts for anything hit per request.
5. Backfill in a separate, resumable step for large tables; never in a single locking migration.
6. Test: upgrade from empty, upgrade from the previous revision with fixture data, downgrade one step.

**Verify:** migration tests; `make migrate` on a fresh DB; append-only enforcement test still passes.

**Pitfalls:** storing JSON blobs you later need to query without an index; enum changes in Postgres (can't remove values easily — plan the rename); putting PHI into columns that are logged.

---

## 17. `clinician-review-and-reason-codes`

**Use when:** changing the review flow, edit diffing, or the reason-code taxonomy.

**Steps**
1. Review is a service with three actions (`APPROVE`, `APPROVE_WITH_EDITS`, `REJECT`). The only code path that writes `APPROVED*` statuses lives here. The React UI calls this service and never sets status itself.
2. Edits are stored as a structured, field-level diff against the exact AI draft revision they were made on (`reviews.draft_revision`). Store both sides; a diff alone is not recoverable if the schema evolves.
3. Reason codes are a closed list (AGENTS.md §15). Edits/rejections without ≥1 code are refused at the API; `OTHER` requires text.
4. Reviewer identity is mandatory and taken from the auth context, never from the request body. A submitter may not approve their own case.
5. Approved reports are immutable; amendment creates a linked new version.
6. **UNVERIFIED drafts (critic loop exhausted):** the API serves them flagged (`drafts.unverified = true`); the UI must render the "UNVERIFIED — critic did not approve" banner with issues highlighted per claim. The review service refuses `APPROVE`/`APPROVE_WITH_EDITS` on an unverified draft unless the edit set resolves every open issue (checked against `verdicts.issues_json`, not client input).
7. Keep the `FeedbackStore.record` seam (V1 → Postgres writer; no retrieval). Do **not** add embedding or retrieval code.
8. Add an analytics query (view or script) over reason codes — it feeds `rag-readiness-gate`.

**Verify:** tests for each action, forbidden self-approval, unverified-approval block, immutability, diff round-trip; reason-code analytics query runs on fixture data.

**Pitfalls:** letting the UI decide status; losing the original draft when the clinician edits; making reason codes optional "to reduce friction" (that destroys the V2 dataset); allowing approval of a draft with unresolved critic issues.

---

## 18. `phi-review`

**Use when:** any change that adds a log line, trace attribute, fixture, error message, export, new field carrying text; and before every release.

**Steps**
1. Trace the new data's path: input → rail → extraction → prompt → LLM → DB → log/trace → API response → (vision endpoint). Mark every place it persists or leaves the process.
2. Confirm de-identification runs **before** extraction and that `ClinicalNote` has no field intended for direct identifiers.
3. Grep logs/trace attributes for note text, report text, and **upload filenames** (filenames often contain names/MRNs — never log or store them; store the content hash only).
4. Error messages and problem+json bodies must not echo user content.
5. Fixtures: synthetic only. If you create a new fixture, state how it was generated.
6. Confirm retention/erasure job coverage for any new table or object-store prefix.
7. Re-check the LLM data path: only derived facts and de-identified note text go to the provider; images never do (I-14).
8. Re-check the vision endpoint path: private endpoint, TLS, data terms on file, region compatible with the data-residency decision.

**Verify:** checklist completed in the PR; log-scrubbing tests pass; secret/PHI scanner clean.

**Pitfalls:** exception messages that include the offending input; debug flags left on; analytics exports that join reviewer text with case metadata.

---

## 19. `ablation-run`

**Use when:** evaluating architecture changes, producing paper/README numbers, or validating routing/prompt/threshold changes.

**Steps**
1. Freeze inputs: case set (with versioned manifest), cached `CADResult`s — **recorded endpoint responses held constant**, never live vision calls — model snapshots, prompt versions, embedding model for `S_inter` (eval-only, AGENTS.md §19.2).
2. Run S1–S4 (AGENTS.md §19.1) on identical cases. Only the architecture varies. For stochastic LLM paths run ≥3 samples per case.
3. Compute: `S_inter` (state whether similarity or distance, and the embedding model), unique-fact coverage, unsupported-claim rate, omission rate, minority-opinion preservation (curated set), rounds-to-approval, latency, tokens/cost.
4. Aggregate with bootstrap 95% CIs; show per-case paired differences, not just means.
5. Add the clinician-rubric subset when available; if LLM-as-judge is used, validate it against clinician scores on the same subset and state the judge model.
6. Persist as a single run ID with config hashes; generate tables/figures **from the stored run** (no hand-typed numbers).
7. Write a short interpretation that includes negative or null results and threats to validity (same-base-model convergence, embedding dependence, small N, synthetic-case bias).

**Verify:** run ID; CIs present; every number in docs traceable to a run.

**Pitfalls:** comparing stages on different case sets; reading a single sample as an effect; reporting only the metric that moved; letting vision outputs vary between stages (defeats the isolation comparison).

---

## 20. `bad-report-triage`

**Use when:** a clinician or test flags a report as wrong, unsafe, or odd.

**Procedure — locate the stage before touching a prompt.** Work backwards:
1. **Vision transport:** check `vision_calls` — cold-start retries, revision-pin mismatch, timeouts? A transport problem impersonates a model problem.
2. **CAD:** open `cad_results`. Are the numbers right for this image (overlay check)? Quality flags/QC? If wrong → vision issue (`vision-train-eval`, `cdr-and-mask-validation`). Do not "fix" it downstream.
3. **Extraction:** open `clinical_notes.extraction_json`. Did the note facts come out right (IOP, meds, intolerances)? Over-extracted invented values? PHI redaction fire?
4. **Context:** inspect `routed_contexts`. Did the right facts reach each role? Was a caveat fact missing?
5. **Specialist:** inspect each `SubReport` and its `agent_runs` (prompt version, attempts). Was the error present in a sub-report, or introduced by the Director?
6. **Director:** compare claims to sub-reports. Dropped disagreement? Invented claim? Numeric ref wrong?
7. **Verification:** which issues were raised and which were missed? If the draft is wrong and the verifier approved it, you have a verifier gap → add a check (`verifier-check`) and a fault class (`critic-fault-injection`) before anything else.
8. **Guardrails/review:** did a rail pass something it should have caught? What did the clinician change (reason code)?
9. Classify the root cause: vision-transport | vision-model | extraction | routing | prompt | director synthesis | verifier gap | knowledge gap | guardrail gap | UI/review.
10. Fix at the *earliest responsible* stage, add a regression test using a de-identified/synthetic reproduction, and record the reason-code statistics.

**Verify:** root-cause label recorded; regression test fails before the fix and passes after.

**Pitfalls:** rewriting the director prompt to mask a vision error; treating every quality problem as a "needs RAG" problem (check `rag-readiness-gate` first); ignoring the extraction layer when the note facts look wrong in the report.

---

## 21. `rag-readiness-gate`

**Use when:** someone proposes RAG, pgvector memory, or fine-tuning. V1 forbids them (I-10); this skill is how the decision is made, not how it is implemented.

**Steps**
1. Pull the review analytics from `clinician-review-and-reason-codes`: share of edits by reason code over the evaluation window, with N.
2. Pull verifier/critic issue statistics: knowledge-type vs evidence-type rejections.
3. Check for recurring identical corrections (repeated mistakes) — the `reviews` table is the raw material here.
4. Check clinician rubric scores for accuracy/safety against the agreed floor.
5. Compare against the thresholds agreed with clinical co-authors **before** looking at the data (AGENTS.md §23). Write down the decision and the numbers.
6. If the evidence points at evidence/numeric errors, the answer is verifier/prompt work, not retrieval. If it points at knowledge: propose, in order, curated guideline retrieval → correction memory → fine-tuning, each as a new ablation stage against S4.
7. Produce a design note only after the gate is passed; do not write retrieval code in V1 branches.

**Verify:** decision record with data, thresholds, and sign-off.

**Pitfalls:** deciding from anecdotes; adding RAG because the paper's architecture figure includes pgvector.

---

## 22. `release-checklist`

**Use when:** cutting any version intended for use outside dev.

1. `make ci` green on a clean checkout; `make test-live` run recently and results attached.
2. Model artifacts: artifact hashes match `configs/models.yaml`; endpoint revision pin matches the deployed (dev) endpoint; model card updated (data, metrics incl. external, calibration, limitations, intended use, subgroup results).
3. Prompts: versions pinned; last prompt-regression comparison attached; no unreviewed `# CLINICAL-REVIEW:` markers remain in routing/thresholds (or each is listed as an accepted risk with an owner).
4. Evaluation: latest ablation run ID linked; every number in README/paper reproducible from it; fault-injection recall report linked.
5. DB: migrations tested up/down on a copy of production-shaped data; backup and restore rehearsed.
6. Security: dependency audit, secret scan, auth role tests, `phi-review` completed (incl. upload filenames and the endpoint privacy/terms check); threat model doc current.
7. Observability: OTel + self-hosted Langfuse dashboards and alerts deployed; runbook covers vision endpoint failure/cold-start spike, LLM outage/timeouts, critic-loop exhaustion spike, review backlog.
8. Product language: UI and report templates carry the screening-support wording, the UNVERIFIED banner behavior, and clinician attribution; no copy implies autonomous diagnosis.
9. Provider terms: OpenAI and Hugging Face data-retention/ZDR terms re-checked and recorded in README.
10. Regulatory/ethics posture documented: intended use statement, whether clinical use triggers medical-device rules in the target jurisdiction, and who owns that determination. (This is for the humans to resolve; the agent flags it and does not decide it.)
11. Rollback plan: previous images/artifacts retained; how to freeze new case intake safely.

**Verify:** checklist in the release PR with links; unresolved items explicitly accepted by a named person.

---

## Appendix — conventions used across skills

- `# CLINICAL-REVIEW:` — comment marker for any value or rule that encodes a clinical judgement. Grep for it before release; each one needs a named reviewer or an accepted-risk entry.
- **Run ID** — MLflow run (training/eval) or eval-harness ID; anything quoted in docs must carry one.
- **Synthetic fixture** — generated, not derived from a real patient; the generator is committed next to the fixture.
- **Smoke vs full** — smoke runs use fake/recorded LLMs, recorded vision responses, and a tiny fixed set (CI-safe); full runs use live LLMs, the dev endpoint, and the full frozen set.
- **UNVERIFIED draft** — a DirectorDraft whose critic loop exhausted `CRITIC_MAX_ROUNDS`; served flagged, never final without clinician edit-and-approve that resolves every open issue.
