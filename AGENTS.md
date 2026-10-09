# AGENTS.md — CONSILIUM

Read this entire file before you write or change anything. It is the contract between the human maintainers and any coding agent working in this repo. This file has been aligned with the locked V1 design decisions (see §26 for the deployment and compliance baseline).

Precedence when documents disagree: **AGENTS.md > SKILLS.md > README.md > code comments**. If the code contradicts this file, do not silently "fix" either one — stop and tell the human which one you think is wrong and why.

---

## 1. What CONSILIUM is

CONSILIUM is a clinical-reporting system for **glaucoma screening support** from retinal fundus images plus an optional clinician-supplied medical note. It is a directed graph, not a chatbot:

```
Fundus image (+ optional free-text medical note)
        │
  Image validation / quality pre-checks        (client-side: decode, size, resolution)
        │
  ┌─────┴───────────────────────────┐
  │ Vision service (remote endpoint) │  SwinV2 raw logits + SegFormer raw masks
  └─────┬───────────────────────────┘      (served via Hugging Face Inference Endpoint;
        │                                   dev uses a local fake — see §8.0)
        │   raw logits + masks + endpoint revision
  Backend post-processing (THIS repo): calibration → tier, mask post-processing →
        measurements → vertical CDR (+ area-based), mask QC, quality gate
        │   CADResult  (deterministic, numeric, versioned — computed here, never by the endpoint's hidden logic)
        │
  Note intake: de-identification → GPT-4o-mini structured extraction → Pydantic validation
        │   ClinicalNote (typed facts; free text preserved but marked untrusted)
        │
  Context Builder  →  GlobalContext (C_global): every fact has a stable ID
        │
  Context Router   →  role-specific sub-contexts (C_k), allowlist projection (deterministic)
        │
  ┌─────┼──────────────┐
  Ophthalmologist  Optometrist  Pharmacist     (LLM, isolated, parallel, GPT-4o-mini default)
  └─────┼──────────────┘
        │   SubReports (structured, evidence-referenced)
  Director Agent  →  DirectorDraft (structured claims + explicit disagreements)
        │
  Verification:  deterministic verifier (FIRST)  +  Adversarial Critic (LLM, different model/setting)
        │   ↺ reject → Director revises (bounded loop, CRITIC_MAX_ROUNDS=3)
  Guardrails (NeMo: input rail on the raw note, output rail on the draft)
        │
  Human clinician review  →  approve / edit-and-approve / reject   (HITL, mandatory)
        │
  Approved report (immutable, clinician-attributed)  +  structured correction record (for V2)
```

The research claim (team paper): when several LLM agents read the same context they converge ("consensus collapse") and hallucinations slip through. CONSILIUM's answer is **structural**: isolate context per role, audit the synthesis against ground-truth CAD data, keep a human accountable. Every design decision below serves one of those three things. If a change weakens isolation, auditability, or human accountability, it is wrong regardless of how convenient it is.

---

## 2. Invariants (non-negotiable)

Violating any of these is a bug even if all tests pass. If a task seems to require breaking one, stop and ask.

| # | Invariant |
|---|-----------|
| I-1 | **LLMs never see the image and never create or alter CAD numbers.** Numbers originate in `vision/` (from endpoint outputs) and flow through `CADResult`. An LLM may only *reference* them by fact ID. |
| I-2 | **The Context Router is deterministic code** driven by `configs/routing.yaml`. No LLM call, no heuristics, no "smart" fallback. Allowlist only: a field not explicitly granted to a role is not delivered. |
| I-3 | **Every factual claim carries `evidence_refs`** that resolve to fact IDs the author was allowed to see. Unresolvable refs fail validation. |
| I-4 | **Nothing is final without a clinician action.** Pipeline output is always `DRAFT`/`PENDING_REVIEW`. Only the review service can set `APPROVED`. A report the critic never approved is shown only as an `UNVERIFIED` draft and can never become final without clinician fix-and-approve. |
| I-5 | **Fail closed.** If vision fails, a specialist fails after retries, or the critic loop cannot converge, the case surfaces as `FAILED` or `NEEDS_ATTENTION` with the reason. No silently degraded report, ever. |
| I-6 | **Minimum necessary PHI.** Opaque case IDs, no names/DOB/MRN in prompts, logs, traces or fixtures. De-identification runs on the raw note **before** extraction and before the Context Builder. Upload filenames (which often contain names/MRNs) are never logged or stored. |
| I-7 | **Clinical notes are untrusted input** (prompt-injection vector). They are delimited, length-capped, passed through the input rail, and never concatenated into system prompts. Structured fields extracted from them are data, not instructions. |
| I-8 | **Every LLM boundary is schema-validated both ways** (Pydantic v2, strict). Free-text-in-free-text-out between components is forbidden. This includes the note-extraction step. |
| I-9 | **Prompts are versioned artifacts** (`prompts/<component>/vN.md`). Changing a prompt requires an eval run (SKILLS: `prompt-change`). Prompt version is recorded on every agent run. |
| I-10 | **V1 has no RAG, no embeddings in the serving path, no pgvector, no fine-tuning of LLMs.** Do not add them "because it would be better". The only permitted embeddings anywhere in the repo are inside `eval/` (for the diversity metric) and are never imported by serving code. Leave the seams described in §14 and §23, nothing more. |
| I-11 | **No performance claim without a measured number** produced by `eval/` and stored with its run ID. This applies to code comments, docs, README and the paper. |
| I-12 | **Language is screening support, not diagnosis.** Reports never state a definitive diagnosis, never state glaucoma *stage/severity* unless a clinician-provided note supplies it, never give drug doses or regimens. |
| I-13 | **Vision inference is a replaceable dependency behind the `VisionClient` interface.** The API/worker never loads PyTorch or runs a forward pass in-process (CPU or GPU). Calibration, screening tiers, CDR and mask QC are computed **in this repo** from raw endpoint outputs; thresholds and formulas are versioned here. The endpoint returns raw logits and masks only — nothing else. |
| I-14 | **The raw fundus image never goes to the LLM provider** and never appears in logs, traces, or prompts. It only ever transits to the configured vision endpoint (or the local fake). The endpoint must be private, its data terms reviewed, and its region checked against data-residency decisions (§26). |

---

## 3. Scope

**In scope for V1**
- Single-image (one eye) case processing; laterality carried through the whole pipeline (OD / OS / UNKNOWN).
- SwinV2 classification + SegFormer disc/cup segmentation, fine-tuned (training in-repo), **served remotely** via Hugging Face Inference Endpoints, consumed through `VisionClient`.
- Image-quality gate + mask QC, calibration, screening tiers, CDR (backend-computed).
- Note intake: de-identification + LLM structured extraction + validation.
- Context Builder, Router, three specialists, Director, deterministic verifier + LLM Critic, guardrails.
- Clinician review with structured correction capture.
- PostgreSQL for cases, runs, drafts, verdicts, reviews, audit log. **Plain relational storage; no vector columns.**
- A thin **React (Vite) clinician review UI**: case list, draft view with **mask overlay viewer**, edit-and-diff form, approve/reject with reason codes. (The UI consumes the API; it contains no pipeline logic.)
- Evaluation harness: four-stage ablation, diversity metric, critic fault-injection.
- FastAPI service, Docker, CI, observability (OpenTelemetry + self-hosted Langfuse).

**Out of scope for V1 (do not build)**
- RAG / guideline retrieval, pgvector memory, fine-tuning of LLMs.
- In-process GPU inference in the API container (the fake `VisionClient` covers local dev and tests).
- Bilateral reasoning across two eyes in one report (V2; carry `laterality` so it is possible).
- Visual field / OCT image ingestion (their *values* may arrive via the clinical note).
- Autonomous model retraining from feedback.
- A patient-facing portal. Approved reports may support clinician–patient communication, but every surface in V1 is clinician-facing and every report carries clinician attribution and screening-support wording.
- Multi-tenant auth beyond a single clinic-level role model (see §16).

---

## 4. Pipeline state machine

One `CaseState` object flows through a LangGraph `StateGraph`. LangGraph is thin wiring only: nodes are plain async functions `(state, deps) -> state patch`; all real logic lives in ordinary modules that never import LangGraph; **state is persisted to Postgres after every node** (do not depend on LangGraph checkpointing).

```
RECEIVED
  → IMAGE_PRECHECKED       (decode, format, size, resolution floor)     fail → REJECTED_INPUT
  → INPUT_RAIL_PASSED      (raw note: injection, PHI patterns, limits)    fail → NEEDS_ATTENTION
  → NOTE_EXTRACTED         (LLM structured extraction, ≤2 repair retries) fail → FAILED
  → VISION_CALLED          (VisionClient → endpoint; cold-start aware)    fail → FAILED
  → CAD_COMPLETE           (backend: calibrate → tier, masks → CDR, QC)   fail → FAILED
  → CONTEXT_BUILT          (C_global with fact IDs)
  → ROUTED                 (C_k per role)
  → SPECIALISTS_DONE       (3 parallel, each ≤ 2 retries)                 fail → FAILED
  → DRAFTED                (director)
  → AUDITING ⇄ REVISING    (deterministic verifier FIRST, then LLM critic;
                            max CRITIC_MAX_ROUNDS=3)                     no converge → NEEDS_ATTENTION
  → OUTPUT_RAIL_PASSED
  → PENDING_REVIEW
  → APPROVED | APPROVED_WITH_EDITS | REJECTED   (clinician only)
```

Rules:
- Persist state after **every** transition (`cases.status`, `audit_log`). A crash must be resumable from the last persisted state. Vision results are never recomputed if `cad_results` exists (idempotency guard on every node).
- The audit loop is bounded by `CRITIC_MAX_ROUNDS` (default 3, env-configurable). On exhaustion the case goes to `NEEDS_ATTENTION` with **unresolved issues attached** to the draft. The UI shows that draft only with a prominent **"UNVERIFIED — critic did not approve"** banner and the flagged claims highlighted; it can never become final except by clinician fix-and-approve (I-4). It is never auto-approved and never silently dropped.
- **Real role disagreement ≠ failed audit.** A `Disagreement` entry in the draft is preserved visibly; it does not by itself trigger `NEEDS_ATTENTION`. Only unresolved critic issues (unsupported claim, numeric mismatch, omission, unsafe wording, suppressed disagreement) do.
- Verification order is fixed: the **deterministic verifier runs first** (numeric checks, references, coverage, forbidden language). Only a draft that passes it reaches the LLM Critic. Numeric mismatches and unsafe wording are deterministic — they are fixed by round 2 or the Director is genuinely failing and a human should see it.
- Quality-gate failures return an actionable reason (e.g. `OPTIC_DISC_NOT_VISIBLE`, `UNDEREXPOSED`, `NOT_A_FUNDUS_IMAGE`) instead of a low-confidence report.
- Idempotency: `POST /cases` accepts an `Idempotency-Key`; re-submission returns the existing case.
- Cold-start handling: the vision endpoint may scale to zero. The `VisionClient` treats 503/timeouts as retryable with backoff; total wait is bounded by `VISION_TIMEOUT_S`. If the budget is exceeded the case fails closed (`FAILED`, reason `VISION_UNAVAILABLE`) — never a degraded report.

---

## 5. Repository layout

```
consilium/
├── AGENTS.md  SKILLS.md  README.md
├── pyproject.toml  uv.lock  Makefile  docker-compose.yml  .env.example
├── configs/
│   ├── routing.yaml            # role → allowed fact IDs (single source of truth for I-2)
│   ├── thresholds.yaml         # screening tiers, quality gate, tolerances (versioned with model)
│   ├── models.yaml             # model/endpoint IDs, artifact hashes, preprocessing stats, revision pins
│   └── guardrails/             # NeMo Guardrails config + colang
├── prompts/
│   ├── note_extractor/v1.md
│   ├── ophthalmologist/v1.md
│   ├── optometrist/v1.md
│   ├── pharmacist/v1.md
│   ├── director/v1.md
│   └── critic/v1.md
├── deploy/
│   └── vision_endpoint/        # HF Inference Endpoint custom handler (preprocessing + model here),
│                               # revision-pinned; NOT imported by src/ (deployed artifact only)
├── src/consilium/
│   ├── api/                    # FastAPI routers, deps, error mapping
│   ├── core/                   # settings, logging, errors, ids, clock
│   ├── schemas/                # ALL cross-component Pydantic models (see §7)
│   ├── vision/                 # VisionClient protocol + HTTP impl + fake impl, postprocess,
│   │                           # quality (client-side pre-checks + mask QC), cdr, calibration, pipeline
│   ├── intake/                 # note de-identification, note_extractor.py (via LLMClient), validation
│   ├── context/                # builder.py, router.py, facts.py
│   ├── llm/                    # LLMClient protocol, OpenAI impl, fake impl, retry/budget
│   ├── agents/                 # base.py, specialists, director.py, prompt_loader.py
│   ├── verify/                 # deterministic.py, critic.py, verdict merge
│   ├── guardrails/             # NeMo wiring + deterministic output checks
│   ├── graph/                  # state.py, nodes.py, build.py  (thin wiring only)
│   ├── persistence/            # models.py (SQLAlchemy), repositories.py, alembic/
│   ├── review/                 # clinician review service, reason codes, diff
│   ├── eval/                   # datasets, runners, metrics, ablation, fault injection, reports
│   └── observability/          # OpenTelemetry + Langfuse wiring
├── frontend/                   # React (Vite) clinician review UI; no pipeline logic; talks to API only
├── training/                   # NOT imported by serving code. Own deps group.
│   ├── classification/  segmentation/  calibration/  data/  endpoint_packaging/
├── tests/
│   ├── unit/  contract/  integration/  adversarial/  e2e/  live/
├── scripts/
└── data/                       # gitignored; DVC-tracked pointers only
```

Hard boundaries:
- `schemas/` imports nothing from the rest of the package. Everything else may import `schemas/`.
- `vision/` must not import `llm/`, `agents/`, or `graph/`. `agents/` must not import `vision/` (agents receive facts, never CAD machinery).
- `training/` and `deploy/` are never imported by `src/consilium`. Serving consumes **remote inference** through `VisionClient`; training produces the weights that get deployed to the endpoint.
- Only `persistence/` touches the database. Only `llm/` touches the OpenAI SDK. Only `vision/` (HTTP impl) talks to the vision endpoint.
- `intake/` uses `llm/` for extraction; it never imports `agents/` or `context/`.

---

## 6. Stack (decided — do not substitute without asking)

| Concern | Choice | Notes |
|---|---|---|
| Language / env | Python ≥3.11, `uv` | Lockfile committed. |
| Validation | Pydantic v2, `strict=True`, `extra="forbid"` on all LLM-facing models | |
| Orchestration | LangGraph | Thin wiring only (§4). Nodes are plain async functions; state persisted to Postgres after every node. |
| LLM | OpenAI via official SDK, structured outputs bound to Pydantic models | **Default model GPT-4o-mini for all five agents.** Model IDs from config/env, pinned to dated snapshots, never hard-coded. Critic runs on a **different model or at minimum a different temperature/setting** than the Director — self-critique by the same configuration is weak evidence. |
| Vision serving | Hugging Face Inference Endpoints (private) | SwinV2 + SegFormer behind one custom handler. Returns **raw logits + masks only**. ~$0.50/hr always-on (~$360/mo); scale-to-zero is cheaper but adds cold-start latency the pipeline must tolerate (§4). Images leave our infra — endpoint must be private, data terms reviewed, region checked (I-14, §26). |
| Vision consumption | `VisionClient` protocol (HTTP impl + `FakeVisionClient`) | Mirrors the `LLMClient` pattern: timeouts, bounded retries with backoff + jitter, revision pin check per response. Backend owns calibration, tiers, CDR, QC. |
| Vision training | PyTorch + HF `transformers`, in `training/` | Fine-tune from public pretrained weights. REFUGE primary dev/eval set; see §8.5. |
| Guardrails | NeMo Guardrails | **Input + output rails only** (§13). |
| API | FastAPI + Uvicorn | |
| Frontend | React + Vite + TypeScript | Thin clinician review screen: mask overlay viewer, edit-and-diff form. (Chosen over Streamlit: the edit-diff and overlay interactions need a real UI.) |
| DB | PostgreSQL 16, SQLAlchemy 2.x, Alembic | No pgvector extension in V1. |
| Object storage | Local FS in dev, S3-compatible in prod | Images stored by content hash, never in Postgres. |
| Experiment / data tracking | MLflow + DVC | Every model artifact has a run ID and sha256. |
| LLM observability | **Langfuse, self-hosted** | Prompt versions, token cost, per-agent traces, critic/reviewer annotations. Self-host because real note text will reach it; Cloud acceptable for dev with synthetic data only. Self-host adds ClickHouse + Redis + blob storage — check current docs before committing. **No prompt/response bodies in traces by default** (§18). |
| Infra observability | OpenTelemetry traces, structured JSON logs, Prometheus metrics | PHI-scrubbed. |
| Tests | pytest, hypothesis, pytest-asyncio, respx | |
| Quality | ruff (lint+format), mypy `--strict` on `src/` | CI blocking. |
| Packaging | Docker (CPU image; GPU only in training/endpoint contexts), docker-compose for local | Local stack: api, postgres, object store, **fake vision service**, (optional Langfuse). |

---

## 7. Data contracts

All cross-component types live in `src/consilium/schemas/`. Below is the authoritative shape; implement faithfully, extend only via the `schema-change` skill.

```python
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

class Laterality(StrEnum):
    OD = "OD"; OS = "OS"; UNKNOWN = "UNKNOWN"

class Role(StrEnum):
    OPHTHALMOLOGIST = "ophthalmologist"
    OPTOMETRIST = "optometrist"
    PHARMACIST = "pharmacist"

class AgentName(StrEnum):          # identity for agent_runs / prompt versioning
    NOTE_EXTRACTOR = "note_extractor"
    OPHTHALMOLOGIST = "ophthalmologist"; OPTOMETRIST = "optometrist"; PHARMACIST = "pharmacist"
    DIRECTOR = "director"; CRITIC = "critic"

class ScreeningTier(StrEnum):          # derived from CALIBRATED probability via thresholds.yaml
    LOW = "low"; INTERMEDIATE = "intermediate"; HIGH = "high"

# ---------- vision ----------
class QualityFlag(StrEnum):
    UNDEREXPOSED="underexposed"; OVEREXPOSED="overexposed"; BLURRED="blurred"
    DISC_NOT_VISIBLE="disc_not_visible"; NOT_FUNDUS="not_fundus"; LOW_RESOLUTION="low_resolution"

class ImageQuality(Strict):
    score: float = Field(ge=0, le=1)
    gradable: bool
    flags: tuple[QualityFlag, ...] = ()

class ClassifierOutput(Strict):
    p_raw: float = Field(ge=0, le=1)          # raw logit→prob from endpoint
    p_calibrated: float = Field(ge=0, le=1)   # temperature-scaled HERE, in-repo
    tier: ScreeningTier
    model_version: str            # endpoint model revision + MLflow run ID

class MaskQC(Strict):
    cup_within_disc: bool
    single_component_each: bool
    disc_area_plausible: bool     # within thresholds.yaml range
    ok: bool                      # AND of the above

class Measurements(Strict):
    disc_area_px: int = Field(gt=0)
    cup_area_px: int = Field(ge=0)
    disc_vertical_diameter_px: int = Field(gt=0)
    cup_vertical_diameter_px: int = Field(ge=0)

class CDRMetrics(Strict):
    vertical: float = Field(ge=0, le=1)       # PRIMARY: cup vertical diameter / disc vertical diameter
    area_based: float = Field(ge=0, le=1)     # secondary: sqrt(cup_area / disc_area), disc mask INCLUDES cup
    mask_convention: str = "disc_includes_cup"

class CADResult(Strict):
    quality: ImageQuality
    classifier: ClassifierOutput
    measurements: Measurements
    cdr: CDRMetrics
    mask_qc: MaskQC
    segmenter_version: str        # endpoint model revision + MLflow run ID
    laterality: Laterality

# endpoint wire format (what VisionClient returns; NOT persisted as-is)
class RawVisionOutput(Strict):
    p_raw: float = Field(ge=0, le=1)
    disc_mask_rle: str            # run-length-encoded, resolution declared in header
    cup_mask_rle: str
    mask_width: int = Field(gt=0); mask_height: int = Field(gt=0)
    endpoint_revision: str        # must equal the pinned revision in models.yaml

# ---------- clinical note ----------
class ClinicalNote(Strict):       # post-de-identification, post-extraction
    age_years: int | None = Field(default=None, ge=0, le=120)
    iop_mmhg_od: float | None = None
    iop_mmhg_os: float | None = None
    current_medications: tuple[str, ...] = ()
    drug_intolerances_or_allergies: tuple[str, ...] = ()
    comorbidities: tuple[str, ...] = ()
    prior_glaucoma_dx: str | None = None
    exam_findings: str | None = Field(default=None, max_length=2000)  # free text, untrusted
    free_text: str | None = Field(default=None, max_length=4000)      # free text, untrusted

class NoteExtraction(Strict):     # provenance for the extraction step
    note: ClinicalNote
    extractor_model: str          # pinned snapshot, e.g. gpt-4o-mini-YYYY-MM-DD
    extractor_prompt_version: str
    redactions: tuple[str, ...] = ()   # which PHI patterns fired (name of pattern, never the value)
    extraction_confidence: Confidence  # low confidence surfaces a caveat fact

# ---------- facts & contexts ----------
class FactSource(StrEnum):
    CAD = "cad"; NOTE = "note"; DERIVED = "derived"

class Fact(Strict):
    id: str                        # stable, dotted: "cad.cdr.vertical", "note.meds", ...
    label: str
    value: str | float | int | bool | tuple[str, ...]
    unit: str | None = None
    source: FactSource

class GlobalContext(Strict):       # C_global
    case_id: str
    laterality: Laterality
    facts: tuple[Fact, ...]
    # builder guarantees: unique IDs, every CAD field represented, caveat facts for QC flags
    # and for low-confidence note extraction

class RoleContext(Strict):         # C_k
    role: Role
    case_id: str
    laterality: Laterality
    facts: tuple[Fact, ...]
    routing_version: str           # hash of routing.yaml used
    def allowed_ids(self) -> frozenset[str]: ...

# ---------- agent outputs ----------
class Confidence(StrEnum):
    LOW="low"; MODERATE="moderate"; HIGH="high"

class Finding(Strict):
    text: str = Field(min_length=1, max_length=600)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    confidence: Confidence

class Recommendation(Strict):
    text: str = Field(min_length=1, max_length=500)
    triggered_by: tuple[str, ...] = Field(min_length=1)   # fact IDs that motivate it
    basis: str = "model_knowledge"                        # V1: always; flagged for clinician

class SubReport(Strict):
    role: Role
    findings: tuple[Finding, ...]
    recommendations: tuple[Recommendation, ...]
    uncertainties: tuple[str, ...] = ()
    declined_out_of_scope: tuple[str, ...] = ()           # things it noticed but is not allowed to judge

class ClaimKind(StrEnum):
    NUMERIC="numeric"; CATEGORICAL="categorical"; INTERPRETIVE="interpretive"; RECOMMENDATION="recommendation"

class Claim(Strict):
    id: str                                               # "c1", "c2", ...
    kind: ClaimKind
    text: str
    evidence_refs: tuple[str, ...]                        # resolve against GlobalContext
    source_roles: tuple[Role, ...]                        # who asserted it
    numeric_value: float | None = None                    # REQUIRED when kind == NUMERIC
    numeric_ref: str | None = None                        # fact ID the number must equal (REQUIRED for NUMERIC)

class Disagreement(Strict):
    topic: str
    positions: tuple[tuple[Role, str], ...] = Field(min_length=2)
    resolution: str                                       # how the draft handles it, or "left to clinician"

class DirectorDraft(Strict):
    impression: str                                       # screening-support wording only
    claims: tuple[Claim, ...]
    disagreements: tuple[Disagreement, ...] = ()
    recommendations: tuple[Recommendation, ...] = ()
    limitations: tuple[str, ...]                          # must include QC caveats from context
    revision: int = 0

# ---------- audit ----------
class IssueType(StrEnum):
    UNSUPPORTED="unsupported"; NUMERIC_MISMATCH="numeric_mismatch"; OMITTED_FINDING="omitted_finding"
    OUT_OF_SCOPE="out_of_scope"; UNSAFE_LANGUAGE="unsafe_language"; DISAGREEMENT_SUPPRESSED="disagreement_suppressed"
    BAD_REFERENCE="bad_reference"

class Issue(Strict):
    type: IssueType
    claim_id: str | None
    detail: str
    evidence: tuple[str, ...] = ()
    raised_by: str                                         # "deterministic" | "llm_critic"

class CriticVerdict(Strict):
    approved: bool
    issues: tuple[Issue, ...] = ()
    round: int

    @model_validator(mode="after")
    def _consistent(self):
        if self.approved and self.issues:
            raise ValueError("approved verdict cannot carry issues")
        if not self.approved and not self.issues:
            raise ValueError("rejection must state at least one issue")
        return self
```

Contract rules:
- Fact IDs are an API. They are defined in one module (`context/facts.py`) as constants. Never inline a fact-ID string anywhere else.
- A specialist's `evidence_refs` and `triggered_by` must be ⊆ `RoleContext.allowed_ids()`. This is a **second, independent enforcement of isolation** (the router restricts what the agent sees; this validator rejects an agent that cites something it should not know).
- `Claim.kind == NUMERIC` requires `numeric_value` and `numeric_ref`; the verifier compares against the fact within tolerance from `thresholds.yaml`.
- `DirectorDraft.limitations` must contain a caveat for every `QualityFlag`, for `mask_qc.ok == False`, and for low-confidence note extraction. The verifier checks this; it is not left to the LLM.
- Extraction provenance (`NoteExtraction`) is stored with the case; the raw note text lives only in the case record (I-6).

---

## 8. Vision subsystem

### 8.0 Serving architecture — read first
V1 serves vision models **remotely** on a Hugging Face Inference Endpoint; nothing GPU-bound runs in the API container.

- **Endpoint contract.** One custom handler (`deploy/vision_endpoint/`) hosts both models and returns **raw logits + raw masks only** (encoded as in `RawVisionOutput`). It performs preprocessing (resize, normalization) inside the handler so train/serve skew is controlled in one place (`training/` exports the preprocessing spec; the handler consumes it). All *interpretation* — calibration, tier mapping, CDR, mask QC, quality scoring — happens **in this repo** (`src/consilium/vision/`) so thresholds and formulas stay versioned and testable.
- **Revision pinning.** `configs/models.yaml` pins the endpoint's model revision. Every response's `endpoint_revision` is checked against the pin; mismatch → `VISION_UNAVAILABLE` (fail closed) + alert. Training packaging bumps the pin deliberately, never silently.
- **Cold starts.** The endpoint may scale to zero. The client treats 503/connection timeouts as retryable with exponential backoff + jitter, bounded by `VISION_TIMEOUT_S` (default 120s). The pipeline never degrades to a report without vision output (I-5).
- **Client pattern.** `VisionClient` mirrors `LLMClient`: protocol + HTTP impl + `FakeVisionClient` (deterministic canned outputs keyed by fixture, including a slow/cold-start mode). All tests and local dev use the fake; the HTTP impl is exercised by contract tests (recorded responses, no network in CI).
- **Privacy.** The image leaves our infrastructure at this boundary. Requirements (I-14, §26): private endpoint (not public), HF data-usage terms reviewed and documented in `training/data/DATASETS.md` equivalents, region compatible with the data-residency decision, TLS only. If a future deployment must keep images on-prem (hospital requirement), swap the `VisionClient` binding to a self-hosted handler — the interface is deliberately deployment-agnostic.
- **Cost note.** Always-on ≈ $0.50/hr ≈ $360/month for both models. Scale-to-zero cuts this but adds cold-start latency; make the choice in config, not code.

### 8.1 Models and training
- **Classifier:** SwinV2 (start with a small variant; justify any larger one with a measured gain). Binary glaucoma vs non-glaucoma → probability.
- **Segmenter:** SegFormer, 3-class (background / disc rim-or-disc / cup) or 2 nested binary masks. **Pick one mask convention and encode it in `CDRMetrics.mask_convention`.** Default: nested (disc mask includes cup).
- Training lives under `training/`; the endpoint packages trained weights via `training/endpoint_packaging/`; the deployed artifact hash is recorded in `configs/models.yaml` and verified at request time via the revision pin.

### 8.2 CDR — read this, the paper formula needs care
The paper (and MedChat) write `CDR = sqrt(|M_cup| / (|M_cup| + |M_disc|))`. That is only a cup-to-*disc* area ratio if `M_disc` denotes the **rim-only** (disc minus cup) region. With the common nested convention (disc mask includes the cup) the denominator double-counts the cup and the value is biased low. Also, the clinically used metric is the **vertical** cup-to-disc ratio (a height ratio), not an area ratio.

Therefore:
- `cdr.vertical` = cup vertical diameter / disc vertical diameter, from the masks' vertical extents. **This is the number agents and reports use.**
- `cdr.area_based` = `sqrt(cup_area / disc_area)` with `disc_area` from the nested disc mask. Secondary, for comparison with the paper/MedChat.
- Unit-test both with synthetic concentric masks of known ratios, including degenerate cases (cup = 0, cup ≈ disc, tilted/elliptical disc).
- Say this plainly in the README's "Differences from the paper" section.

### 8.3 Calibration and tiers
- Raw softmax scores are not probabilities. Fit temperature scaling (or isotonic if justified) on a held-out **validation** set; evaluate on a separate **test** set. The calibrator is **backend code + versioned parameters** operating on endpoint logits — it never lives inside the endpoint.
- `ScreeningTier` thresholds in `configs/thresholds.yaml` are chosen on validation data to meet a stated operating point (e.g. target sensitivity), then frozen. Changing thresholds is a model-version change.
- Report: AUROC, sensitivity @ fixed specificity and specificity @ fixed sensitivity, Brier score, ECE, reliability diagram. Segmentation: Dice, IoU for disc and cup, vCDR MAE vs. expert annotation. Always **per dataset**, and **external-dataset** results separately from in-distribution results.

### 8.4 Quality gates
Two layers:
1. **Client-side pre-checks** (before calling the endpoint): decode check, format allowlist, resolution floor, file size cap. Cheap, keeps garbage off the endpoint.
2. **Post-inference quality + QC** (backend, on returned outputs/masks): exposure/blur measures, disc-visibility heuristic, fundus-vs-not heuristic, mask QC (`MaskQC`). A non-gradable image never reaches the LLMs. Gate thresholds in `thresholds.yaml`. Failures return an actionable reason (`NOT_A_FUNDUS_IMAGE`, `UNDEREXPOSED`, …), never a low-confidence report.

### 8.5 Data hygiene and dataset plan (the part that silently ruins glaucoma papers)
- **Split by patient, never by image.** Both eyes of a patient go to the same split.
- Check for cross-dataset duplicates before declaring an "external" set external.
- **Locked V1 dataset plan** (verify terms before use; do not assume):
  | Dataset | Role | Notes |
  |---|---|---|
  | REFUGE (MICCAI 2018, 1,200 fundus images) | Primary vision dev/eval | Disc/cup masks + clinical labels; free for research via grand-challenge.org. Confirm the current terms at registration. |
  | Harvard-FairVision | Final ablation / external eval only | CC BY-NC-ND 4.0: **non-commercial research only, never for clinical decisions**; SLO images, ~600 GB. Record the license in `DATASETS.md`. |
  | ORIGA | **Not used** | Reported no longer publicly available. Do not build around it. |
- Record dataset license and usage terms in `training/data/DATASETS.md`; some fundus datasets restrict redistribution or require data-use agreements.
- Document label provenance (single grader vs consensus; clinical diagnosis vs CDR-derived label).
- Report subgroup performance where metadata permits (device, ethnicity, age band). Failure on a subgroup is a reportable limitation.

---

## 9. Context Builder and Router

### 9.1 Builder
`context/builder.py` turns `CADResult` + `ClinicalNote` into a `GlobalContext` of `Fact`s with stable IDs. It also emits **caveat facts** (`cad.caveat.<flag>`) for any quality flag or QC failure, and `note.caveat.extraction` when note-extraction confidence is low, so caveats are first-class evidence the Director must carry into `limitations`.

Canonical fact IDs (non-exhaustive; the module is authoritative):

```
cad.p_glaucoma.calibrated   cad.p_glaucoma.tier
cad.cdr.vertical            cad.cdr.area_based
cad.disc.area_px            cad.cup.area_px
cad.disc.vdiam_px           cad.cup.vdiam_px
cad.quality.score           cad.quality.flags
cad.maskqc.ok               cad.laterality
cad.caveat.<flag>
note.age                    note.iop.od / note.iop.os
note.meds                   note.intolerances
note.comorbidities          note.prior_dx
note.exam_findings          note.free_text
note.caveat.extraction
```

### 9.2 Router
`context/router.py` is a pure function: `route(global_ctx, routing_cfg) -> dict[Role, RoleContext]`.

`configs/routing.yaml` is the **single source of truth**. Starting policy (the paper's principle — each role gets role-specific facts and is deliberately denied others, e.g. the pharmacist never sees pixel-level segmentation):

| Fact group | Ophthalmologist | Optometrist | Pharmacist |
|---|:-:|:-:|:-:|
| `cad.p_glaucoma.tier` | ✔ | ✔ | ✔ |
| `cad.p_glaucoma.calibrated` | ✔ | ✔ | ✗ |
| `cad.cdr.vertical` | ✔ | ✔ | ✗ |
| `cad.cdr.area_based`, disc/cup areas & diameters | ✔ | ✗ | ✗ |
| `cad.quality.*`, `cad.caveat.*`, `cad.maskqc.ok` | ✔ | ✔ | ✗ |
| `cad.laterality` | ✔ | ✔ | ✔ |
| `note.iop.*`, `note.exam_findings`, `note.prior_dx` | ✔ | ✔ | ✔ (IOP, prior_dx only) |
| `note.meds`, `note.intolerances`, `note.comorbidities` | ✗ | ✗ | ✔ |
| `note.age` | ✔ | ✔ | ✔ |
| `note.free_text` | ✔ | ✔ | ✗ |

Treat this table as a **proposal to be reviewed by a clinician co-author** before freezing — role boundaries are a clinical judgement, not an engineering one. Record the reviewer and date in `routing.yaml`.

Enforcement:
1. Router emits only allowlisted facts.
2. A **prompt-leak test** renders each role's final prompt string and asserts that no forbidden fact ID, label, or value appears (property-based over randomly generated `GlobalContext`s; see §20).
3. Specialist output validator rejects refs outside `allowed_ids()`.
4. `routing_version` (hash of the YAML) is stored with every run.

Pitfall: values can leak through *derived* text — e.g. the free-text note quoting "CDR 0.8" to the pharmacist. `note.free_text` and `note.exam_findings` are therefore withheld from roles that must not see CAD-equivalent content, and the leak test includes numeric-value matching, not just ID matching.

---

## 10. Note intake (de-identification + extraction)

The raw medical note arrives as free text from the submitter. Before it becomes `ClinicalNote` facts:

1. **Input rail on raw text** (deterministic first, NeMo for model-needed checks): prompt-injection / instruction-like content detection, PHI pattern block/redact (names, phone, MRN-like IDs), length and charset caps. The raw note never touches an LLM before this rail passes.
2. **De-identification**: pattern-based redaction runs **before extraction** (I-6). Record which patterns fired (`NoteExtraction.redactions` — pattern names only, never values).
3. **Structured extraction** via `LLMClient` (default GPT-4o-mini, prompt `prompts/note_extractor/vN.md`, versioned like every other prompt): raw (redacted) text → `ClinicalNote`. Pydantic-validated; malformed output gets ≤2 repair retries with the validation error fed back; then `FAILED` (I-5).
4. **Untrusted-data discipline**: extracted fields are data, not instructions. The extractor prompt declares the input non-instructional; anything instruction-like that survives the rail is ignored by contract. `exam_findings`/`free_text` remain marked untrusted downstream (I-7).
5. **Provenance**: model snapshot, prompt version, redaction pattern names, and extraction confidence are stored with the case. Low confidence ⇒ the builder emits `note.caveat.extraction`, which the Director must carry into `limitations`.

Do **not** add an embedding model for note understanding — comprehension is the LLM's job; embeddings are out of scope in V1 (I-10).

---

## 11. Specialist agents

Three isolated agents: **Ophthalmologist, Optometrist, Pharmacist.** Same base class, different prompt, different `RoleContext`. They run concurrently and never see each other's output. Default model: GPT-4o-mini (config).

Each specialist:
- Receives: system prompt (`prompts/<role>/vN.md`, rendered from a Jinja template with the facts as a structured block) + its facts. Untrusted note text is placed inside a clearly delimited data block that the prompt declares as non-instructional.
- Returns: `SubReport` via structured output. Anything else → validation error → retry (≤2) with the validation error fed back as a repair hint → else fail the case (I-5).
- Must populate `declined_out_of_scope` instead of speculating beyond its role (e.g. the pharmacist notes "cannot comment on disc morphology").
- Must express uncertainty in `uncertainties` rather than hedging in prose.

Role intent (the prompt files are authoritative, these are the design constraints):
- **Ophthalmologist:** structural interpretation of CDR and probability tier, plausibility versus note findings, escalation considerations. Never states a stage unless the note does.
- **Optometrist:** examination and follow-up pathway — what additional tests/measurements would clarify (VF, OCT, gonioscopy, repeat imaging), referral timing language, image-quality implications.
- **Pharmacist:** medication-safety lens only — intolerances/contraindication considerations relevant to **drug classes** commonly used in glaucoma, interaction flags against listed medications/comorbidities. **No doses, no regimens, no brand-specific instructions.** Everything is phrased as "consider … subject to prescriber judgement".

Generation settings: low temperature, bounded `max_output_tokens`, request timeout, per-case token budget (`LLM_CASE_TOKEN_BUDGET`). Settings are configuration, not constants in code.

Diversity is a measured property, not an assumption: see `S_inter` in §19. If routed specialists still converge, fix the routing/prompts — do not paper over it by injecting fake disagreement.

---

## 12. Director

Input: three `SubReport`s **plus the read-only `GlobalContext` fact sheet** (so it quotes numbers by reference instead of inventing them). Output: `DirectorDraft`.

Design decisions (deliberately different from MedChat's director):
- **Attribute, don't launder.** MedChat's director is told never to reference sub-reports. CONSILIUM's thesis is that preserved dissent is a safety feature. Every `Claim` lists `source_roles`; conflicts go in `disagreements` with positions per role. The director must not smooth a real disagreement into a confident sentence.
- Numbers are claims of kind `NUMERIC` with `numeric_ref` — free-floating numerals in prose that are not backed by a claim are flagged by the verifier.
- `limitations` is mandatory and verified (QC caveats, extraction caveat, "screening support only", absence of VF/OCT if absent).
- On revision rounds the director receives the previous draft + the `CriticVerdict.issues` and must address each issue ID or explicitly justify non-change. It does not receive the critic's reasoning beyond the issue list.
- Wording contract (I-12): impression uses "findings are consistent with / raise suspicion of / warrant evaluation for", never "patient has".

---

## 13. Verification: deterministic verifier + Adversarial Critic

The paper's critic is an LLM. In production we do not trust an LLM to be the *only* thing checking an LLM. Verification is two layers in a fixed order, and a draft passes only if **both** pass.

### 13.1 Deterministic verifier (`verify/deterministic.py`) — runs FIRST, cheap, exact
- Schema + reference resolution (every `evidence_refs` entry exists in `GlobalContext`) → `BAD_REFERENCE`.
- Numeric check: each `NUMERIC` claim's `numeric_value` equals the referenced fact within tolerance → `NUMERIC_MISMATCH`.
- Stray-number scan: numerals in `impression`/claim text/recommendations not attributable to a numeric claim → `UNSUPPORTED`.
- Mandatory coverage: laterality, screening tier, vertical CDR, and every caveat fact appear in claims/limitations → `OMITTED_FINDING`.
- Forbidden-language lexicon (definitive diagnosis phrasing, dose/regimen patterns such as `\d+\s?(mg|mcg|%)`, "stage"/"severity" without note support) → `UNSAFE_LANGUAGE`.
- Disagreement preservation: if two sub-reports hold polar positions on a topic tagged in a small configured topic list, the draft must contain a `Disagreement` for it → `DISAGREEMENT_SUPPRESSED`. (Conservative: this check may flag for review; it never auto-resolves.)
- Role-scope: a claim sourced from a role may not rely on refs outside that role's `allowed_ids()` → `OUT_OF_SCOPE`.

### 13.2 LLM Critic (`verify/critic.py`) — semantic check
- Input: the draft's claims + the `GlobalContext` facts. Not the specialists' reasoning (keeps it independent of the story the draft tells itself).
- Task: for each claim, is it entailed by the cited evidence? Does the impression assert anything no claim supports? Is anything in the facts material but absent?
- Output: `CriticVerdict` with per-claim issues. Runs on a **different model or at minimum a different temperature/setting** from the Director (default: GPT-4o-mini at temperature 0 vs the Director's configured generation temperature; a different pinned snapshot is better and is a one-line config change). Self-critique by the same configuration is weak evidence.
- The Critic checks claims against **both** the CAD metrics (calibrated p, CDR, quality/QC flags) **and** the clinical-note facts (IOP, meds, intolerances, comorbidities) — with note free text treated as untrusted data.
- Known limit: an LLM critic can approve a wrong statement. That is why the deterministic layer exists, why it runs first, and why a human is final (I-4).

### 13.3 Merge and loop
`verdict = merge(deterministic, critic)`; approved ⇔ no issues from either. Rejections go back to the Director (max `CRITIC_MAX_ROUNDS=3`). If the loop ends unapproved → `NEEDS_ATTENTION`; draft shown only as `UNVERIFIED` with open issues highlighted (§4). **Disagreement entries alone never trigger this** — only unresolved audit issues do.

### 13.4 Validating the critic itself
A critic you have not attacked is a critic you cannot trust. `eval/fault_injection.py` takes known-good drafts and applies seeded faults:
1. alter a number (CDR 0.62 → 0.82, and subtle 0.62 → 0.58),
2. fabricate a finding with a plausible ref,
3. fabricate a finding with no ref,
4. delete a mandatory finding / caveat,
5. insert a dose/definitive-diagnosis sentence,
6. flip laterality,
7. suppress a real disagreement.
Report per-fault-class **recall** and the **false-rejection rate** on clean drafts. Layer-by-layer (deterministic only, LLM only, both). This is the real evidence for the paper's hallucination claim.

---

## 14. Guardrails (NeMo Guardrails — narrow and deliberate)

CONSILIUM is a pipeline, not a conversation. Use only what protects it:

- **Input rail** — on the raw medical note's free text, **before extraction**: prompt-injection and instruction-like content detection, length/charset limits, PHI pattern detection (names, phone, MRN-like strings → block or redact, per config). Triggered before routing.
- **Output rail** — on `DirectorDraft` before it is shown to a clinician: definitive-diagnosis phrasing, dose/regimen content, PHI echo, missing mandatory disclaimers, non-clinical content.
- **Not used:** dialog/topic rails (no free dialog exists), retrieval rails (no retrieval in V1), and **no guardrail in front of SwinV2/SegFormer** (deterministic models; they get schema + quality gates instead).

Division of responsibility — do not blur it:

| Layer | Question it answers |
|---|---|
| Pydantic | Is this the right *shape*? |
| Router + leak tests | Did this agent see only what it should? |
| Deterministic verifier | Are the numbers/references/coverage right? |
| LLM Critic | Is each claim *entailed* by evidence? |
| NeMo rails | Is the content *allowed* (policy, injection, PHI, tone)? |
| Clinician | Is this clinically right for this patient? |

Implementation split: simple patterns (dose regex, phrase lists, PHI patterns) live in deterministic Python pre-checks — cheaper and testable offline; NeMo handles what needs a model (chiefly injection detection). Guardrail failures are logged as audit events with the rail name and reason, never swallowed. Rail config lives in `configs/guardrails/`; each rule has a test in `tests/adversarial/`.

---

## 15. Clinician review and feedback capture

Review is a service (`review/`), not a UI concern. The React UI is a client of this service and contains no status logic.

Actions: `APPROVE`, `APPROVE_WITH_EDITS`, `REJECT`. Edits are field-level changes to the `DirectorDraft`, stored as a structured diff against the exact AI draft revision they were made on.

Every edit/rejection requires a **reason code** (closed taxonomy, extendable only via `schema-change`):

```
FACTUAL_ERROR_CAD        # misread/misstated a CAD value
CLINICAL_INTERPRETATION  # wrong clinical reading of correct facts
KNOWLEDGE_GAP            # missing or wrong domain knowledge
UNSAFE_RECOMMENDATION
OMISSION
TONE_OR_WORDING
SCOPE_VIOLATION
OTHER   (free-text required)
```

Why this matters now even though V1 has no learning loop: **this table is the dataset for V2.** The RAG decision gate (§23) is evaluated on these reason codes. Capture richly, store plainly, retrieve nothing.

Seams for V2 (do not implement now): a `FeedbackStore` protocol in `review/` with `record(correction)`; V1 implementation writes to Postgres. A future retrieval implementation would be injected into the Director's prompt builder via an interface that V1 binds to a no-op. Corrections, when eventually retrieved, are *advisory constraints* and never override CAD facts or verifier output.

An approved report is immutable. Amendments create a new version linked to the old one.

---

## 16. Persistence

PostgreSQL via SQLAlchemy 2.x; migrations via Alembic only (no `create_all` outside tests).

Tables (columns abbreviated; add `created_at`, `updated_at`, and FK/index definitions in the migration):

```
cases              id (uuid, opaque), status, laterality, image_sha256, image_uri,
                   idempotency_key (unique), created_by
clinical_notes     case_id, note_json (ClinicalNote), extraction_json (NoteExtraction),
                   raw_redacted_text_enc    # encrypted at rest; raw text never in logs
cad_results        case_id, cad_json (CADResult), raw_vision_json (RawVisionOutput),
                   classifier_version, segmenter_version, endpoint_revision
vision_calls       id, case_id, endpoint_revision, latency_ms, attempts, status, error
routed_contexts    case_id, role, routing_version, facts_json, prompt_hash
agent_runs         id, case_id, agent (note_extractor|role|director|critic), prompt_version, model_id,
                   input_tokens, output_tokens, latency_ms, attempt, status, output_json, error
drafts             case_id, revision, draft_json, producer_run_id, unverified (bool)
verdicts           case_id, revision, round, approved, issues_json, layer
reviews            case_id, draft_revision, reviewer_id, action, edits_diff_json,
                   reason_codes, free_text_reason, decided_at
audit_log          id, case_id, event, actor, detail_json, at   -- append-only
```

Rules:
- `audit_log`, `verdicts`, `reviews` are append-only (revoke UPDATE/DELETE in the DB role).
- Never store the free-text note in `audit_log`/logs; it lives only in `clinical_notes`, encrypted at rest at the storage layer.
- Images are content-addressed (`sha256`) in object storage; DB stores URI + hash.
- Retention and deletion policy must be implemented as a job with tests (right-to-erasure path), even if the policy values are set later.

---

## 17. API

```
POST /v1/cases                       multipart: image + raw note text  (Idempotency-Key header)
GET  /v1/cases/{id}                  status + links
GET  /v1/cases/{id}/draft            DirectorDraft + open issues + CAD summary + overlay URL;
                                     UNVERIFIED banner data when critic did not approve
POST /v1/cases/{id}/review           {action, edits?, reason_codes?, comment?}
GET  /v1/cases/{id}/report           only when APPROVED*; includes clinician attribution
GET  /v1/cases/{id}/overlay          mask overlay image for the review UI
GET  /healthz   /readyz   /metrics
```

- Pipeline runs asynchronously (worker/background task); API returns 202 + case ID.
- Error model: RFC 7807 problem+json; internal errors never leak stack traces, prompts, or user content.
- Authentication: bearer tokens with at least `submitter` and `clinician` roles; only `clinician` can call `/review`. A submitter cannot approve their own case if they are the same identity.
- Mask overlays for the review UI are generated from stored masks; the UI renders them as an interactive overlay (React), with the edit-and-diff form beside it.

---

## 18. Observability

- One trace per case (OpenTelemetry); spans for each graph node, each LLM call, and each vision call. Attributes: case ID (opaque), model ID, prompt version, token counts, retry attempt, verifier issue counts, endpoint revision, cold-start flag. **No prompt/response bodies and no image data in traces by default**; a debug flag may enable bodies in non-production only.
- **Langfuse (self-hosted)** sits on the LLM layer: prompt versions, per-agent generations, token cost, critic and reviewer annotations. Self-host because real note text will eventually reach it; Langfuse Cloud is acceptable for dev with synthetic data only. Note the self-host footprint: recent versions need ClickHouse, Redis, and blob storage alongside Postgres — check current docs before committing.
- Metrics: stage latency histograms, LLM error/retry/timeout counts, vision endpoint latency + cold-start count, critic rejection rate by issue type, rounds-to-approval, `NEEDS_ATTENTION` rate, clinician approval/edit/reject rates by reason code, cost per case.
- Alerts (document thresholds in README): vision failure spike, vision cold-start rate spike, critic-approval-in-round-1 rate suddenly near 100% (possible critic degradation), edit-rate drift.

---

## 19. Evaluation

`eval/` is a first-class package, run in CI (small fixed set, fake or recorded LLM and recorded vision responses) and on demand (full set, live LLM). Every run stores: git SHA, config hashes, model IDs, prompt versions, dataset version, seed, and metrics, tagged with a run ID.

### 19.1 Four-stage ablation (from the paper)
| Stage | Configuration |
|---|---|
| S1 | Single agent, full context |
| S2 | Multi-agent, **shared** context (MedChat-like baseline) |
| S3 | Multi-agent, **routed** context |
| S4 | Routed + Verification (full CONSILIUM) |

### 19.2 Metrics
- **Reasoning diversity.** `S_inter` = mean pairwise cosine similarity of sub-report embeddings (report 1 − S_inter as "diversity" if you want a distance; **pick one and use it consistently** — the paper's text mixes "distance" and "similarity"). The embedding model used for this metric is fixed and versioned for the whole study and lives **only in `eval/`** (I-10); report absolute values *and* which model produced them, since the value is embedding-dependent. Complement with a model-free measure: **unique-fact coverage** — fraction of each role's expected role-specific findings surfaced.
- **Hallucination/unsupported-claim rate** — claim-level, using the verifier's claim list plus a clinician-labelled subset. Report precision/recall of the critic via §13.4.
- **Omission rate** — mandatory-finding coverage.
- **Minority-opinion preservation** — on a curated set of cases where roles *should* disagree, fraction where the draft retains the dissent.
- **Operational:** rounds-to-approval, latency p50/p95, tokens and cost per case.
- **Clinical quality:** rubric scoring by clinicians on a sampled subset (accuracy, completeness, safety, usefulness). LLM-as-judge may supplement, never replace; if used, document judge model and validate it against clinician scores on the same subset.

### 19.3 Rigor
- Fixed case set with a frozen holdout; N large enough for CIs (bootstrap, 95%). Repeat LLM runs (≥3 seeds/samples) and report variance — LLM output is stochastic even at low temperature.
- Same cases, same CAD results (recorded endpoint responses), same model snapshots across stages; only the architecture varies.
- The paper's results section is qualitative and its Figure 3 has no axis values. The harness must produce the real numbers; nothing in the paper, README or reports may cite an improvement that is not in a stored run (I-11).
- Report negative results. If S3 does not beat S2 on diversity, that is a finding, not a bug to hide.

---

## 20. Testing requirements

CI blocks merge on: `ruff`, `mypy --strict src/`, `pytest -m "not live"`, coverage floor (set in `pyproject.toml`; raise over time, never lower).

| Suite | What it proves |
|---|---|
| `unit/` | CDR math (synthetic masks), tier mapping, calibration, fact-ID builder, config parsing, diff logic, de-identification patterns, extraction validators |
| `contract/` | Every schema round-trips; JSON-Schema snapshots committed; any change is a conscious diff |
| `unit/test_router_leak.py` | **Property-based (hypothesis):** for arbitrary `GlobalContext`, no forbidden fact ID, label, or *numeric value* appears in any role's rendered prompt |
| `unit/test_specialist_scope.py` | Specialist output citing out-of-scope refs is rejected |
| `unit/test_verifier.py` | Each deterministic check, including boundary tolerances |
| `adversarial/` | Prompt-injection notes, PHI-in-note, definitive-diagnosis bait, dose bait, oversized input, unicode tricks; asserts rails/verifier catch them |
| `integration/` | Graph with **fake LLM client and fake vision client**: happy path, specialist failure, critic reject→revise→approve, loop exhaustion → NEEDS_ATTENTION (UNVERIFIED draft contract), crash-resume, vision cold-start retry, vision revision-pin mismatch |
| `integration/` (DB) | Alembic upgrade/downgrade, append-only enforcement, idempotency |
| `e2e/` | Image fixture (synthetic or licensed-safe) + note fixture → full pipeline with recorded LLM + recorded vision responses → `PENDING_REVIEW`; review action → approved report |
| `live/` (`-m live`, manual/nightly) | Real OpenAI calls and (optionally) the dev endpoint on a tiny fixed set; checks schema conformance rate and regression on key metrics |

LLM testing rule: **the default test suite never calls a real LLM and never calls the network.** All agents depend on the `LLMClient` protocol; all vision access depends on the `VisionClient` protocol; tests use fakes returning canned structured outputs (including deliberately malformed ones) and recorded HTTP fixtures.

Fixtures contain no real patient data. Synthetic notes and synthetic images only.

---

## 21. Code standards

- Typed everywhere; no `Any` at module boundaries; `mypy --strict` clean.
- Functions at boundaries take and return schema models, not dicts.
- No global mutable state; dependencies injected (needed for the fake-LLM, fake-vision, and fake-store tests).
- Async at I/O edges (LLM, DB, object store, vision endpoint); CPU-bound post-processing (mask ops, calibration) is pure and cheap — if it ever becomes heavy, run it in a bounded executor, never block the event loop.
- Errors: a small exception hierarchy in `core/errors.py` (`InputRejected`, `VisionFailure`, `AgentFailure`, `ValidationFailure`, `PolicyViolation`); each maps to a case status and an API problem type. Do not catch bare `Exception` except at the node boundary where it is converted and recorded.
- Logging: structured, no PHI, no upload filenames, include case ID and node name.
- Determinism: seed everything in training/eval; record seeds.
- Dependencies pinned via `uv.lock`; adding one requires a stated reason in the PR. Heavy ML deps stay in the `vision`/`training` groups so the API image stays lean.
- Commits are small and each leaves CI green. A PR changing prompts, thresholds, routing, or schemas says so in the title.

---

## 22. Implementation order

Do these in order. Each phase has an **exit criterion**; don't start the next until it is met. Use the matching skill in SKILLS.md.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 | Repo scaffold, CI, lint/type/test gates, Docker, docker-compose (api, postgres, object store, **fake vision service**, optional Langfuse), settings, logging | `make ci` green on empty skeleton |
| P1 | `schemas/` complete + contract tests + fact-ID module | JSON-Schema snapshots committed; round-trip tests pass |
| P2 | Vision: data pipeline, SegFormer, SwinV2 training on REFUGE, quality gates, CDR, calibration, endpoint packaging, custom handler deployed to a **dev HF endpoint**, `VisionClient` (HTTP + fake) with revision pin, cold-start handling, `vision.pipeline.run(image_uri) -> CADResult` | Reported metrics on val/test stored in MLflow; CDR unit tests pass; fake + recorded contract tests green; revision-pin mismatch test passes |
| P3 | Note intake: de-identification, extractor prompt v1, extraction via LLMClient, validation + repair retry | Adversarial note suite passes; extraction provenance stored; redaction tests green |
| P4 | Context builder + router + routing.yaml + leak tests | Leak property test passes at ≥1000 examples; clinician-reviewed routing recorded |
| P5 | `LLMClient` hardening (retry/timeout/budget), prompt loader, three specialists + prompts v1 | Scope-violation tests pass; live schema-conformance ≥ target on a small set |
| P6 | Director + prompt v1 | Claims carry valid refs; disagreements preserved on curated cases |
| P7 | Deterministic verifier | All check types unit-tested; seeded-fault recall measured |
| P8 | LLM Critic (different model/setting) + merge + bounded loop + UNVERIFIED draft contract | Fault-injection report per layer; false-rejection rate measured |
| P9 | Graph orchestration, state persistence, resume | Crash-resume and every failure path covered in integration tests |
| P10 | Guardrails (input/output rails) | Adversarial suite passes; rail events audited |
| P11 | Persistence, API, auth, review service, reason codes | E2E: submit → draft → review → approved report with audit trail |
| P12 | React review UI (case list, overlay viewer, edit-and-diff, reason codes, UNVERIFIED banner) | Clinician can complete a full review workflow against a dev stack |
| P13 | Full evaluation: S1–S4 ablation, CIs, clinician rubric subset | Run stored; README/paper numbers regenerated from stored runs only |
| P14 | Hardening: security review, load test, observability dashboards (OTel + Langfuse), runbook, DR/backup test | Documented runbook; load target met; threat-model doc reviewed |

---

## 23. V2 gate — when (and only when) to add RAG

The V1 decision is "OpenAI-backed agents first; add retrieval only if the answers are poor". Make "poor" measurable so the decision isn't vibes:

Open the V2 discussion when **any** holds over a statistically meaningful sample of clinician-reviewed cases:
- `KNOWLEDGE_GAP` + `CLINICAL_INTERPRETATION` reason codes exceed an agreed share of edits (set the threshold with the clinical co-authors up front);
- the critic/verifier repeatedly rejects drafts for domain-knowledge errors rather than evidence mismatches;
- the same clinician correction recurs across cases (repeated mistakes — the original motivation for the paper's feedback memory);
- clinician rubric scores for "accuracy" or "safety" fall below the agreed floor.

If the dominant problem is *numeric/evidence* errors, RAG is the wrong fix — tighten the verifier and prompts. If it's *knowledge*, add guideline retrieval first (curated, versioned sources); clinician-correction memory second (the stored `reviews` rows are the raw material); LLM fine-tuning last and only on curated, consented data. Each addition is evaluated as a new ablation stage (S5, S6…) against S4.

---

## 24. Known issues in the source paper (do not propagate)

1. **CDR formula** — see §8.2.
2. **No quantitative results** — the results section and Figure 3 are qualitative; the ablation table is qualitative. Produce real numbers (§19).
3. **"Similarity" vs "distance"** — introduction speaks of pairwise cosine *distance*, results of cosine *similarity* `S_inter`. Choose one.
4. **Critic scope** — the paper checks drafts against CAD metrics; clinical-note facts are also ground truth in practice. Our verifier treats `note.*` facts as evidence, with the caveat that they are clinician-supplied and untrusted as text.
5. **Memory** — the paper's pgvector feedback memory is deferred by decision (I-10); the paper should describe V1 accordingly or label the memory as design-only until implemented and evaluated. Corrections are captured relationally now.
6. **Isolation claim** — "physical separation stops semantic convergence" is a hypothesis; it is also possible that same-base-model agents converge regardless. The ablation decides.
7. **Vision serving assumed in-repo** — the paper assumes models run inside the application. V1 serves them remotely on a private HF Inference Endpoint behind `VisionClient` (§8.0); all interpretation stays in-repo.

---

## 25. Deployment topologies and compliance posture

Cloud provider (AWS/GCP/Azure vs on-prem) is **deliberately undecided** — the architecture must not bake in a vendor. The seams that keep it portable:

- `VisionClient` binding: HF Inference Endpoint (default) ↔ self-hosted handler (on-prem GPU box, Triton, or a second compose service) — one config value, no code change.
- Object storage: local FS (dev) ↔ S3-compatible (prod) — one URI scheme.
- Postgres: container (dev) ↔ managed (prod) — one `DATABASE_URL`.
- Langfuse self-hosted alongside the stack (§18).

Compliance posture for V1 (HIPAA / data residency TBD by the team — do not guess):
- Encryption in transit everywhere (TLS; API↔endpoint, API↔DB, API↔object store).
- Encryption at rest: DB volumes, object store, backups; `clinical_notes.raw_redacted_text_enc` column-level encryption.
- Access controls: bearer-token auth, `submitter`/`clinician` roles, no self-approval, append-only audit tables, DB role privileges revoked for UPDATE/DELETE on audit tables.
- Minimum necessary PHI (I-6): opaque case IDs, de-identified notes before any LLM call, no image to the LLM provider (I-14).
- Provider review: check OpenAI's and Hugging Face's current data-retention / zero-data-retention terms before any real patient-derived content is processed; record the decision in `README.md`.
- The clinician-facing report is the only output surface; every report carries screening-support wording and clinician attribution. Nothing in the UI implies autonomous diagnosis.

---

## 26. How you (the coding agent) should work here

1. **Before any change:** read this file and the relevant skill in SKILLS.md. For anything touching schemas, routing, prompts, thresholds, the verifier, the endpoint contract, or the extraction pipeline, the matching skill is mandatory.
2. **Plan first** for any change spanning more than one module: state which invariants and contracts are touched and which tests will prove it.
3. **Tests with the change**, not after. A bug fix starts with a failing test.
4. **Never** — without explicit human instruction:
   - add RAG, embeddings in the serving path, pgvector, or LLM fine-tuning;
   - let an LLM produce or modify a CAD number;
   - widen a role's routing allowlist;
   - set a case to `APPROVED` from anywhere except the review service;
   - loosen a verifier tolerance, a forbidden-language rule, or a calibration threshold to make a test pass;
   - lower the coverage floor or mark a failing test `xfail` to get CI green;
   - commit data, weights, secrets, or anything resembling real patient information;
   - call a real LLM or the network from the default test suite;
   - run a forward pass in the API process (use `VisionClient`);
   - claim a metric you did not measure.
5. **When unsure about a clinical question** (what a role should see, what wording is safe, what a threshold should be): do not guess. Implement the mechanism, make the clinical decision a config value, mark it `# CLINICAL-REVIEW:` and surface it in your summary.
6. **Report honestly.** If a check fails, say so. If you couldn't run the live suite or GPU training, say so rather than implying it passed.
7. **Leave the repo runnable:** `make ci` passes, migrations apply cleanly, docs match code.

### Commands (create these Makefile targets in P0)

```
make setup        # uv sync --all-groups, pre-commit install
make lint         # ruff check + ruff format --check
make typecheck    # mypy --strict src/
make test         # pytest -m "not live"          (no network, fakes only)
make test-live    # pytest -m live               (needs OPENAI_API_KEY; optional live endpoint)
make ci           # lint + typecheck + test + schema-snapshot check
make eval-smoke   # small fixed set, fake/recorded LLM + recorded vision
make eval-full    # full ablation, live LLM + dev endpoint, stores a run
make migrate      # alembic upgrade head
make serve        # uvicorn consilium.api.main:app --reload
make up / down    # docker compose stack (api, postgres, object store, fake vision, [langfuse])
```

### Definition of done (any task)
- Invariants intact; contracts unchanged or changed via `schema-change`.
- Tests added/updated; `make ci` green.
- No PHI, secrets, or data artifacts committed.
- Docs/README/config comments reflect the change.
- Summary states: what changed, what was verified and how, what was *not* verified, and any `CLINICAL-REVIEW` items.
