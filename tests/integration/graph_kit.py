"""Shared offline kit for the graph integration suites (plan 9.3 / 9.4).

FakeLLMClient + FakeVisionClient + MemoryImageStore + the real local Postgres
(``repos`` fixture). No network, no real LLM (§20). The Director's canned
draft is derived from the GlobalContext the pipeline will ACTUALLY build from
the fake vision output, so it is verifier-clean by construction (no numerals
in prose, every caveat token carried into ``limitations``).
"""

from __future__ import annotations

from typing import Any

from consilium.context import builder
from consilium.context.facts import F
from consilium.core.settings import DEFAULT_CRITIC_PROMPT_VERSION, Settings
from consilium.core.settings import DEFAULT_DIRECTOR_PROMPT_VERSION as DIRECTOR_V
from consilium.graph.build import GraphRunner, NodeHook
from consilium.graph.deps import MemoryImageStore, build_fake_deps
from consilium.intake.extractor import estimate_confidence
from consilium.llm.client_fake import CannedCall, CannedResponse, FakeLLMClient, FakeMode
from consilium.persistence.repositories import Repositories
from consilium.schemas.agents import Claim, DirectorDraft
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import ClaimKind, Laterality, Role
from consilium.schemas.note import ClinicalNote, NoteExtraction
from consilium.vision import client_fake as vf
from consilium.vision import pipeline as vision_pipeline
from consilium.vision.client import VisionClient
from consilium.vision.config import load_models_config, load_thresholds

PIN = load_models_config().endpoint.pinned_revision
IMAGE = vision_pipeline.synthetic_fundus_bytes(256)
IMAGE_URI = "mem://case-image-1"
IMAGE_SHA = "a" * 64
# Synthetic, PHI-free note text (fixtures never contain real patient data).
NOTE_TEXT = "Synthetic referral. IOP right eye 21 mmHg. Age 65."
NOTE_PAYLOAD: dict[str, object] = {"age_years": 65, "iop_mmhg_od": 21.0}

APPROVE: dict[str, Any] = {"approved": True, "issues": [], "round": 1}
REJECT: dict[str, Any] = {
    "approved": False,
    "issues": [
        {
            "type": "unsupported",
            "claim_id": "c2",
            "detail": "tier claim needs the calibrated probability as evidence",
            "evidence": [],
            "raised_by": "llm_critic",
        }
    ],
    "round": 1,
}

_SUBREPORT = {
    "findings": [
        {
            "text": "Screening tier is available.",
            "evidence_refs": [F.P_TIER],
            "confidence": "moderate",
        }
    ],
    "recommendations": [{"text": "Follow the screening pathway.", "triggered_by": [F.P_TIER]}],
    "uncertainties": [],
    "declined_out_of_scope": [],
}


async def expected_ctx() -> GlobalContext:
    """The GlobalContext the graph will build for IMAGE + NOTE_PAYLOAD."""
    cad = await vision_pipeline.run(
        IMAGE,
        client=vf.FakeVisionClient(revision=PIN),
        thresholds=load_thresholds(),
        models=load_models_config(),
        laterality=Laterality.UNKNOWN,
    )
    note = ClinicalNote.model_validate(NOTE_PAYLOAD, strict=False)
    extraction = NoteExtraction(
        note=note,
        extractor_model="fake",
        extractor_prompt_version="v1",
        extraction_confidence=estimate_confidence(note),
    )
    return builder.build(cad, note, extraction, "kit")


def clean_draft(ctx: GlobalContext) -> DirectorDraft:
    facts = {f.id: f for f in ctx.facts}
    cdr = facts[F.CDR_V].value
    assert isinstance(cdr, float)
    caveats = [
        "Caveat carried: " + f.id.rsplit(".", 1)[-1].replace("_", " ") + "."
        for f in ctx.facts
        if f.id.startswith("cad.caveat.") or f.id == F.CAVEAT_EXTRACTION
    ]
    return DirectorDraft(
        impression="Findings warrant evaluation for glaucoma by a clinician.",
        claims=(
            Claim(
                id="c1",
                kind=ClaimKind.NUMERIC,
                text="Vertical cup-to-disc ratio is as measured.",
                evidence_refs=(F.CDR_V,),
                source_roles=(Role.OPHTHALMOLOGIST,),
                numeric_value=cdr,
                numeric_ref=F.CDR_V,
            ),
            Claim(
                id="c2",
                kind=ClaimKind.CATEGORICAL,
                text="Screening tier is as computed.",
                evidence_refs=(F.P_TIER,),
                source_roles=(Role.OPHTHALMOLOGIST,),
            ),
            Claim(
                id="c3",
                kind=ClaimKind.CATEGORICAL,
                text="Laterality is recorded as unknown.",
                evidence_refs=(F.LAT,),
                source_roles=(Role.OPHTHALMOLOGIST,),
            ),
        ),
        limitations=("Screening support only; not a diagnosis.", *caveats),
    )


def fake_llm(
    draft: DirectorDraft,
    *,
    critic: tuple[dict[str, Any], ...] = (APPROVE,),
    refuse_role: str | None = None,
) -> FakeLLMClient:
    d = draft.model_dump(mode="json")
    keys: dict[tuple[str, str, str], CannedResponse] = {
        ("note_extractor", "v1", "default"): CannedResponse(
            (CannedCall(FakeMode.VALID, NOTE_PAYLOAD),)
        ),
        ("director", DIRECTOR_V, "default"): CannedResponse((CannedCall(FakeMode.VALID, d),)),
        ("critic", DEFAULT_CRITIC_PROMPT_VERSION, "default"): CannedResponse(
            tuple(CannedCall(FakeMode.VALID, c) for c in critic)
        ),
    }
    for role in ("ophthalmologist", "optometrist", "pharmacist"):
        call = (
            CannedCall(FakeMode.REFUSAL)
            if role == refuse_role
            else CannedCall(FakeMode.VALID, {"role": role, **_SUBREPORT})
        )
        keys[(role, "v1", "default")] = CannedResponse((call,))
    return FakeLLMClient(keys)


def make_runner(
    settings: Settings,
    repos: Repositories,
    *,
    llm: FakeLLMClient,
    vision: VisionClient | None = None,
    images: dict[str, bytes] | None = None,
    hook: NodeHook | None = None,
) -> GraphRunner:
    deps = build_fake_deps(
        settings,
        llm=llm,
        vision=vision or vf.FakeVisionClient(revision=PIN),
        image_store=MemoryImageStore(images if images is not None else {IMAGE_URI: IMAGE}),
    )
    return GraphRunner(deps, repos, on_node_persisted=hook)


async def submit(runner: GraphRunner, *, key: str | None = None, note: bool = True) -> Any:
    return await runner.submit(
        image_uri=IMAGE_URI,
        image_sha256=IMAGE_SHA,
        created_by="submitter-test",
        raw_note=NOTE_TEXT if note else None,
        idempotency_key=key,
    )


async def count_rows(repos: Repositories, model: type, case_id: str, **where: Any) -> int:
    """Test-only row count for one case (optionally filtered by column values)."""
    import uuid

    from sqlalchemy import func, select

    stmt = select(func.count()).select_from(model).where(model.case_id == uuid.UUID(case_id))  # type: ignore[attr-defined]
    for col, val in where.items():
        stmt = stmt.where(getattr(model, col) == val)
    async with repos._session_factory() as s:
        return int((await s.execute(stmt)).scalar_one())
