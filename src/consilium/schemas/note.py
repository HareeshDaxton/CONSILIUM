"""Clinical-note contracts (AGENTS.md §7, §10).

``ClinicalNote`` is the structured, de-identified output of note intake;
``NoteExtraction`` is the provenance record stored alongside it
(``clinical_notes.note_json`` / ``extraction_json``).

UNTRUSTED DATA (I-7): ``exam_findings`` and ``free_text`` may contain
injection attempts. They are data, never instructions — downstream prompts
place them in delimited, non-instructional blocks and never concatenate them
into system prompts.
"""

from pydantic import Field

from consilium.schemas.base import Strict
from consilium.schemas.enums import Confidence


class ClinicalNote(Strict):
    """Post-de-identification, post-extraction structured note."""

    age_years: int | None = Field(default=None, ge=0, le=120)
    iop_mmhg_od: float | None = None
    iop_mmhg_os: float | None = None
    current_medications: tuple[str, ...] = ()
    drug_intolerances_or_allergies: tuple[str, ...] = ()
    comorbidities: tuple[str, ...] = ()
    prior_glaucoma_dx: str | None = None
    exam_findings: str | None = Field(default=None, max_length=2000)  # untrusted
    free_text: str | None = Field(default=None, max_length=4000)  # untrusted


class NoteExtraction(Strict):
    """Provenance for the extraction step (I-8, §10.5).

    ``redactions`` records which PHI patterns fired — pattern NAMES only,
    never the redacted values (I-6).
    """

    note: ClinicalNote
    extractor_model: str  # pinned snapshot, e.g. gpt-4o-mini-YYYY-MM-DD
    extractor_prompt_version: str
    redactions: tuple[str, ...] = ()
    extraction_confidence: Confidence  # LOW surfaces a caveat fact downstream
