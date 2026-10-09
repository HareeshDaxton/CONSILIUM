"""Fact-ID constants — the single source of truth (AGENTS.md §7, §9.1).

Fact IDs are an API. They are defined ONLY here; never inline a fact-ID
string anywhere else in the codebase (schema-change skill, doc/SKILLS.md §2).
The Context Builder (P4) emits exactly these IDs; routing.yaml allowlists
reference them; agents cite them in evidence_refs.

Shape follows ARCHITECTURE.md §5.6 exactly.
"""


class F:
    # --- CAD facts (source: vision post-processing, I-1) ---
    P_CAL = "cad.p_glaucoma.calibrated"
    P_TIER = "cad.p_glaucoma.tier"
    CDR_V = "cad.cdr.vertical"
    CDR_A = "cad.cdr.area_based"
    DISC_AREA = "cad.disc.area_px"
    CUP_AREA = "cad.cup.area_px"
    DISC_VD = "cad.disc.vdiam_px"
    CUP_VD = "cad.cup.vdiam_px"
    Q_SCORE = "cad.quality.score"
    Q_FLAGS = "cad.quality.flags"
    MASKQC = "cad.maskqc.ok"
    LAT = "cad.laterality"

    # --- note facts (source: de-identified clinician note, untrusted text) ---
    AGE = "note.age"
    IOP_OD = "note.iop.od"
    IOP_OS = "note.iop.os"
    MEDS = "note.meds"
    INTOL = "note.intolerances"
    COMORB = "note.comorbidities"
    PRIOR_DX = "note.prior_dx"
    EXAM = "note.exam_findings"
    FREE = "note.free_text"
    CAVEAT_EXTRACTION = "note.caveat.extraction"

    @staticmethod
    def caveat(flag: str) -> str:
        """Caveat fact ID for a quality flag: ``cad.caveat.<flag>``."""
        return f"cad.caveat.{flag}"
