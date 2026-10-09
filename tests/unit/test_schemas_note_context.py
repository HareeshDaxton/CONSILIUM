"""Section 1.3 — note + context contracts (AGENTS.md §7, §9)."""

import pytest
from pydantic import ValidationError

from consilium.schemas.context import Fact, GlobalContext, RoleContext
from consilium.schemas.enums import Confidence, FactSource, Laterality, Role
from consilium.schemas.note import ClinicalNote, NoteExtraction


class TestClinicalNote:
    def test_defaults_all_empty(self) -> None:
        note = ClinicalNote()
        assert note.age_years is None
        assert note.current_medications == ()
        assert note.exam_findings is None
        assert note.free_text is None

    @pytest.mark.parametrize("age", [-1, 121])
    def test_age_bounds(self, age: int) -> None:
        with pytest.raises(ValidationError):
            ClinicalNote(age_years=age)

    def test_age_boundary_values_ok(self) -> None:
        assert ClinicalNote(age_years=0).age_years == 0
        assert ClinicalNote(age_years=120).age_years == 120

    def test_free_text_length_caps(self) -> None:
        with pytest.raises(ValidationError):
            ClinicalNote(exam_findings="x" * 2001)
        with pytest.raises(ValidationError):
            ClinicalNote(free_text="x" * 4001)
        assert ClinicalNote(exam_findings="x" * 2000).exam_findings is not None
        assert ClinicalNote(free_text="x" * 4000).free_text is not None

    def test_full_construction_round_trip(self) -> None:
        note = ClinicalNote(
            age_years=67,
            iop_mmhg_od=24.0,
            iop_mmhg_os=22.5,
            current_medications=("metformin",),
            drug_intolerances_or_allergies=("sulfa",),
            comorbidities=("type 2 diabetes",),
            prior_glaucoma_dx="none",
            exam_findings="disc appears healthy",
            free_text="patient reports no complaints",
        )
        assert ClinicalNote.model_validate_json(note.model_dump_json()) == note


class TestNoteExtraction:
    def test_round_trip_with_provenance(self) -> None:
        extraction = NoteExtraction(
            note=ClinicalNote(age_years=67),
            extractor_model="gpt-4o-mini-2024-07-18",
            extractor_prompt_version="note_extractor/v1",
            redactions=("person_name", "phone_number"),
            extraction_confidence=Confidence.MODERATE,
        )
        restored = NoteExtraction.model_validate_json(extraction.model_dump_json())
        assert restored == extraction
        assert restored.extraction_confidence == Confidence.MODERATE

    def test_redactions_default_empty(self) -> None:
        extraction = NoteExtraction(
            note=ClinicalNote(),
            extractor_model="m",
            extractor_prompt_version="v1",
            extraction_confidence=Confidence.HIGH,
        )
        assert extraction.redactions == ()


def _fact(fid: str, value: str | float | int | bool | tuple[str, ...] = "v") -> Fact:
    return Fact(id=fid, label=fid, value=value, source=FactSource.CAD)


class TestFact:
    def test_value_union_preserves_types(self) -> None:
        """Smart union must not coerce: int stays int, bool stays bool."""
        assert isinstance(_fact("a", 5).value, int)
        assert isinstance(_fact("b", True).value, bool)
        assert isinstance(_fact("c", 0.62).value, float)
        assert isinstance(_fact("d", "high").value, str)
        assert _fact("e", ("a", "b")).value == ("a", "b")

    def test_id_and_label_required_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            _fact("")

    def test_unit_optional(self) -> None:
        fact = Fact(
            id="cad.cdr.vertical", label="vCDR", value=0.62, unit="ratio", source=FactSource.CAD
        )
        assert fact.unit == "ratio"


def _global_ctx(facts: tuple[Fact, ...]) -> GlobalContext:
    return GlobalContext(case_id="case-1", laterality=Laterality.OD, facts=facts)


class TestGlobalContext:
    def test_unique_ids_enforced(self) -> None:
        with pytest.raises(ValidationError, match="duplicate fact IDs"):
            _global_ctx((_fact("cad.cdr.vertical"), _fact("cad.cdr.vertical")))

    def test_distinct_ids_ok(self) -> None:
        ctx = _global_ctx((_fact("cad.cdr.vertical"), _fact("cad.p_glaucoma.tier")))
        assert ctx.fact_ids() == {"cad.cdr.vertical", "cad.p_glaucoma.tier"}

    def test_round_trip(self) -> None:
        ctx = _global_ctx((_fact("cad.cdr.vertical", 0.62),))
        assert GlobalContext.model_validate_json(ctx.model_dump_json()) == ctx


class TestRoleContext:
    def _role_ctx(self) -> RoleContext:
        return RoleContext(
            role=Role.PHARMACIST,
            case_id="case-1",
            laterality=Laterality.OS,
            facts=(_fact("note.meds"), _fact("note.age")),
            routing_version="sha256:abc",
        )

    def test_allowed_ids_is_fact_id_set(self) -> None:
        assert self._role_ctx().allowed_ids() == frozenset({"note.meds", "note.age"})

    def test_unique_ids_enforced(self) -> None:
        with pytest.raises(ValidationError, match="duplicate fact IDs"):
            RoleContext(
                role=Role.OPTOMETRIST,
                case_id="case-1",
                laterality=Laterality.OD,
                facts=(_fact("x"), _fact("x")),
                routing_version="sha256:abc",
            )

    def test_round_trip(self) -> None:
        ctx = self._role_ctx()
        restored = RoleContext.model_validate_json(ctx.model_dump_json())
        assert restored == ctx
        assert restored.allowed_ids() == ctx.allowed_ids()
