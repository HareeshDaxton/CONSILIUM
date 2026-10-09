"""Section 1.4 — agent output + audit contracts (AGENTS.md §7, §11-13)."""

import pytest
from pydantic import ValidationError

from consilium.schemas.agents import (
    Claim,
    DirectorDraft,
    Disagreement,
    Finding,
    Recommendation,
    SubReport,
)
from consilium.schemas.audit import CriticVerdict, Issue
from consilium.schemas.enums import ClaimKind, Confidence, IssueType, Role


class TestFinding:
    def test_valid(self) -> None:
        f = Finding(
            text="vCDR elevated",
            evidence_refs=("cad.cdr.vertical",),
            confidence=Confidence.HIGH,
        )
        assert f.confidence == Confidence.HIGH

    def test_text_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Finding(text="", evidence_refs=("x",), confidence=Confidence.LOW)
        with pytest.raises(ValidationError):
            Finding(text="x" * 601, evidence_refs=("x",), confidence=Confidence.LOW)

    def test_evidence_refs_required(self) -> None:
        """I-3: a finding without evidence refs cannot be constructed."""
        with pytest.raises(ValidationError):
            Finding(text="claim", evidence_refs=(), confidence=Confidence.LOW)


class TestRecommendation:
    def test_basis_default_model_knowledge(self) -> None:
        rec = Recommendation(text="consider VF testing", triggered_by=("cad.p_glaucoma.tier",))
        assert rec.basis == "model_knowledge"

    def test_triggered_by_required(self) -> None:
        with pytest.raises(ValidationError):
            Recommendation(text="do something", triggered_by=())


class TestSubReport:
    def test_round_trip(self) -> None:
        report = SubReport(
            role=Role.OPTOMETRIST,
            findings=(
                Finding(
                    text="tier intermediate",
                    evidence_refs=("cad.p_glaucoma.tier",),
                    confidence=Confidence.MODERATE,
                ),
            ),
            recommendations=(
                Recommendation(text="repeat imaging", triggered_by=("cad.quality.score",)),
            ),
            uncertainties=("no OCT available",),
            declined_out_of_scope=("cannot comment on medication interactions",),
        )
        assert SubReport.model_validate_json(report.model_dump_json()) == report


def _numeric_claim(cid: str = "c1") -> Claim:
    return Claim(
        id=cid,
        kind=ClaimKind.NUMERIC,
        text="vertical CDR is 0.62",
        evidence_refs=("cad.cdr.vertical",),
        source_roles=(Role.OPHTHALMOLOGIST,),
        numeric_value=0.62,
        numeric_ref="cad.cdr.vertical",
    )


class TestClaim:
    def test_numeric_requires_value_and_ref(self) -> None:
        base = {
            "id": "c1",
            "kind": ClaimKind.NUMERIC,
            "text": "n",
            "evidence_refs": ("cad.cdr.vertical",),
            "source_roles": (Role.OPHTHALMOLOGIST,),
        }
        with pytest.raises(ValidationError, match="NUMERIC claim requires"):
            Claim(**base)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="NUMERIC claim requires"):
            Claim(**base, numeric_value=0.62)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="NUMERIC claim requires"):
            Claim(**base, numeric_ref="cad.cdr.vertical")  # type: ignore[arg-type]

    def test_non_numeric_ok_without_value(self) -> None:
        claim = Claim(
            id="c2",
            kind=ClaimKind.INTERPRETIVE,
            text="findings raise suspicion",
            evidence_refs=("cad.p_glaucoma.tier",),
            source_roles=(Role.OPHTHALMOLOGIST, Role.OPTOMETRIST),
        )
        assert claim.numeric_value is None

    def test_empty_evidence_refs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                id="c1",
                kind=ClaimKind.CATEGORICAL,
                text="tier high",
                evidence_refs=(),
                source_roles=(Role.OPHTHALMOLOGIST,),
            )

    def test_empty_source_roles_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                id="c1",
                kind=ClaimKind.CATEGORICAL,
                text="tier high",
                evidence_refs=("cad.p_glaucoma.tier",),
                source_roles=(),
            )


class TestDisagreement:
    def test_requires_two_positions(self) -> None:
        with pytest.raises(ValidationError):
            Disagreement(
                topic="escalation",
                positions=((Role.OPHTHALMOLOGIST, "refer now"),),
                resolution="left to clinician",
            )

    def test_positions_must_be_distinct_roles(self) -> None:
        with pytest.raises(ValidationError, match="distinct roles"):
            Disagreement(
                topic="escalation",
                positions=(
                    (Role.OPHTHALMOLOGIST, "refer now"),
                    (Role.OPHTHALMOLOGIST, "monitor"),
                ),
                resolution="left to clinician",
            )

    def test_valid(self) -> None:
        d = Disagreement(
            topic="referral timing",
            positions=(
                (Role.OPHTHALMOLOGIST, "refer within weeks"),
                (Role.OPTOMETRIST, "routine referral"),
            ),
            resolution="left to clinician",
        )
        assert len(d.positions) == 2


def _draft(claims: tuple[Claim, ...] = (_numeric_claim(),)) -> DirectorDraft:
    return DirectorDraft(
        impression="Findings raise suspicion of glaucomatous change; evaluation warranted.",
        claims=claims,
        limitations=("screening support only", "image quality caveat: none"),
    )


class TestDirectorDraft:
    def test_limitations_mandatory(self) -> None:
        with pytest.raises(ValidationError):
            DirectorDraft(  # type: ignore[call-arg]
                impression="x",
                claims=(_numeric_claim(),),
            )

    def test_duplicate_claim_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            _draft(claims=(_numeric_claim("c1"), _numeric_claim("c1")))

    def test_revision_default_zero(self) -> None:
        assert _draft().revision == 0

    def test_round_trip(self) -> None:
        draft = _draft()
        assert DirectorDraft.model_validate_json(draft.model_dump_json()) == draft


def _issue() -> Issue:
    return Issue(
        type=IssueType.NUMERIC_MISMATCH,
        claim_id="c1",
        detail="claim says 0.82, fact says 0.62",
        evidence=("cad.cdr.vertical",),
        raised_by="deterministic",
    )


class TestIssue:
    def test_round_trip(self) -> None:
        issue = _issue()
        assert Issue.model_validate_json(issue.model_dump_json()) == issue

    def test_claim_id_optional_for_draft_level_issues(self) -> None:
        issue = Issue(
            type=IssueType.OMITTED_FINDING,
            claim_id=None,
            detail="mandatory caveat missing from limitations",
            raised_by="deterministic",
        )
        assert issue.claim_id is None


class TestCriticVerdict:
    def test_approved_with_issues_rejected(self) -> None:
        with pytest.raises(ValidationError, match="approved verdict cannot carry issues"):
            CriticVerdict(approved=True, issues=(_issue(),), round=1)

    def test_rejection_without_issues_rejected(self) -> None:
        with pytest.raises(ValidationError, match="rejection must state at least one issue"):
            CriticVerdict(approved=False, issues=(), round=1)

    def test_approved_clean(self) -> None:
        verdict = CriticVerdict(approved=True, round=2)
        assert verdict.issues == ()

    def test_rejection_with_issues(self) -> None:
        verdict = CriticVerdict(approved=False, issues=(_issue(),), round=1)
        assert len(verdict.issues) == 1

    def test_round_trip(self) -> None:
        verdict = CriticVerdict(approved=False, issues=(_issue(),), round=3)
        assert CriticVerdict.model_validate_json(verdict.model_dump_json()) == verdict
