"""Deterministic verifier — runs FIRST, before the LLM critic (AGENTS.md §13.1).

Seven pure checks, each ``(draft, ...) -> tuple[Issue, ...]`` with
``raised_by="deterministic"``. A draft reaches the LLM critic only if this
layer returns zero issues. All thresholds/lexicon/topic configuration comes
from ``configs/thresholds.yaml`` — nothing is tuned in code and tolerances
are never loosened to make a draft pass (§26.4).

The checks are deliberately exact and cheap: reference resolution, numeric
equality within per-fact tolerance, stray-numeral scan, mandatory coverage,
forbidden language (via the 7.1 lexicon), disagreement preservation against
configured polar topics, and role-scope (a claim may not rely on facts its
source role was never routed — the third isolation enforcement after the
router and the specialist output validator).
"""

from __future__ import annotations

import re

from consilium.context.facts import F
from consilium.context.router import RoutingConfig, route
from consilium.schemas.agents import DirectorDraft, SubReport
from consilium.schemas.audit import Issue
from consilium.schemas.context import Fact, GlobalContext
from consilium.schemas.enums import ClaimKind, IssueType, Laterality, Role
from consilium.verify.lexicon import STAGE_SEVERITY_PATTERN, LexiconCategory, scan_text
from consilium.vision.config import Thresholds

RAISED_BY = "deterministic"

_NUMERAL = re.compile(r"\b\d+(?:\.\d+)?\b")
_CAVEAT_PREFIX = "cad.caveat."
# Note facts that can supply stage/severity support for the I-12 gate.
_NOTE_STAGE_SOURCES = (F.PRIOR_DX, F.EXAM, F.FREE)


def _issue(
    issue_type: IssueType, claim_id: str | None, detail: str, evidence: tuple[str, ...] = ()
) -> Issue:
    return Issue(
        type=issue_type, claim_id=claim_id, detail=detail, evidence=evidence, raised_by=RAISED_BY
    )


def _fact_map(global_ctx: GlobalContext) -> dict[str, Fact]:
    return {f.id: f for f in global_ctx.facts}


def _numeric_value(fact: Fact) -> float | None:
    if isinstance(fact.value, bool):
        return None
    if isinstance(fact.value, (int, float)):
        return float(fact.value)
    return None


# Representation-noise guard for "exactly at tolerance" boundaries:
# 0.52 - 0.5 == 0.020000000000000004 in binary floating point. 1e-12 absorbs
# representation error only; it never hides a real tolerance violation.
_EPS = 1e-12


def _beyond_tolerance(value: float, reference: float, tolerance: float) -> bool:
    return abs(value - reference) - tolerance > _EPS


def check_references(draft: DirectorDraft, global_ctx: GlobalContext) -> tuple[Issue, ...]:
    """BAD_REFERENCE — every evidence_ref / triggered_by / numeric_ref must
    resolve to a fact ID present in C_global (I-3)."""
    ids = _fact_map(global_ctx)
    issues: list[Issue] = []
    for claim in draft.claims:
        for ref in claim.evidence_refs:
            if ref not in ids:
                issues.append(
                    _issue(
                        IssueType.BAD_REFERENCE,
                        claim.id,
                        f"evidence ref '{ref}' does not resolve to any fact in the context",
                        (ref,),
                    )
                )
        if claim.numeric_ref is not None and claim.numeric_ref not in ids:
            issues.append(
                _issue(
                    IssueType.BAD_REFERENCE,
                    claim.id,
                    f"numeric ref '{claim.numeric_ref}' does not resolve to any fact in"
                    " the context",
                    (claim.numeric_ref,),
                )
            )
    for rec in draft.recommendations:
        for ref in rec.triggered_by:
            if ref not in ids:
                issues.append(
                    _issue(
                        IssueType.BAD_REFERENCE,
                        None,
                        f"recommendation triggers on unknown fact '{ref}'",
                        (ref,),
                    )
                )
    return tuple(issues)


def check_numerics(
    draft: DirectorDraft, global_ctx: GlobalContext, thresholds: Thresholds
) -> tuple[Issue, ...]:
    """NUMERIC_MISMATCH — a NUMERIC claim's value must equal the referenced
    fact within the per-fact tolerance from thresholds.yaml."""
    facts = _fact_map(global_ctx)
    issues: list[Issue] = []
    for claim in draft.claims:
        if claim.kind is not ClaimKind.NUMERIC:
            continue
        # Schema guarantees numeric_value/numeric_ref are set for NUMERIC.
        assert claim.numeric_value is not None and claim.numeric_ref is not None
        fact = facts.get(claim.numeric_ref)
        if fact is None:
            continue  # unresolved refs are BAD_REFERENCE, not a numeric issue
        fact_value = _numeric_value(fact)
        if fact_value is None:
            issues.append(
                _issue(
                    IssueType.NUMERIC_MISMATCH,
                    claim.id,
                    f"numeric ref '{claim.numeric_ref}' points at a non-numeric fact",
                    (claim.numeric_ref,),
                )
            )
            continue
        tolerance = thresholds.numeric_tolerances.tolerance_for(claim.numeric_ref)
        if _beyond_tolerance(claim.numeric_value, fact_value, tolerance):
            issues.append(
                _issue(
                    IssueType.NUMERIC_MISMATCH,
                    claim.id,
                    f"claim value {claim.numeric_value} differs from fact '{claim.numeric_ref}'"
                    f" value {fact_value} by more than tolerance {tolerance}",
                    (claim.numeric_ref,),
                )
            )
    return tuple(issues)


def check_stray_numbers(
    draft: DirectorDraft, global_ctx: GlobalContext, thresholds: Thresholds
) -> tuple[Issue, ...]:
    """UNSUPPORTED — numerals in impression/claim/recommendation prose that are
    neither whitelisted (dates, follow-up intervals, ages) nor attributable to
    a cited fact value or a NUMERIC claim."""
    facts = _fact_map(global_ctx)
    whitelist = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            thresholds.stray_number_whitelist.date,
            thresholds.stray_number_whitelist.follow_up_interval,
            thresholds.stray_number_whitelist.age,
        )
    )
    tolerances = thresholds.numeric_tolerances

    def attributed(value: float, ref_ids: tuple[str, ...]) -> bool:
        for ref in ref_ids:
            fact = facts.get(ref)
            if fact is None:
                continue
            fact_value = _numeric_value(fact)
            if fact_value is not None and not _beyond_tolerance(
                value, fact_value, tolerances.tolerance_for(ref)
            ):
                return True
        for claim in draft.claims:
            if (
                claim.kind is ClaimKind.NUMERIC
                and claim.numeric_value is not None
                and not _beyond_tolerance(value, claim.numeric_value, tolerances.default)
            ):
                return True
        return False

    scopes: list[tuple[str | None, str, tuple[str, ...]]] = []
    all_refs = tuple(dict.fromkeys(ref for c in draft.claims for ref in c.evidence_refs))
    scopes.append((None, draft.impression, all_refs))
    for claim in draft.claims:
        scopes.append((claim.id, claim.text, claim.evidence_refs))
    for rec in draft.recommendations:
        scopes.append((None, rec.text, rec.triggered_by))

    issues: list[Issue] = []
    for claim_id, text, ref_ids in scopes:
        stripped = text
        for pattern in whitelist:
            stripped = pattern.sub(" ", stripped)
        for match in _NUMERAL.finditer(stripped):
            value = float(match.group())
            if not attributed(value, ref_ids):
                issues.append(
                    _issue(
                        IssueType.UNSUPPORTED,
                        claim_id,
                        f"numeral '{match.group()}' is not attributable to any cited fact"
                        " or numeric claim",
                    )
                )
    return tuple(issues)


def check_coverage(draft: DirectorDraft, global_ctx: GlobalContext) -> tuple[Issue, ...]:
    """OMITTED_FINDING — laterality, screening tier and vertical CDR must be
    cited by some claim; every caveat fact must be cited or covered by a
    limitation mentioning it (§7 contract rules)."""
    cited = {ref for claim in draft.claims for ref in claim.evidence_refs}
    issues: list[Issue] = []
    for required in (F.LAT, F.P_TIER, F.CDR_V):
        if required not in cited:
            issues.append(
                _issue(
                    IssueType.OMITTED_FINDING,
                    None,
                    f"mandatory fact '{required}' is not cited by any claim",
                    (required,),
                )
            )
    limitations_text = " ".join(draft.limitations).lower()
    for fact in global_ctx.facts:
        if not (fact.id.startswith(_CAVEAT_PREFIX) or fact.id == F.CAVEAT_EXTRACTION):
            continue
        if fact.id in cited:
            continue
        token = fact.id.rsplit(".", 1)[-1].replace("_", " ")
        if token not in limitations_text:
            issues.append(
                _issue(
                    IssueType.OMITTED_FINDING,
                    None,
                    f"caveat '{fact.id}' is carried by neither a claim nor any limitation",
                    (fact.id,),
                )
            )
    return tuple(issues)


def check_language(draft: DirectorDraft, global_ctx: GlobalContext) -> tuple[Issue, ...]:
    """UNSAFE_LANGUAGE — forbidden phrases from verify/lexicon.py. Stage/
    severity wording is allowed only when the clinical note itself supplies it
    (I-12)."""
    facts = _fact_map(global_ctx)

    def _note_supplies_stage(ref: str) -> bool:
        value = facts[ref].value
        return isinstance(value, str) and STAGE_SEVERITY_PATTERN.search(value) is not None

    note_supports_stage = any(
        _note_supplies_stage(ref) for ref in _NOTE_STAGE_SOURCES if ref in facts
    )
    scopes: list[tuple[str | None, str]] = [(None, draft.impression)]
    scopes.extend((claim.id, claim.text) for claim in draft.claims)
    scopes.extend((None, rec.text) for rec in draft.recommendations)
    issues: list[Issue] = []
    for claim_id, text in scopes:
        for hit in scan_text(text):
            if hit.category is LexiconCategory.STAGE_SEVERITY and note_supports_stage:
                continue
            issues.append(
                _issue(
                    IssueType.UNSAFE_LANGUAGE,
                    claim_id,
                    f"forbidden {hit.category.value} wording: '{hit.matched}'",
                )
            )
    return tuple(issues)


def check_disagreement_preservation(
    draft: DirectorDraft, sub_reports: tuple[SubReport, ...], thresholds: Thresholds
) -> tuple[Issue, ...]:
    """DISAGREEMENT_SUPPRESSED — if two roles take polar stances on a
    configured topic, the draft must carry a visible Disagreement for it
    (conservative: flags for review, never resolves — §13.1)."""
    issues: list[Issue] = []
    for topic_cfg in thresholds.disagreement_topics:
        keywords = tuple(k.lower() for k in topic_cfg.keywords)
        stance_a = tuple(m.lower() for m in topic_cfg.stance_a)
        stance_b = tuple(m.lower() for m in topic_cfg.stance_b)
        roles_a: set[Role] = set()
        roles_b: set[Role] = set()
        for report in sub_reports:
            text = " ".join(
                [f.text for f in report.findings] + [r.text for r in report.recommendations]
            ).lower()
            if not any(k in text for k in keywords):
                continue
            if any(m in text for m in stance_a):
                roles_a.add(report.role)
            if any(m in text for m in stance_b):
                roles_b.add(report.role)
        polar = any(ra is not rb for ra in roles_a for rb in roles_b)
        if not polar:
            continue
        covered = any(
            any(k in d.topic.lower() for k in keywords)
            or any(k in position.lower() for _, position in d.positions for k in keywords)
            for d in draft.disagreements
        )
        if not covered:
            roles = sorted(r.value for r in roles_a | roles_b)
            issues.append(
                _issue(
                    IssueType.DISAGREEMENT_SUPPRESSED,
                    None,
                    f"roles {roles} take polar stances on '{topic_cfg.topic}' but the draft"
                    " carries no Disagreement for it",
                )
            )
    return tuple(issues)


def check_laterality_consistency(
    draft: DirectorDraft, global_ctx: GlobalContext
) -> tuple[Issue, ...]:
    """UNSUPPORTED — a claim citing ``cad.laterality`` must not state a
    conflicting laterality token (catches the §13.4 laterality-flip fault).
    Exempt when the fact value is UNKNOWN (no token to conflict with)."""
    facts = _fact_map(global_ctx)
    fact = facts.get(F.LAT)
    if fact is None or not isinstance(fact.value, str) or fact.value == Laterality.UNKNOWN:
        return ()
    tokens = {
        Laterality.OD: ("od", "right eye"),
        Laterality.OS: ("os", "left eye"),
    }
    expected = tokens.get(Laterality(fact.value), ())
    if not expected:
        return ()
    conflicting = tuple(t for pair in tokens.values() for t in pair if t not in expected)
    issues: list[Issue] = []
    for claim in draft.claims:
        if F.LAT not in claim.evidence_refs:
            continue
        text = claim.text.lower()
        hit = next((t for t in conflicting if re.search(rf"\b{re.escape(t)}\b", text)), None)
        if hit is not None:
            issues.append(
                _issue(
                    IssueType.UNSUPPORTED,
                    claim.id,
                    f"claim cites {F.LAT} (value '{fact.value}') but states conflicting"
                    f" laterality token '{hit}'",
                    (F.LAT,),
                )
            )
    return tuple(issues)


def check_role_scope(
    draft: DirectorDraft, global_ctx: GlobalContext, routing: RoutingConfig
) -> tuple[Issue, ...]:
    """OUT_OF_SCOPE — a claim sourced from a role may not rely on refs outside
    that role's allowlist (§13.1). Allowed sets are derived by re-running the
    deterministic router, so the check can never drift from the policy."""
    allowed = {role: ctx.allowed_ids() for role, ctx in route(global_ctx, routing).items()}
    issues: list[Issue] = []
    for claim in draft.claims:
        for role in claim.source_roles:
            role_allowed = allowed.get(role, frozenset())
            outside = tuple(ref for ref in claim.evidence_refs if ref not in role_allowed)
            if outside:
                issues.append(
                    _issue(
                        IssueType.OUT_OF_SCOPE,
                        claim.id,
                        f"role '{role.value}' cannot support refs outside its routing"
                        f" allowlist: {list(outside)}",
                        outside,
                    )
                )
    return tuple(issues)


def verify_draft(
    draft: DirectorDraft,
    global_ctx: GlobalContext,
    sub_reports: tuple[SubReport, ...],
    thresholds: Thresholds,
    routing: RoutingConfig,
) -> tuple[Issue, ...]:
    """Run every deterministic check and return all issues (empty = pass).

    This is the complete FIRST layer of verification (§13.3); only a draft
    with zero issues here reaches the LLM critic (P8).
    """
    return (
        *check_references(draft, global_ctx),
        *check_numerics(draft, global_ctx, thresholds),
        *check_stray_numbers(draft, global_ctx, thresholds),
        *check_coverage(draft, global_ctx),
        *check_language(draft, global_ctx),
        *check_disagreement_preservation(draft, sub_reports, thresholds),
        *check_laterality_consistency(draft, global_ctx),
        *check_role_scope(draft, global_ctx, routing),
    )
