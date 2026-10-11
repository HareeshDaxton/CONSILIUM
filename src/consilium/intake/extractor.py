"""Structured note extraction via LLMClient (AGENTS.md §10 steps 3-4, I-7/I-8).

The extractor receives the REDACTED note only — de-identification always runs
first (fixed order, SKILLS.md note-extraction). The note travels inside an
``<untrusted-clinical-note>`` data block that the versioned prompt declares
non-instructional; extracted fields are data, not instructions.

Malformed/schema-violating output gets ≤2 repair retries with the validation
error fed back (``generate_structured``); then ValidationFailure → FAILED.

Extraction confidence is a DETERMINISTIC heuristic (fraction of informative
fields populated) — LLM self-confidence is not trusted (§10.5). LOW
confidence makes the Context Builder (P4) emit ``note.caveat.extraction``.
"""

from pathlib import Path

from consilium.core.settings import DEFAULT_MODEL_SNAPSHOT
from consilium.llm.budget import TokenBudget
from consilium.llm.client import ChatMessage, LLMClient, ParsedResponse
from consilium.llm.prompt_loader import PromptLoader
from consilium.llm.retry import LLMRetryPolicy, generate_structured
from consilium.schemas.enums import AgentName, Confidence
from consilium.schemas.note import ClinicalNote

PROMPT_VERSION = "v1"
PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "note_extractor" / f"{PROMPT_VERSION}.md"
)

# CLINICAL-REVIEW: deterministic confidence heuristic (§10.5). Counts
# informative fields the extraction populated; LLM self-reports are not
# trusted. Thresholds are a first proposal for the clinical co-author.
_HIGH_CONFIDENCE_MIN_FIELDS = 4

_MAX_OUTPUT_TOKENS = 2000  # ClinicalNote is small; caps runaway generations


def load_extractor_prompt(path: Path = PROMPT_PATH) -> str:
    """Load the versioned extractor prompt (immutable once released — I-9)."""
    # Delegates to the shared loader (Section 5.2); ``path`` stays injectable
    # for tests.
    if path == PROMPT_PATH:
        return PromptLoader().load("note_extractor", PROMPT_VERSION).content
    return path.read_text(encoding="utf-8")


def build_messages(prompt: str, redacted_note: str) -> list[ChatMessage]:
    """System prompt + the note inside an explicitly non-instructional block."""
    return [
        ChatMessage(role="system", content=prompt),
        ChatMessage(
            role="user",
            content=(
                "Extract structured data from the clinical note below.\n\n"
                "<untrusted-clinical-note>\n"
                f"{redacted_note}\n"
                "</untrusted-clinical-note>"
            ),
        ),
    ]


async def extract_note(
    redacted_note: str,
    *,
    llm: LLMClient,
    prompt: str,
    model: str = DEFAULT_MODEL_SNAPSHOT,
    temperature: float = 0.0,  # extraction must be as deterministic as possible
    timeout_s: float = 60.0,
    max_repairs: int = 2,
    retry_policy: LLMRetryPolicy | None = None,
    budget: TokenBudget | None = None,
    fixture: str | None = None,
) -> ParsedResponse[ClinicalNote]:
    """One extraction call with transient retry + bounded repair (I-5/I-8)."""
    tags = {
        "agent": AgentName.NOTE_EXTRACTOR.value,
        "prompt_version": PROMPT_VERSION,
    }
    if fixture is not None:
        tags["fixture"] = fixture
    return await generate_structured(
        llm,
        build_messages(prompt, redacted_note),
        ClinicalNote,
        model=model,
        temperature=temperature,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        timeout_s=timeout_s,
        tags=tags,
        max_repairs=max_repairs,
        retry_policy=retry_policy,
        budget=budget,
    )


def estimate_confidence(note: ClinicalNote) -> Confidence:
    """Deterministic extraction-confidence heuristic.

    Counts populated informative fields (free_text carries no signal — it is
    a copy of the note, not an extraction result).
    """
    populated = sum(
        [
            note.age_years is not None,
            note.iop_mmhg_od is not None,
            note.iop_mmhg_os is not None,
            bool(note.current_medications),
            bool(note.drug_intolerances_or_allergies),
            bool(note.comorbidities),
            note.prior_glaucoma_dx is not None,
            note.exam_findings is not None,
        ]
    )
    if populated >= _HIGH_CONFIDENCE_MIN_FIELDS:
        return Confidence.HIGH
    if populated >= 1:
        return Confidence.MODERATE
    return Confidence.LOW
