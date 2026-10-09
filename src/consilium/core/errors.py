"""Exception hierarchy (AGENTS.md §21).

Every error maps to:
  - a case status from the state machine (AGENTS.md §4), and
  - an RFC 7807 problem type for the API (ARCHITECTURE.md §7).

Rules:
  - ``reason`` is a short, user-safe, actionable machine-ish string
    (e.g. ``VISION_UNAVAILABLE``, ``OPTIC_DISC_NOT_VISIBLE``). It is stored on
    the case and may be shown to users.
  - ``detail`` is for logs/traces only and must NEVER contain note text, image
    data, upload filenames, prompts, or stack-trace-worthy internals (I-6/I-7).
  - Unknown exceptions are converted at the graph-node boundary into
    ``FAILED`` + alert (ARCHITECTURE.md §10) — never caught bare elsewhere.
"""


class AppError(Exception):
    """Base application error. Class attrs define the default mapping."""

    case_status: str = "FAILED"
    problem_type: str = "internal"

    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reason={self.reason!r})"


class InputRejected(AppError):
    """Bad input (image precheck, oversize, wrong format). Fail closed."""

    case_status: str = "REJECTED_INPUT"
    problem_type: str = "input-rejected"


class PolicyViolation(AppError):
    """A guardrail fired (input rail on the raw note, or output rail on a draft).

    Pre-extraction violations reject the input; post-extraction violations send
    the case to NEEDS_ATTENTION (AGENTS.md §14). Pass ``case_status`` to select.
    """

    problem_type: str = "input-rejected"

    def __init__(
        self,
        reason: str,
        *,
        case_status: str = "REJECTED_INPUT",
        detail: str | None = None,
    ) -> None:
        if case_status not in ("REJECTED_INPUT", "NEEDS_ATTENTION"):
            raise ValueError(f"invalid case_status for PolicyViolation: {case_status}")
        super().__init__(reason, detail=detail)
        self.case_status = case_status  # instance-level override of the class default


class VisionFailure(AppError):
    """Vision endpoint/transport/post-processing failure (I-5, I-13).

    Includes: cold-start budget exceeded (VISION_UNAVAILABLE), revision-pin
    mismatch (VISION_REVISION_MISMATCH), malformed payload, non-gradable image.
    Never degrades to a report without vision output.
    """

    case_status: str = "FAILED"
    problem_type: str = "unprocessable"


class AgentFailure(AppError):
    """An LLM agent failed after bounded retries, or refused (I-5)."""

    case_status: str = "FAILED"
    problem_type: str = "unprocessable"


class ValidationFailure(AppError):
    """Schema/structured-output validation failed after repair retries (I-8)."""

    case_status: str = "FAILED"
    problem_type: str = "validation-failed"
