import type { CaseStatus } from "../api/types";

// Display mapping only: the server owns every status and every transition.
type Tone = "progress" | "review" | "attention" | "failed" | "approved" | "rejected";

const TONE: Record<CaseStatus, Tone> = {
  RECEIVED: "progress",
  IMAGE_PRECHECKED: "progress",
  INPUT_RAIL_PASSED: "progress",
  NOTE_EXTRACTED: "progress",
  VISION_CALLED: "progress",
  CAD_COMPLETE: "progress",
  CONTEXT_BUILT: "progress",
  ROUTED: "progress",
  SPECIALISTS_DONE: "progress",
  DRAFTED: "progress",
  AUDITING: "progress",
  REVISING: "progress",
  OUTPUT_RAIL_PASSED: "progress",
  PENDING_REVIEW: "review",
  NEEDS_ATTENTION: "attention",
  FAILED: "failed",
  REJECTED_INPUT: "failed",
  APPROVED: "approved",
  APPROVED_WITH_EDITS: "approved",
  REJECTED: "rejected",
};

export function StatusBadge({ status }: { status: CaseStatus }) {
  return (
    <span className={`badge badge-${TONE[status]}`} data-testid="status-badge">
      {status.replaceAll("_", " ")}
    </span>
  );
}
