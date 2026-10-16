import type { CaseStatus } from "../../api/types";

// Queue views = server-side status filters (GET /v1/cases?status=…). Pure
// read filters: the UI never changes a status.
export interface QueueFilter {
  id: string;
  label: string;
  statuses: readonly CaseStatus[];
}

export const QUEUE_FILTERS: readonly QueueFilter[] = [
  { id: "review", label: "Review queue", statuses: ["PENDING_REVIEW", "NEEDS_ATTENTION"] },
  { id: "attention", label: "Needs attention", statuses: ["NEEDS_ATTENTION"] },
  { id: "failed", label: "Failed", statuses: ["FAILED", "REJECTED_INPUT"] },
  { id: "approved", label: "Approved", statuses: ["APPROVED", "APPROVED_WITH_EDITS"] },
  { id: "rejected", label: "Rejected", statuses: ["REJECTED"] },
  { id: "all", label: "All", statuses: [] },
];
