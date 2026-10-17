// Wire types mirroring the API view models (src/consilium/api/schemas_http.py,
// review/service.py) and the contract models they embed (schemas/). Hand-kept
// in sync; tests/contract/test_frontend_fixtures.py validates the UI fixtures
// against the real Pydantic models so drift fails CI.
//
// The UI only DISPLAYS these values. It never computes or sets a status.

export type CaseStatus =
  | "RECEIVED"
  | "IMAGE_PRECHECKED"
  | "REJECTED_INPUT"
  | "INPUT_RAIL_PASSED"
  | "NOTE_EXTRACTED"
  | "VISION_CALLED"
  | "CAD_COMPLETE"
  | "CONTEXT_BUILT"
  | "ROUTED"
  | "SPECIALISTS_DONE"
  | "DRAFTED"
  | "AUDITING"
  | "REVISING"
  | "OUTPUT_RAIL_PASSED"
  | "PENDING_REVIEW"
  | "APPROVED"
  | "APPROVED_WITH_EDITS"
  | "REJECTED"
  | "FAILED"
  | "NEEDS_ATTENTION";

export type Laterality = "OD" | "OS" | "UNKNOWN";
export type Role = "ophthalmologist" | "optometrist" | "pharmacist";
export type ClaimKind = "numeric" | "categorical" | "interpretive" | "recommendation";
export type Confidence = "low" | "moderate" | "high";
export type FactSource = "cad" | "note" | "derived";
export type IssueType =
  | "unsupported"
  | "numeric_mismatch"
  | "omitted_finding"
  | "out_of_scope"
  | "unsafe_language"
  | "disagreement_suppressed"
  | "bad_reference";

export type FactValue = string | number | boolean | string[];

export interface Me {
  sub: string;
  roles: string[];
  // Display-only profile, present after Google sign-in (never used for authz).
  email?: string | null;
  name?: string | null;
  picture?: string | null;
}

export interface AuthConfig {
  google_client_id: string | null;
}

export interface SessionView {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: Me;
}

export interface SubmitResponse {
  case_id: string;
  status: CaseStatus;
  status_url: string;
}

export interface FailureView {
  type: string;
  reason: string;
}

export interface CadSummary {
  tier: string;
  p_calibrated: number;
  cdr_vertical: number;
  quality_flags: string[];
  mask_qc_ok: boolean;
}

export interface CaseView {
  case_id: string;
  status: CaseStatus;
  laterality: Laterality;
  created_at: string | null;
  failure: FailureView | null;
  cad_summary: CadSummary | null;
  unverified: boolean;
  /** Server-decided: the pipeline still has steps to run. */
  in_progress: boolean;
}

export interface CaseListItem {
  case_id: string;
  status: CaseStatus;
  laterality: Laterality;
  created_at: string | null;
  tier: string | null;
  unverified: boolean;
}

export interface CaseListView {
  items: CaseListItem[];
}

export interface TimelineEvent {
  event: string;
  actor: string;
  at: string;
}

export interface FactView {
  id: string;
  label: string;
  value: FactValue;
  unit: string | null;
  source: FactSource;
}

// ---- DirectorDraft (schemas/agents.py) ----

export interface Recommendation {
  text: string;
  triggered_by: string[];
  basis: string;
}

export interface Claim {
  id: string;
  kind: ClaimKind;
  text: string;
  evidence_refs: string[];
  source_roles: Role[];
  numeric_value: number | null;
  numeric_ref: string | null;
}

export interface Disagreement {
  topic: string;
  positions: [Role, string][];
  resolution: string;
}

export interface DirectorDraft {
  impression: string;
  claims: Claim[];
  disagreements: Disagreement[];
  recommendations: Recommendation[];
  limitations: string[];
  revision: number;
}

export interface Issue {
  type: IssueType;
  claim_id: string | null;
  detail: string;
  evidence: string[];
  raised_by: string;
}

export interface EvidenceChip {
  label: string;
  value: FactValue;
  unit: string | null;
}

export interface DraftView {
  case_id: string;
  case_status: CaseStatus;
  revision: number;
  draft: DirectorDraft;
  open_issues: Issue[];
  unverified: boolean;
  unverified_banner: string | null;
  overlay_url: string;
  evidence: Record<string, EvidenceChip>;
}

// ---- review (review/service.py) ----

export type ReviewAction = "APPROVE" | "APPROVE_WITH_EDITS" | "REJECT";

export interface ReasonCodeOption {
  code: string;
  description: string;
}

export interface ReviewCommand {
  action: ReviewAction;
  base_revision: number;
  edited_draft?: DirectorDraft;
  reason_codes?: string[];
  comment?: string;
}

export interface ReviewResult {
  review_id: string;
  case_id: string;
  action: ReviewAction;
  status: CaseStatus;
  version: number | null;
  amends_review_id: string | null;
}

export interface ReportView {
  case_id: string;
  status: CaseStatus;
  laterality: Laterality;
  version: number;
  review_id: string;
  amends_review_id: string | null;
  approved_by: string;
  approved_at: string;
  action: ReviewAction;
  draft_revision: number;
  reason_codes: string[];
  report: DirectorDraft;
  screening_notice: string;
}

// RFC 7807 problem+json as rendered by api/errors.py.
export interface Problem {
  type: string;
  title: string;
  status: number;
  reason?: string;
  case_id?: string;
  errors?: { loc: string[]; type: string }[];
}

// ---- system status + screening report (reporting/document.py) ----

export interface SystemStatus {
  vision: { mode: string; reachable: boolean; revision_ok: boolean; device: string | null; detail: string | null };
  llm_configured: boolean;
}

export type ReportState = "approved" | "draft" | "unverified" | "rejected";

export interface MetricRow {
  label: string;
  value: string;
  note: string | null;
}

export interface ReportDocument {
  title: string;
  case_id: string;
  eye: string;
  created_at: string | null;
  generated_at: string;
  state: ReportState;
  state_label: string;
  state_detail: string;
  approved_by: string | null;
  approved_at: string | null;
  version: number | null;
  reason_codes: string[];
  draft_revision: number;
  image_findings: MetricRow[];
  note_facts: MetricRow[];
  impression: string;
  findings: {
    claim_id: string;
    text: string;
    evidence: { fact_id: string; label: string; value: string }[];
    source_roles: string[];
    flagged: string[];
  }[];
  disagreements: { topic: string; positions: [string, string][]; resolution: string }[];
  recommendations: { text: string; triggered_by: string[] }[];
  limitations: string[];
  open_issues: string[];
  model_info: MetricRow[];
  footer: string;
}
