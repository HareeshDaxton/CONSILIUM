import type { Claim, EvidenceChip as Chip, Issue } from "../../api/types";
import { EvidenceChip } from "./EvidenceChip";
import { IssueList } from "./IssueList";

export function ClaimBlock({
  claim,
  evidence,
  issues,
}: {
  claim: Claim;
  evidence: Record<string, Chip>;
  issues: Issue[];
}) {
  const flagged = issues.length > 0;
  return (
    <li
      className={`claim${flagged ? " claim-flagged" : ""}`}
      data-testid={`claim-${claim.id}`}
      data-flagged={flagged}
    >
      <div className="claim-head">
        <span className="mono claim-id">{claim.id}</span>
        <span className="claim-kind">{claim.kind}</span>
        {claim.source_roles.map((r) => (
          <span key={r} className="role-tag">
            {r}
          </span>
        ))}
      </div>
      <p className="claim-text">{claim.text}</p>
      {claim.kind === "numeric" && claim.numeric_ref ? (
        <p className="muted claim-numeric">
          value {claim.numeric_value ?? "—"} must equal <span className="mono">{claim.numeric_ref}</span>
        </p>
      ) : null}
      <div className="chips">
        {claim.evidence_refs.map((ref) => (
          <EvidenceChip key={ref} id={ref} chip={evidence[ref]} />
        ))}
      </div>
      <IssueList issues={issues} testId={`issues-${claim.id}`} />
    </li>
  );
}
