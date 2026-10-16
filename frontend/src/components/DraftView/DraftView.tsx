import type { DraftView as DraftViewData, Issue } from "../../api/types";
import { ClaimBlock } from "./ClaimBlock";
import { DisagreementList } from "./DisagreementList";
import { EvidenceChip } from "./EvidenceChip";
import { IssueList } from "./IssueList";
import { UnverifiedBanner } from "./UnverifiedBanner";

// Read-only rendering of the AI draft exactly as served (ARCHITECTURE.md §5.1).
// Open critic issues are keyed to their claim by `claim_id`; issues without a
// claim are listed at draft level. All text is rendered as plain text.
export function DraftView({ view }: { view: DraftViewData }) {
  const { draft, evidence } = view;
  const byClaim = new Map<string, Issue[]>();
  const draftLevel: Issue[] = [];
  const claimIds = new Set(draft.claims.map((c) => c.id));
  for (const issue of view.open_issues) {
    if (issue.claim_id && claimIds.has(issue.claim_id)) {
      byClaim.set(issue.claim_id, [...(byClaim.get(issue.claim_id) ?? []), issue]);
    } else {
      draftLevel.push(issue);
    }
  }

  return (
    <article className="draft" data-testid="draft-view">
      {view.unverified_banner ? (
        <UnverifiedBanner text={view.unverified_banner} openIssues={view.open_issues.length} />
      ) : null}
      <p className="muted">AI draft · revision {view.revision}</p>

      <section>
        <h3>Impression</h3>
        <p className="impression">{draft.impression}</p>
      </section>

      {draftLevel.length ? (
        <section className="draft-issues">
          <h3>Open issues (whole draft)</h3>
          <IssueList issues={draftLevel} testId="issues-draft" />
        </section>
      ) : null}

      <section>
        <h3>Claims</h3>
        <ol className="claims">
          {draft.claims.map((claim) => (
            <ClaimBlock
              key={claim.id}
              claim={claim}
              evidence={evidence}
              issues={byClaim.get(claim.id) ?? []}
            />
          ))}
        </ol>
      </section>

      <DisagreementList items={draft.disagreements} />

      <section>
        <h3>Recommendations</h3>
        {draft.recommendations.length ? (
          <ul className="recommendations">
            {draft.recommendations.map((rec, i) => (
              <li key={i} data-testid="recommendation">
                <p>{rec.text}</p>
                <div className="chips">
                  {rec.triggered_by.map((ref) => (
                    <EvidenceChip key={ref} id={ref} chip={evidence[ref]} />
                  ))}
                  <span className="muted"> basis: {rec.basis} — for clinician judgement</span>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty">None.</p>
        )}
      </section>

      <section>
        <h3>Limitations</h3>
        <ul className="limitations" data-testid="limitations">
          {draft.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      </section>
    </article>
  );
}
