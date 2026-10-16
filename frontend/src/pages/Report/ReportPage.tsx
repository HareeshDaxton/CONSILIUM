import { Link, useParams } from "react-router-dom";

import { useReport } from "../../api/queries";
import { DisagreementList } from "../../components/DraftView/DisagreementList";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDateTime, shortId } from "../../lib/format";

// Approved report (GET /v1/cases/{id}/report). Served only for APPROVED*
// cases — otherwise the server answers 409 REPORT_NOT_APPROVED, shown as-is.
// Always rendered with clinician attribution and the screening notice.
export function ReportPage() {
  const { id = "" } = useParams();
  const { data, error, isLoading } = useReport(id);

  return (
    <section>
      <p>
        <Link to={`/cases/${id}`}>← Case detail</Link>
      </p>
      <h1>
        Approved report · case <span className="mono">{shortId(id)}</span>
      </h1>
      {error ? <ProblemMessage error={error} context="Report unavailable" /> : null}
      {isLoading ? <p>Loading…</p> : null}
      {data ? (
        <article className="card report" data-testid="report">
          <p className="screening-notice" data-testid="screening-notice">
            {data.screening_notice}
          </p>
          <div className="row">
            <StatusBadge status={data.status} />
            <span>Eye: {data.laterality}</span>
            <span>Version {data.version}</span>
          </div>
          <dl className="kv" data-testid="attribution">
            <dt>Approved by</dt>
            <dd>{data.approved_by}</dd>
            <dt>Approved at</dt>
            <dd>{formatDateTime(data.approved_at)}</dd>
            <dt>Action</dt>
            <dd>{data.action}</dd>
            <dt>Based on AI draft</dt>
            <dd>revision {data.draft_revision}</dd>
            {data.reason_codes.length ? (
              <>
                <dt>Reason codes</dt>
                <dd>{data.reason_codes.join(", ")}</dd>
              </>
            ) : null}
            {data.amends_review_id ? (
              <>
                <dt>Amends</dt>
                <dd className="mono">{data.amends_review_id}</dd>
              </>
            ) : null}
          </dl>
          <h3>Impression</h3>
          <p className="impression">{data.report.impression}</p>
          <h3>Findings</h3>
          <ul>
            {data.report.claims.map((c) => (
              <li key={c.id}>{c.text}</li>
            ))}
          </ul>
          <DisagreementList items={data.report.disagreements} />
          {data.report.recommendations.length ? (
            <>
              <h3>Recommendations (for clinician judgement)</h3>
              <ul>
                {data.report.recommendations.map((r, i) => (
                  <li key={i}>{r.text}</li>
                ))}
              </ul>
            </>
          ) : null}
          <h3>Limitations</h3>
          <ul>
            {data.report.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </article>
      ) : null}
    </section>
  );
}
