import { Link, useParams } from "react-router-dom";

import { useCase, useFacts, useTimeline } from "../../api/queries";
import { useSession } from "../../auth/session";
import { CadSummaryCard } from "../../components/CadSummaryCard";
import { FactsTable } from "../../components/FactsTable";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { Timeline } from "../../components/Timeline";
import { formatDateTime, shortId } from "../../lib/format";

// Case detail. Review/report links are offered to every clinician; whether a
// draft or approved report exists is answered by the server (409 otherwise) —
// the UI does not infer it from the status.
export function CaseDetailPage() {
  const { id = "" } = useParams();
  const { isClinician } = useSession();
  const caseQ = useCase(id);
  const timelineQ = useTimeline(id);
  const factsQ = useFacts(id, isClinician);

  if (caseQ.error) return <ProblemMessage error={caseQ.error} context="Could not load case" />;
  const c = caseQ.data;
  if (!c) return <p>Loading…</p>;

  return (
    <section>
      <p>
        <Link to="/cases">← Cases</Link>
      </p>
      <h1>
        Case <span className="mono">{shortId(c.case_id)}</span> · {c.laterality}
      </h1>
      <div className="row">
        <StatusBadge status={c.status} />
        {c.unverified ? <span className="tag tag-danger">UNVERIFIED — critic did not approve</span> : null}
        <span className="muted">created {formatDateTime(c.created_at)}</span>
      </div>
      {c.failure ? (
        <div className="problem" role="alert" data-testid="failure">
          <strong>{c.failure.type}</strong> · <code>{c.failure.reason}</code>
        </div>
      ) : null}
      {isClinician ? (
        <div className="row actions">
          <Link className="btn btn-primary" to={`/cases/${c.case_id}/review`}>
            Open review
          </Link>
          <Link className="btn" to={`/cases/${c.case_id}/report`}>
            Approved report
          </Link>
        </div>
      ) : null}
      <div className="grid-2">
        <div>
          {c.cad_summary ? (
            <CadSummaryCard cad={c.cad_summary} />
          ) : (
            <div className="card">
              <h2>CAD summary</h2>
              <p className="empty">Not available yet.</p>
            </div>
          )}
          {isClinician ? (
            <div className="card">
              <h2>Note facts (de-identified)</h2>
              {factsQ.error ? <ProblemMessage error={factsQ.error} /> : null}
              {factsQ.data ? <FactsTable facts={factsQ.data} source="note" /> : null}
            </div>
          ) : null}
        </div>
        <div className="card">
          <h2>Timeline</h2>
          {timelineQ.error ? <ProblemMessage error={timelineQ.error} /> : null}
          {timelineQ.data ? <Timeline events={timelineQ.data} /> : null}
        </div>
      </div>
    </section>
  );
}
