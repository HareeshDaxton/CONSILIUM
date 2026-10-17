import { Link, useSearchParams } from "react-router-dom";

import { useCaseList, useQueueCount } from "../../api/queries";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDateTime, shortId } from "../../lib/format";
import { QUEUE_FILTERS } from "./filters";
import type { QueueFilter } from "./filters";

const COUNT_LIMIT = 200;

function QueueCard({ filter, active, onSelect }: { filter: QueueFilter; active: boolean; onSelect: () => void }) {
  const { data } = useQueueCount(filter.statuses, COUNT_LIMIT);
  const count = data === undefined ? "–" : data >= COUNT_LIMIT ? `${COUNT_LIMIT}+` : String(data);
  return (
    <button
      type="button"
      className={`queue-card queue-${filter.id}${active ? " is-active" : ""}`}
      aria-label={filter.label}
      aria-pressed={active}
      onClick={onSelect}
    >
      <span className="queue-label">{filter.label}</span>
      <span className="queue-count" data-testid={`count-${filter.id}`}>
        {count}
      </span>
    </button>
  );
}

// Main page: the clinician worklist. Queue cards are server-side status
// filters (counts = rows the server returns for that filter); the UI never
// decides what a status means.
export function CaseListPage() {
  const [params, setParams] = useSearchParams();
  const active = QUEUE_FILTERS.find((f) => f.id === params.get("queue")) ?? QUEUE_FILTERS[0]!;
  const { data, error, isLoading } = useCaseList(active.statuses);

  return (
    <section className="worklist">
      <header className="page-head">
        <div>
          <p className="eyebrow">Cases</p>
          <h1>All screenings</h1>
          <p className="muted">Every submitted case, refreshed every 10 seconds. Open one to see its report.</p>
        </div>
      </header>

      <nav className="queue-grid" aria-label="Queue filters">
        {QUEUE_FILTERS.map((f) => (
          <QueueCard key={f.id} filter={f} active={f.id === active.id} onSelect={() => setParams({ queue: f.id })} />
        ))}
      </nav>

      <div className="card table-card">
        <div className="table-head">
          <h2>{active.label}</h2>
          {data ? <span className="muted">{data.items.length} shown</span> : null}
        </div>
        {error ? <ProblemMessage error={error} context="Could not load cases" /> : null}
        {isLoading ? <p className="empty">Loading…</p> : null}
        {data && data.items.length === 0 ? <p className="empty">No cases in this view.</p> : null}
        {data && data.items.length > 0 ? (
          <table className="table">
            <thead>
              <tr>
                <th>Case</th>
                <th>Eye</th>
                <th>Status</th>
                <th>Screening tier</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((c) => (
                <tr key={c.case_id} data-testid="case-row">
                  <td>
                    <Link to={`/cases/${c.case_id}`} className="mono case-link">
                      {shortId(c.case_id)}
                    </Link>
                  </td>
                  <td>{c.laterality}</td>
                  <td>
                    <StatusBadge status={c.status} />
                    {c.unverified ? <span className="tag tag-danger">UNVERIFIED</span> : null}
                  </td>
                  <td>{c.tier ? <span className={`tier tier-${c.tier}`}>{c.tier}</span> : "—"}</td>
                  <td className="muted">{formatDateTime(c.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </section>
  );
}
