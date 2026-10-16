import { Link, useSearchParams } from "react-router-dom";

import { useCaseList } from "../../api/queries";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDateTime, shortId } from "../../lib/format";
import { QUEUE_FILTERS } from "./filters";

export function CaseListPage() {
  const [params, setParams] = useSearchParams();
  const active = QUEUE_FILTERS.find((f) => f.id === params.get("queue")) ?? QUEUE_FILTERS[0]!;
  const { data, error, isLoading } = useCaseList(active.statuses);

  return (
    <section>
      <h1>Cases</h1>
      <nav className="tabs" aria-label="Queue filters">
        {QUEUE_FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            className={`tab${f.id === active.id ? " tab-active" : ""}`}
            aria-pressed={f.id === active.id}
            onClick={() => setParams({ queue: f.id })}
          >
            {f.label}
          </button>
        ))}
      </nav>
      {error ? <ProblemMessage error={error} context="Could not load cases" /> : null}
      {isLoading ? <p>Loading…</p> : null}
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
                  <Link to={`/cases/${c.case_id}`} className="mono">
                    {shortId(c.case_id)}
                  </Link>
                </td>
                <td>{c.laterality}</td>
                <td>
                  <StatusBadge status={c.status} />
                  {c.unverified ? <span className="tag tag-danger">UNVERIFIED</span> : null}
                </td>
                <td>{c.tier ?? "—"}</td>
                <td>{formatDateTime(c.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}
