import { useOverlayLayer } from "../../api/queries";
import type { ReportDocument } from "../../api/types";
import { formatDateTime } from "../../lib/format";

// Renders the server's ReportDocument verbatim — the same document the PDF and
// Word exports are built from. The state banner text comes from the server.
export function ReportDocumentView({ doc }: { doc: ReportDocument }) {
  const overlay = useOverlayLayer(doc.case_id, "composite");

  return (
    <article className={`report-doc report-state-${doc.state}`} data-testid="report-document">
      <header className="report-doc-head">
        <div>
          <p className="eyebrow">CONSILIUM</p>
          <h2>{doc.title}</h2>
        </div>
        <dl className="report-meta">
          <div>
            <dt>Case</dt>
            <dd className="mono">{doc.case_id.slice(0, 8)}</dd>
          </div>
          <div>
            <dt>Eye</dt>
            <dd>{doc.eye}</dd>
          </div>
          <div>
            <dt>Submitted</dt>
            <dd>{formatDateTime(doc.created_at)}</dd>
          </div>
        </dl>
      </header>

      <div className="report-banner" role="status" data-testid="report-state">
        <strong>{doc.state_label}</strong>
        <span>{doc.state_detail}</span>
        {doc.approved_by ? (
          <span className="report-attribution">
            Signed by <b>{doc.approved_by}</b> · {formatDateTime(doc.approved_at)} · version {doc.version}
          </span>
        ) : null}
      </div>

      <section className="report-section report-findings-grid">
        <figure className="report-figure">
          {overlay.url ? (
            <img src={overlay.url} alt="Fundus image with optic disc and cup outlines" />
          ) : (
            <div className="report-figure-empty">Overlay unavailable</div>
          )}
          <figcaption>Optic disc (cyan) and cup (magenta) from the SegFormer masks.</figcaption>
        </figure>
        <div>
          <h3>Image findings</h3>
          <dl className="metric-list" data-testid="image-findings">
            {doc.image_findings.map((m) => (
              <div key={m.label}>
                <dt>{m.label}</dt>
                <dd>{m.value}</dd>
                {m.note ? <p className="muted">{m.note}</p> : null}
              </div>
            ))}
          </dl>
        </div>
      </section>

      {doc.note_facts.length ? (
        <section className="report-section">
          <h3>Clinical note (de-identified)</h3>
          <dl className="metric-list metric-list-inline">
            {doc.note_facts.map((m) => (
              <div key={m.label}>
                <dt>{m.label}</dt>
                <dd className="untrusted">{m.value}</dd>
              </div>
            ))}
          </dl>
        </section>
      ) : null}

      <section className="report-section">
        <h3>Impression</h3>
        <p className="impression">{doc.impression}</p>
      </section>

      <section className="report-section">
        <h3>Findings</h3>
        <ol className="report-findings">
          {doc.findings.map((f) => (
            <li key={f.claim_id} className={f.flagged.length ? "is-flagged" : ""}>
              <p>{f.text}</p>
              <p className="evidence">
                {f.evidence.map((e) => (
                  <span key={e.fact_id} className="chip" title={e.fact_id}>
                    {e.label}: {e.value}
                  </span>
                ))}
                <span className="muted"> — {f.source_roles.join(", ")}</span>
              </p>
              {f.flagged.map((issue) => (
                <p key={issue} className="issue">
                  ⚠ {issue}
                </p>
              ))}
            </li>
          ))}
        </ol>
        {doc.open_issues.map((issue) => (
          <p key={issue} className="issue">
            ⚠ {issue}
          </p>
        ))}
      </section>

      {doc.disagreements.length ? (
        <section className="report-section disagreements">
          <h3>Preserved disagreements</h3>
          {doc.disagreements.map((d) => (
            <div key={d.topic}>
              <strong>{d.topic}</strong>
              <ul>
                {d.positions.map(([role, text]) => (
                  <li key={role + text}>
                    <span className="role-tag">{role}</span> {text}
                  </li>
                ))}
              </ul>
              <p className="muted">Resolution: {d.resolution}</p>
            </div>
          ))}
        </section>
      ) : null}

      <section className="report-section">
        <h3>Recommendations (for clinician judgement)</h3>
        {doc.recommendations.length ? (
          <ul>
            {doc.recommendations.map((r) => (
              <li key={r.text}>
                {r.text} <span className="muted">— triggered by {r.triggered_by.join(", ")}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty">None.</p>
        )}
      </section>

      <section className="report-section">
        <h3>Limitations</h3>
        <ul>
          {doc.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>

      <section className="report-section report-models">
        <h3>Model information</h3>
        <dl className="metric-list metric-list-inline">
          {doc.model_info.map((m) => (
            <div key={m.label}>
              <dt>{m.label}</dt>
              <dd className="mono">{m.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <footer className="report-footer">{doc.footer}</footer>
    </article>
  );
}
