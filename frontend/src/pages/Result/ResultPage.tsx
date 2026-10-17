import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError } from "../../api/client";
import type { ExportFormat } from "../../api/client";
import { useCaseProgress, useLiveTimeline, useReportDocument } from "../../api/queries";
import { useApi, useSession } from "../../auth/session";
import { ProblemMessage } from "../../components/ProblemMessage";
import { ReportDocumentView } from "../../components/Report/ReportDocumentView";
import { StatusBadge } from "../../components/StatusBadge";
import { Timeline } from "../../components/Timeline";
import { shortId } from "../../lib/format";

// Pipeline steps as recorded in the audit timeline (display labels only).
const STEPS: readonly [string, string][] = [
  ["node.precheck_image", "Image checks"],
  ["node.input_rail", "Note safety check"],
  ["node.extract_note", "Clinical note understood (de-identified)"],
  ["node.call_vision", "SwinV2 + SegFormer on the GPU server"],
  ["node.build_context", "Facts assembled"],
  ["node.route_context", "Context routed to each specialist"],
  ["node.run_specialists", "Ophthalmologist · Optometrist · Pharmacist"],
  ["node.draft", "Director drafts the report"],
  ["node.audit", "Verifier + adversarial critic audit"],
  ["node.output_rail", "Output safety check"],
  ["node.persist_pending", "Report ready"],
];

// Plain-language help for the server's failure codes (display only).
const FAILURE_HINTS: Record<string, string> = {
  VISION_UNAVAILABLE:
    "The GPU vision server could not be reached. Check that Deva's server is running and that VISION_ENDPOINT_URL / VISION_ENDPOINT_TOKEN are correct.",
  VISION_REVISION_MISMATCH:
    "The vision server runs different model versions than this project expects. Give Deva the Deva/ folder from this project version.",
  NOT_A_FUNDUS_IMAGE: "This does not look like a retinal fundus photograph.",
  OPTIC_DISC_NOT_VISIBLE: "The optic disc could not be found. Use a disc-centred fundus photograph.",
  UNDEREXPOSED: "The image is too dark to grade.",
  OVEREXPOSED: "The image is too bright to grade.",
  BLURRED: "The image is too blurred to grade.",
  LOW_RESOLUTION: "The image resolution is too low.",
  DEGENERATE_MASK: "The optic disc segmentation was empty for this image.",
  SPECIALIST_FAILED:
    "The specialist AI agents could not run. If the detail mentions LLM_REQUEST_REJECTED, check OPENAI_API_KEY in .env and your OpenAI billing.",
  LLM_REQUEST_REJECTED: "OpenAI refused the request — check OPENAI_API_KEY in .env and your OpenAI billing.",
  LLM_OUTPUT_INVALID: "The AI returned malformed output even after retries. Please try again.",
  LLM_REFUSAL: "The AI model declined to process this case.",
  INTERNAL: "An internal error stopped the pipeline safely (fail closed).",
};

function Progress({ caseId, fast }: { caseId: string; fast: boolean }) {
  const timeline = useLiveTimeline(caseId, fast);
  const done = new Set(timeline.data?.map((e) => e.event) ?? []);
  const current = STEPS.findIndex(([event]) => !done.has(event));
  return (
    <div className="card progress-card" data-testid="progress">
      <h2>
        <span className="spinner" aria-hidden="true" /> Generating the report…
      </h2>
      <p className="muted">This usually takes under a minute. You can leave this page; the case keeps running.</p>
      <ol className="steps">
        {STEPS.map(([event, label], i) => (
          <li key={event} className={done.has(event) ? "is-done" : i === current ? "is-current" : ""}>
            <span className="step-dot" aria-hidden="true" />
            {label}
          </li>
        ))}
      </ol>
    </div>
  );
}

function ExportButtons({ caseId }: { caseId: string }) {
  const api = useApi();
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function download(format: ExportFormat) {
    setBusy(format);
    setError(null);
    try {
      const { blob, filename } = await api.exportReport(caseId, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <button type="button" className="btn btn-primary" disabled={busy !== null} onClick={() => void download("pdf")}>
        {busy === "pdf" ? "Preparing PDF…" : "Download PDF"}
      </button>
      <button type="button" className="btn" disabled={busy !== null} onClick={() => void download("docx")}>
        {busy === "docx" ? "Preparing Word…" : "Download Word (.docx)"}
      </button>
      {error ? <ProblemMessage error={error} context="Export failed" /> : null}
    </>
  );
}

// The screening result: live progress while the pipeline runs, then the report.
export function ResultPage() {
  const { id = "" } = useParams();
  const { isClinician } = useSession();
  const caseQ = useCaseProgress(id);
  const c = caseQ.data;
  const ready = !!c && !c.in_progress && !c.failure;
  const docQ = useReportDocument(id, ready && isClinician);
  const [showAudit, setShowAudit] = useState(false);
  const timeline = useLiveTimeline(id, false);

  if (caseQ.error) return <ProblemMessage error={caseQ.error} context="Could not load the case" />;
  if (!c) return <p>Loading…</p>;

  return (
    <section className="result">
      <p>
        <Link to="/app">← New screening</Link>
      </p>
      <div className="result-head">
        <h1>
          Screening report <span className="mono muted">{shortId(c.case_id)}</span>
        </h1>
        <StatusBadge status={c.status} />
      </div>

      {c.in_progress ? <Progress caseId={id} fast /> : null}

      {c.failure ? (
        <div className="card failure-card" role="alert" data-testid="failure">
          <h2>The report could not be generated</h2>
          <p>
            <code>{c.failure.type}</code>
            {c.failure.reason !== c.failure.type ? (
              <>
                {" "}
                · <span className="muted">{c.failure.reason}</span>
              </>
            ) : null}
          </p>
          <p>{FAILURE_HINTS[c.failure.type] ?? "The pipeline stopped safely (fail closed). No report was produced."}</p>
          <Link to="/app" className="btn btn-primary">
            Start a new screening
          </Link>
        </div>
      ) : null}

      {ready && !isClinician ? (
        <p className="notice">The report is ready. Only clinicians can view and export reports.</p>
      ) : null}

      {ready && isClinician ? (
        <>
          <div className="row actions report-actions">
            {docQ.data ? <ExportButtons caseId={id} /> : null}
            <Link to={`/cases/${id}/review`} className="btn">
              Review &amp; sign
            </Link>
          </div>
          {docQ.error ? (
            docQ.error instanceof ApiError && docQ.error.status === 409 ? (
              <p className="notice">No report is available for this case ({docQ.error.reason}).</p>
            ) : (
              <ProblemMessage error={docQ.error} context="Could not load the report" />
            )
          ) : null}
          {docQ.isLoading ? <p>Loading report…</p> : null}
          {docQ.data ? <ReportDocumentView doc={docQ.data} /> : null}
          <p className="hint">
            A draft becomes final only when a clinician signs it in “Review &amp; sign” — and not the same person who
            submitted the case.
          </p>
        </>
      ) : null}

      <div className="card audit-card">
        <button type="button" className="link-button" onClick={() => setShowAudit((v) => !v)} aria-expanded={showAudit}>
          {showAudit ? "Hide" : "Show"} audit trail
        </button>
        {showAudit && timeline.data ? <Timeline events={timeline.data} /> : null}
      </div>
    </section>
  );
}
