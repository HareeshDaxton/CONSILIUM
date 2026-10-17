import { Link, useNavigate, useParams } from "react-router-dom";

import { useCase, useDraft } from "../../api/queries";
import type { ReviewResult } from "../../api/types";
import { DraftView } from "../../components/DraftView/DraftView";
import { EditDiffForm } from "../../components/EditDiffForm/EditDiffForm";
import { MaskOverlayViewer } from "../../components/MaskOverlayViewer/MaskOverlayViewer";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { shortId } from "../../lib/format";

// Review screen (clinician only; the API enforces it). Whether a draft exists
// is the server's answer (409 DRAFT_NOT_AVAILABLE), never inferred from status.
export function ReviewScreen() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const caseQ = useCase(id);
  const draftQ = useDraft(id);

  // Back to the case's report page (it shows whatever the server now says:
  // the signed report with export, or the rejection).
  function onSubmitted(_result: ReviewResult) {
    navigate(`/cases/${id}`);
  }

  return (
    <section className="review">
      <p>
        <Link to={`/cases/${id}`}>← Case detail</Link>
      </p>
      <h1>
        Review · case <span className="mono">{shortId(id)}</span>
        {caseQ.data ? ` · ${caseQ.data.laterality}` : ""}{" "}
        {caseQ.data ? <StatusBadge status={caseQ.data.status} /> : null}
      </h1>
      {draftQ.error ? <ProblemMessage error={draftQ.error} context="Draft unavailable" /> : null}
      {draftQ.isLoading ? <p>Loading draft…</p> : null}
      {draftQ.data ? (
        <>
          <div className="review-layout">
            <MaskOverlayViewer caseId={id} />
            <div className="review-main card">
              <DraftView view={draftQ.data} />
            </div>
          </div>
          <EditDiffForm key={draftQ.data.revision} view={draftQ.data} onSubmitted={onSubmitted} />
        </>
      ) : null}
    </section>
  );
}
