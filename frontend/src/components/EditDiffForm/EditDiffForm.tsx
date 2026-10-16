import { useMemo, useState } from "react";

import { useReviewMutation } from "../../api/queries";
import type { DraftView, ReviewAction, ReviewCommand, ReviewResult } from "../../api/types";
import { ProblemMessage } from "../ProblemMessage";
import { OTHER, ReasonCodePicker } from "../ReasonCodePicker/ReasonCodePicker";
import { blankFields, buildEditedDraft, diffDraft, initialEditState } from "./diff";
import type { EditState, EditableText } from "./diff";

// Field-level edit-and-diff form + review actions (ARCHITECTURE.md §5.1).
// Submits the FULL edited draft + the base revision; the server recomputes the
// authoritative diff and alone decides whether the action is allowed (status
// rules, UNVERIFIED resolution, self-approval, reason codes). Buttons are
// enabled from FORM state only (edits present? codes chosen?) — never from the
// case status — and every server refusal is shown verbatim.
export function EditDiffForm({
  view,
  onSubmitted,
}: {
  view: DraftView;
  onSubmitted: (result: ReviewResult) => void;
}) {
  const base = view.draft;
  const [state, setState] = useState<EditState>(() => initialEditState(base));
  const [codes, setCodes] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  const mutation = useReviewMutation(view.case_id);

  const changes = useMemo(() => diffDraft(base, state), [base, state]);
  const blanks = useMemo(() => blankFields(state), [state]);
  const hasEdits = changes.length > 0;
  const codesOk = codes.length > 0 && !(codes.includes(OTHER) && !comment.trim());

  function submit(action: ReviewAction) {
    const cmd: ReviewCommand = { action, base_revision: view.revision };
    if (action === "APPROVE_WITH_EDITS") cmd.edited_draft = buildEditedDraft(base, state);
    if (action !== "APPROVE") cmd.reason_codes = codes;
    if (action !== "APPROVE" && comment.trim()) cmd.comment = comment.trim();
    mutation.mutate(cmd, { onSuccess: (result) => onSubmitted(result) });
  }

  const setClaim = (id: string, patch: Partial<EditableText>) =>
    setState((s) => ({ ...s, claims: { ...s.claims, [id]: { ...s.claims[id]!, ...patch } } }));
  const setListItem = (key: "recommendations" | "limitations", i: number, patch: Partial<EditableText>) =>
    setState((s) => ({ ...s, [key]: s[key].map((e, j) => (j === i ? { ...e, ...patch } : e)) }));

  return (
    <section className="edit-diff card" data-testid="edit-diff-form">
      <h2>Edit and review</h2>

      <div className="edit-field">
        <label htmlFor="edit-impression">Impression</label>
        <textarea
          id="edit-impression"
          rows={2}
          value={state.impression}
          onChange={(e) => setState((s) => ({ ...s, impression: e.target.value }))}
        />
      </div>

      <h3>Claims</h3>
      {base.claims.map((c) => {
        const edit = state.claims[c.id]!;
        return (
          <div key={c.id} className={`edit-field${edit.removed ? " edit-removed" : ""}`}>
            <label htmlFor={`edit-claim-${c.id}`}>
              Claim <span className="mono">{c.id}</span> <span className="muted">({c.kind})</span>
            </label>
            <textarea
              id={`edit-claim-${c.id}`}
              rows={2}
              value={edit.text}
              disabled={edit.removed}
              onChange={(e) => setClaim(c.id, { text: e.target.value })}
            />
            <button type="button" className="btn btn-ghost" onClick={() => setClaim(c.id, { removed: !edit.removed })}>
              {edit.removed ? `Restore claim ${c.id}` : `Remove claim ${c.id}`}
            </button>
          </div>
        );
      })}

      {base.recommendations.length ? <h3>Recommendations</h3> : null}
      {base.recommendations.map((r, i) => {
        const edit = state.recommendations[i]!;
        return (
          <div key={i} className={`edit-field${edit.removed ? " edit-removed" : ""}`}>
            <label htmlFor={`edit-rec-${i}`}>Recommendation {i + 1}</label>
            <textarea
              id={`edit-rec-${i}`}
              rows={2}
              value={edit.text}
              disabled={edit.removed}
              onChange={(e) => setListItem("recommendations", i, { text: e.target.value })}
            />
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setListItem("recommendations", i, { removed: !edit.removed })}
            >
              {edit.removed ? `Restore recommendation ${i + 1}` : `Remove recommendation ${i + 1}`}
            </button>
            <span className="muted"> triggered by {r.triggered_by.join(", ")}</span>
          </div>
        );
      })}

      <h3>Limitations</h3>
      {base.limitations.map((_, i) => {
        const edit = state.limitations[i]!;
        return (
          <div key={i} className={`edit-field${edit.removed ? " edit-removed" : ""}`}>
            <label htmlFor={`edit-lim-${i}`}>Limitation {i + 1}</label>
            <textarea
              id={`edit-lim-${i}`}
              rows={1}
              value={edit.text}
              disabled={edit.removed}
              onChange={(e) => setListItem("limitations", i, { text: e.target.value })}
            />
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setListItem("limitations", i, { removed: !edit.removed })}
            >
              {edit.removed ? `Restore limitation ${i + 1}` : `Remove limitation ${i + 1}`}
            </button>
          </div>
        );
      })}
      {state.addedLimitations.map((text, i) => (
        <div key={`added-${i}`} className="edit-field edit-added">
          <label htmlFor={`edit-newlim-${i}`}>New limitation {i + 1}</label>
          <textarea
            id={`edit-newlim-${i}`}
            rows={1}
            value={text}
            onChange={(e) =>
              setState((s) => ({
                ...s,
                addedLimitations: s.addedLimitations.map((t, j) => (j === i ? e.target.value : t)),
              }))
            }
          />
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() =>
              setState((s) => ({ ...s, addedLimitations: s.addedLimitations.filter((_, j) => j !== i) }))
            }
          >
            Discard new limitation {i + 1}
          </button>
        </div>
      ))}
      <button
        type="button"
        className="btn"
        onClick={() => setState((s) => ({ ...s, addedLimitations: [...s.addedLimitations, ""] }))}
      >
        Add limitation
      </button>

      <h3>Changes vs AI draft (revision {view.revision})</h3>
      {hasEdits ? (
        <table className="table diff-table" data-testid="diff-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>AI text</th>
              <th>Clinician edit</th>
            </tr>
          </thead>
          <tbody>
            {changes.map((ch, i) => (
              <tr key={`${ch.path}-${i}`} data-testid="diff-row" className={`diff-${ch.kind}`}>
                <td className="mono">{ch.path}</td>
                <td className="diff-before">{ch.before ?? "—"}</td>
                <td className="diff-after">{ch.kind === "removed" ? "(removed)" : ch.after}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="empty">No edits.</p>
      )}
      {hasEdits ? (
        <button type="button" className="btn btn-ghost" onClick={() => setState(initialEditState(base))}>
          Reset all edits
        </button>
      ) : null}
      {blanks.length ? <p className="field-error">Empty text: {blanks.join(", ")}</p> : null}

      <ReasonCodePicker selected={codes} onChange={setCodes} comment={comment} onCommentChange={setComment} />

      {mutation.error ? <ProblemMessage error={mutation.error} context="Review refused by server" /> : null}

      <div className="row actions" role="group" aria-label="Review actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={mutation.isPending || hasEdits || codes.length > 0}
          onClick={() => submit("APPROVE")}
          title={hasEdits || codes.length ? "Approve sends no edits and no reason codes" : undefined}
        >
          Approve
        </button>
        <button
          type="button"
          className="btn btn-primary"
          disabled={mutation.isPending || !hasEdits || !codesOk || blanks.length > 0}
          onClick={() => submit("APPROVE_WITH_EDITS")}
        >
          Approve with edits
        </button>
        <button
          type="button"
          className="btn btn-danger"
          disabled={mutation.isPending || hasEdits || !codesOk}
          onClick={() => submit("REJECT")}
          title={hasEdits ? "Reset edits to reject" : undefined}
        >
          Reject
        </button>
      </div>
      <p className="hint">
        The server decides: unverified drafts need edits that resolve every flagged claim; you cannot approve a case
        you submitted.
      </p>
    </section>
  );
}
