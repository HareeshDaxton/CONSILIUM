import { useReasonCodes } from "../../api/queries";
import { ProblemMessage } from "../ProblemMessage";

export const OTHER = "OTHER";

// Closed taxonomy served by GET /v1/meta/reason-codes (AGENTS.md §15). The UI
// renders the server's list; it never adds codes. Required for edits and
// rejections; OTHER requires free text — the server enforces both.
export function ReasonCodePicker({
  selected,
  onChange,
  comment,
  onCommentChange,
}: {
  selected: readonly string[];
  onChange: (codes: string[]) => void;
  comment: string;
  onCommentChange: (text: string) => void;
}) {
  const { data, error, isLoading } = useReasonCodes();
  const otherNeedsText = selected.includes(OTHER) && !comment.trim();

  function toggle(code: string) {
    onChange(selected.includes(code) ? selected.filter((c) => c !== code) : [...selected, code]);
  }

  return (
    <fieldset className="reason-codes" data-testid="reason-codes">
      <legend>Reason codes (required for edits or rejection)</legend>
      {error ? <ProblemMessage error={error} context="Reason codes unavailable" /> : null}
      {isLoading ? <p>Loading…</p> : null}
      <div className="reason-grid">
        {data?.map(({ code, description }) => (
          <label key={code} className="reason">
            <input
              type="checkbox"
              name="reason_code"
              value={code}
              checked={selected.includes(code)}
              onChange={() => toggle(code)}
            />
            <span>
              <span className="mono">{code}</span>
              <span className="muted"> — {description}</span>
            </span>
          </label>
        ))}
      </div>
      <label htmlFor="review-comment" className="comment-label">
        Comment{selected.includes(OTHER) ? " (required for OTHER)" : " (optional)"}
      </label>
      <textarea
        id="review-comment"
        rows={2}
        maxLength={2000}
        value={comment}
        onChange={(e) => onCommentChange(e.target.value)}
        aria-invalid={otherNeedsText}
      />
      {otherNeedsText ? <p className="field-error">OTHER requires a free-text reason.</p> : null}
    </fieldset>
  );
}
