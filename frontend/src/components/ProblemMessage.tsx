import { ApiError } from "../api/client";

// Renders the server's problem reason verbatim (short safe codes only — the
// API never puts user content in problem bodies).
export function ProblemMessage({ error, context }: { error: unknown; context?: string }) {
  const reason =
    error instanceof ApiError ? `${error.status || "network"} · ${error.reason}` : "UNEXPECTED_ERROR";
  return (
    <div className="problem" role="alert">
      {context ? <strong>{context}: </strong> : null}
      <code>{reason}</code>
    </div>
  );
}
