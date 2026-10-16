import type { Issue } from "../../api/types";

export function IssueList({ issues, testId }: { issues: Issue[]; testId?: string }) {
  if (!issues.length) return null;
  return (
    <ul className="issues" data-testid={testId}>
      {issues.map((issue, i) => (
        <li key={`${issue.type}-${i}`} className="issue">
          <span className="issue-type">⚠ {issue.type.replaceAll("_", " ")}</span>{" "}
          <span>{issue.detail}</span>
          <span className="muted"> · raised by {issue.raised_by}</span>
          {issue.evidence.length ? (
            <span className="muted mono"> · {issue.evidence.join(", ")}</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
