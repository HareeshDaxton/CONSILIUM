import type { Disagreement } from "../../api/types";

// Preserved dissent is a safety feature (AGENTS.md §12): every position is
// shown with its role; nothing is merged or summarised away.
export function DisagreementList({ items }: { items: Disagreement[] }) {
  if (!items.length) return null;
  return (
    <section className="disagreements" data-testid="disagreements">
      <h3>Disagreements preserved</h3>
      {items.map((d, i) => (
        <div key={`${d.topic}-${i}`} className="disagreement">
          <strong>{d.topic}</strong>
          <ul>
            {d.positions.map(([role, position], j) => (
              <li key={`${role}-${j}`}>
                <span className="role-tag">{role}</span> {position}
              </li>
            ))}
          </ul>
          <p className="muted">Resolution: {d.resolution}</p>
        </div>
      ))}
    </section>
  );
}
