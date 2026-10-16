import type { FactView } from "../api/types";
import { formatValue } from "../lib/format";

// C_global fact sheet. Note-derived free text is de-identified but untrusted
// (I-7): it is rendered as plain text, never as HTML/markdown.
export function FactsTable({ facts, source }: { facts: FactView[]; source: FactView["source"] }) {
  const rows = facts.filter((f) => f.source === source);
  if (!rows.length) return <p className="empty">None.</p>;
  return (
    <table className="table table-compact">
      <tbody>
        {rows.map((f) => (
          <tr key={f.id}>
            <th scope="row">
              {f.label}
              <div className="mono muted">{f.id}</div>
            </th>
            <td className={f.id.startsWith("note.") && typeof f.value === "string" ? "untrusted" : ""}>
              {formatValue(f.value, f.unit)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
