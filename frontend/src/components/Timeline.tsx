import type { TimelineEvent } from "../api/types";
import { formatDateTime } from "../lib/format";

export function Timeline({ events }: { events: TimelineEvent[] }) {
  if (!events.length) return <p className="empty">No events.</p>;
  return (
    <ol className="timeline" data-testid="timeline">
      {events.map((e, i) => (
        <li key={`${e.at}-${i}`}>
          <span className="mono">{e.event}</span>
          <span className="muted">
            {" "}
            · {e.actor} · {formatDateTime(e.at)}
          </span>
        </li>
      ))}
    </ol>
  );
}
