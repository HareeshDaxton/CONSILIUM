import type { EvidenceChip as Chip } from "../../api/types";
import { formatValue } from "../../lib/format";

// One cited fact from the server's evidence map. A ref missing from the map is
// shown as such (never guessed): the verifier should have caught it upstream.
export function EvidenceChip({ id, chip }: { id: string; chip: Chip | undefined }) {
  if (!chip) {
    return (
      <span className="chip chip-missing" title="Not in the case fact sheet">
        <span className="mono">{id}</span> · not in fact sheet
      </span>
    );
  }
  return (
    <span className="chip" title={chip.label} data-testid={`chip-${id}`}>
      <span className="mono">{id}</span> = {formatValue(chip.value, chip.unit)}
    </span>
  );
}
