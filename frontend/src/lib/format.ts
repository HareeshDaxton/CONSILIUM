import type { FactValue } from "../api/types";

export function shortId(id: string): string {
  return id.slice(0, 8);
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Display only: numbers are shown exactly as the server sent them (CAD values
// are computed server-side, I-1); no rounding that could misstate a value.
export function formatValue(value: FactValue, unit?: string | null): string {
  const text = Array.isArray(value)
    ? value.length
      ? value.join(", ")
      : "none"
    : typeof value === "boolean"
      ? value
        ? "yes"
        : "no"
      : String(value);
  return unit ? `${text} ${unit}` : text;
}
