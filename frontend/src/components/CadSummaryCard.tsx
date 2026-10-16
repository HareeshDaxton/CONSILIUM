import type { CadSummary } from "../api/types";

// Values are rendered exactly as computed server-side (I-1).
export function CadSummaryCard({ cad }: { cad: CadSummary }) {
  return (
    <div className="card" data-testid="cad-summary">
      <h2>CAD summary</h2>
      <dl className="kv">
        <dt>Screening tier</dt>
        <dd>{cad.tier}</dd>
        <dt>Calibrated probability</dt>
        <dd>{cad.p_calibrated}</dd>
        <dt>Vertical CDR</dt>
        <dd>{cad.cdr_vertical}</dd>
        <dt>Mask QC</dt>
        <dd>{cad.mask_qc_ok ? "ok" : "FAILED — see caveats"}</dd>
        <dt>Quality flags</dt>
        <dd>{cad.quality_flags.length ? cad.quality_flags.join(", ") : "none"}</dd>
      </dl>
    </div>
  );
}
