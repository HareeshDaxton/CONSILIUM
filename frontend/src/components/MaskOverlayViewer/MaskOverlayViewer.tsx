import { useState } from "react";

import { useOverlayLayer } from "../../api/queries";
import { ProblemMessage } from "../ProblemMessage";

// Stacks the server-rendered layers (fundus image, disc outline, cup outline —
// GET /v1/cases/{id}/overlay?layer=…). Toggles and the opacity slider are pure
// CSS on those images: no canvas, no mask decoding, no geometry in the browser.
export function MaskOverlayViewer({ caseId }: { caseId: string }) {
  const image = useOverlayLayer(caseId, "image");
  const disc = useOverlayLayer(caseId, "disc");
  const cup = useOverlayLayer(caseId, "cup");
  const [showDisc, setShowDisc] = useState(true);
  const [showCup, setShowCup] = useState(true);
  const [opacity, setOpacity] = useState(100);

  const error = image.error ?? disc.error ?? cup.error;

  return (
    <div className="overlay-viewer card" data-testid="overlay-viewer">
      <h2>Fundus + mask overlay</h2>
      {error ? <ProblemMessage error={error} context="Overlay unavailable" /> : null}
      <div className="overlay-stack">
        {image.url ? <img src={image.url} alt="Fundus image" className="overlay-base" /> : null}
        {disc.url ? (
          <img
            src={disc.url}
            alt="Optic disc outline"
            className="overlay-layer"
            data-testid="layer-disc"
            hidden={!showDisc}
            style={{ opacity: opacity / 100 }}
          />
        ) : null}
        {cup.url ? (
          <img
            src={cup.url}
            alt="Optic cup outline"
            className="overlay-layer"
            data-testid="layer-cup"
            hidden={!showCup}
            style={{ opacity: opacity / 100 }}
          />
        ) : null}
      </div>
      <div className="overlay-controls">
        <label>
          <input type="checkbox" checked={showDisc} onChange={(e) => setShowDisc(e.target.checked)} />
          <span className="swatch swatch-disc" aria-hidden="true" /> Disc outline
        </label>
        <label>
          <input type="checkbox" checked={showCup} onChange={(e) => setShowCup(e.target.checked)} />
          <span className="swatch swatch-cup" aria-hidden="true" /> Cup outline
        </label>
        <label className="opacity">
          Outline opacity
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            aria-valuetext={`${opacity}%`}
          />
          <span className="mono">{opacity}%</span>
        </label>
      </div>
      <p className="hint">Masks rendered server-side from the stored segmentation (post-processed as for CDR).</p>
    </div>
  );
}
