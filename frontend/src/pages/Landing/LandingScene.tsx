import { useEffect, useRef, useState } from "react";

import { createLandingScene } from "./scene";

// Fixed full-viewport WebGL canvas behind the landing copy. Loaded lazily so
// three.js never ships with the clinician app pages. Without WebGL it falls
// back to a static gradient (the copy carries the whole story either way).
export default function LandingScene({ getProgress }: { getProgress: () => number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return undefined;
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let handle: ReturnType<typeof createLandingScene> | null = null;
    try {
      handle = createLandingScene(el, {
        getProgress,
        reducedMotion,
        isCompact: () => window.innerWidth < 900,
      });
    } catch {
      setFailed(true);
      return undefined;
    }
    const onPointer = (e: PointerEvent) => {
      handle?.setPointer((e.clientX / window.innerWidth) * 2 - 1, (e.clientY / window.innerHeight) * 2 - 1);
    };
    if (!reducedMotion) window.addEventListener("pointermove", onPointer, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onPointer);
      handle?.dispose();
    };
  }, [getProgress]);

  return (
    <div className="landing-canvas" aria-hidden="true">
      {failed ? <div className="landing-canvas-fallback" /> : <canvas ref={canvas} />}
    </div>
  );
}
