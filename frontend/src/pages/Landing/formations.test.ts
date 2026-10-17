import { describe, expect, it } from "vitest";

import { POINT_COUNT, STAGE_LAYOUT, buildFormations, easeInOut } from "./formations";

describe("landing formations", () => {
  const formations = buildFormations();

  it("one formation per chapter, each with exactly POINT_COUNT points", () => {
    expect(formations).toHaveLength(STAGE_LAYOUT.length);
    for (const f of formations) {
      expect(f.positions).toHaveLength(POINT_COUNT * 3);
      expect(f.colors).toHaveLength(POINT_COUNT * 3);
    }
  });

  it("positions are finite and framed; colours are valid", () => {
    // plain loops + one assertion per formation (≈190k expect() calls is slow)
    formations.forEach((f, n) => {
      let maxRadius = 0;
      let bad = 0;
      for (let i = 0; i < POINT_COUNT; i += 1) {
        const [x, y, z] = [f.positions[i * 3]!, f.positions[i * 3 + 1]!, f.positions[i * 3 + 2]!];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) bad += 1;
        maxRadius = Math.max(maxRadius, Math.hypot(x, y, z));
      }
      const badColours = Array.from(f.colors).filter((c) => !(c >= 0 && c <= 1)).length;
      expect({ formation: n, bad, badColours, framed: maxRadius < 4 }).toEqual({
        formation: n,
        bad: 0,
        badColours: 0,
        framed: true,
      });
    });
  });

  it("is deterministic and the closing chapter returns to the opening shape", () => {
    const again = buildFormations();
    formations.forEach((f, i) => expect(Array.from(again[i]!.positions)).toEqual(Array.from(f.positions)));
    expect(formations.at(-1)).toBe(formations[0]);
  });

  it("consecutive chapters are different shapes (something actually morphs)", () => {
    for (let i = 0; i < formations.length - 2; i += 1) {
      const a = formations[i]!.positions;
      const b = formations[i + 1]!.positions;
      let moved = 0;
      for (let k = 0; k < a.length; k += 1) moved += Math.abs(a[k]! - b[k]!);
      expect(moved / a.length).toBeGreaterThan(0.2);
    }
  });

  it("easing is monotonic from 0 to 1", () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    let prev = 0;
    for (let t = 0.05; t <= 1; t += 0.05) {
      expect(easeInOut(t)).toBeGreaterThanOrEqual(prev);
      prev = easeInOut(t);
    }
  });
});
