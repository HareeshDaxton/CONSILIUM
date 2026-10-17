// Particle formations for the landing page's scroll story. Pure + seeded, so
// the scene is deterministic and these shapes are unit-testable without WebGL.
// Each formation places exactly POINT_COUNT points (xyz + rgb); the scene
// morphs between consecutive formations as the page scrolls.
//
//   0 eye        — the patient's eye (sphere + iris)
//   1 fundus     — retina with vessels, optic disc (cyan) and cup (magenta):
//                  the same colours as the server-rendered mask overlay
//   2 isolation  — three separate specialist clusters fed one-way by a router
//   3 audit      — verifier + critic rings around the drafted claims
//   4 clinician  — the signed report
//   5 eye        — closing call-to-action (same as 0)

export const POINT_COUNT = 5200;

export type RGB = readonly [number, number, number];

export interface Formation {
  positions: Float32Array;
  colors: Float32Array;
}

const hex = (h: number): RGB => [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];

export const PALETTE = {
  deep: hex(0x1d4ed8),
  sky: hex(0x60a5fa),
  teal: hex(0x2dd4bf),
  cyan: hex(0x22d3ee),
  violet: hex(0xa78bfa),
  magenta: hex(0xf472d0),
  amber: hex(0xfbbf24),
  ember: hex(0xf97316),
  vessel: hex(0xef4444),
  mint: hex(0x34d399),
  paper: hex(0xe2e8f0),
  slate: hex(0x94a3b8),
} as const;

/** mulberry32: tiny seeded PRNG. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Builder {
  readonly positions = new Float32Array(POINT_COUNT * 3);
  readonly colors = new Float32Array(POINT_COUNT * 3);
  private i = 0;
  constructor(readonly rand: () => number) {}

  get remaining(): number {
    return POINT_COUNT - this.i;
  }

  push(x: number, y: number, z: number, c: RGB, brightness = 1): void {
    if (this.i >= POINT_COUNT) return;
    const k = this.i * 3;
    this.positions[k] = x;
    this.positions[k + 1] = y;
    this.positions[k + 2] = z;
    this.colors[k] = Math.min(1, c[0] * brightness);
    this.colors[k + 1] = Math.min(1, c[1] * brightness);
    this.colors[k + 2] = Math.min(1, c[2] * brightness);
    this.i += 1;
  }

  jitter(scale: number): number {
    return (this.rand() - 0.5) * 2 * scale;
  }

  done(): Formation {
    if (this.i !== POINT_COUNT) throw new Error(`formation placed ${this.i}/${POINT_COUNT} points`);
    return { positions: this.positions, colors: this.colors };
  }
}

const mix = (a: RGB, b: RGB, t: number): RGB => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

function rotateY(x: number, z: number, angle: number): [number, number] {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c + z * s, -x * s + z * c];
}

function ring(
  b: Builder,
  n: number,
  cx: number,
  cy: number,
  r: number,
  c: RGB,
  { thickness = 0.03, tiltX = 0, z = 0, brightness = 1 } = {},
): void {
  for (let k = 0; k < n; k += 1) {
    const a = (k / n) * Math.PI * 2 + b.jitter(0.01);
    const rr = r + b.jitter(thickness);
    const x = cx + Math.cos(a) * rr;
    const y0 = Math.sin(a) * rr;
    const y = cy + y0 * Math.cos(tiltX);
    const zz = z + y0 * Math.sin(tiltX) + b.jitter(thickness);
    b.push(x, y, zz, c, brightness * (0.8 + b.rand() * 0.4));
  }
}

function eye(seed: number): Formation {
  const b = new Builder(rng(seed));
  const R = 2.1;
  const shell = 3400;
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < shell; k += 1) {
    const y = 1 - (k / (shell - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const a = golden * k;
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    const front = (z + 1) / 2;
    b.push(x * R, y * R, z * R, mix(PALETTE.deep, PALETTE.sky, front), 0.35 + front * 0.55);
  }
  // iris: radial streaks on the front cap
  for (let k = 0; k < 1300; k += 1) {
    const a = b.rand() * Math.PI * 2;
    const r = 0.55 + Math.pow(b.rand(), 0.8) * 0.55;
    const z = Math.sqrt(R * R - r * r) + 0.02;
    const streak = 0.6 + 0.4 * Math.abs(Math.sin(a * 24));
    b.push(Math.cos(a) * r, Math.sin(a) * r, z, mix(PALETTE.teal, PALETTE.cyan, b.rand()), streak);
  }
  // pupil rim
  ring(b, b.remaining, 0, 0, 0.5, PALETTE.paper, { z: Math.sqrt(R * R - 0.25) + 0.03, thickness: 0.02 });
  return b.done();
}

function fundus(seed: number): Formation {
  const b = new Builder(rng(seed));
  const R = 2.6;
  const disc: [number, number] = [0.85, 0.05];
  const dish = (x: number, y: number) => -0.12 * (x * x + y * y);
  // retinal field: warm, brighter towards the optic disc
  for (let k = 0; k < 2800; k += 1) {
    const a = b.rand() * Math.PI * 2;
    const r = Math.sqrt(b.rand()) * R;
    const x = Math.cos(a) * r;
    const y = Math.sin(a) * r;
    const near = Math.max(0, 1 - Math.hypot(x - disc[0], y - disc[1]) / 2.2);
    b.push(x, y, dish(x, y) + b.jitter(0.05), mix(PALETTE.ember, PALETTE.amber, near), 0.28 + near * 0.5);
  }
  // vessels arching out of the disc
  const vessels = 8;
  for (let v = 0; v < vessels; v += 1) {
    const base = (v / vessels) * Math.PI * 2 + 0.3;
    const bend = (v % 2 === 0 ? 1 : -1) * (0.5 + b.rand() * 0.4);
    const length = 1.7 + b.rand() * 1.1;
    for (let k = 0; k < 175; k += 1) {
      const t = b.rand();
      const a = base + bend * t;
      const x = disc[0] + Math.cos(a) * length * t + b.jitter(0.025);
      const y = disc[1] + Math.sin(a) * length * t + b.jitter(0.025);
      if (Math.hypot(x, y) > R) continue;
      b.push(x, y, dish(x, y) + 0.04, PALETTE.vessel, 0.75 - t * 0.35);
    }
  }
  ring(b, 650, disc[0], disc[1], 0.46, PALETTE.cyan, { z: dish(disc[0], disc[1]) + 0.08, brightness: 1.15 });
  ring(b, b.remaining, disc[0], disc[1], 0.24, PALETTE.magenta, {
    z: dish(disc[0], disc[1]) + 0.1,
    brightness: 1.2,
  });
  return b.done();
}

function isolation(seed: number): Formation {
  const b = new Builder(rng(seed));
  const hub: [number, number, number] = [0, -0.15, 0];
  const clusters: [number, number, number, RGB][] = [
    [-2.05, -1.0, 0, PALETTE.teal],
    [2.05, -1.0, 0, PALETTE.sky],
    [0, 1.55, 0, PALETTE.violet],
  ];
  for (const [cx, cy, cz, c] of clusters) {
    for (let k = 0; k < 1300; k += 1) {
      const a = b.rand() * Math.PI * 2;
      const u = b.rand() * 2 - 1;
      const r = 0.72 * Math.cbrt(b.rand());
      const s = Math.sqrt(1 - u * u);
      b.push(cx + r * s * Math.cos(a), cy + r * s * Math.sin(a), cz + r * u, c, 0.55 + (1 - r / 0.72) * 0.6);
    }
  }
  // one-way context beams: router → each role (never role ↔ role)
  for (const [cx, cy, cz] of clusters) {
    for (let k = 0; k < 300; k += 1) {
      const t = b.rand();
      b.push(
        hub[0] + (cx - hub[0]) * t * 0.8 + b.jitter(0.02),
        hub[1] + (cy - hub[1]) * t * 0.8 + b.jitter(0.02),
        hub[2] + (cz - hub[2]) * t * 0.8 + b.jitter(0.02),
        PALETTE.paper,
        0.25 + (1 - t) * 0.35,
      );
    }
  }
  while (b.remaining > 0) {
    const a = b.rand() * Math.PI * 2;
    const u = b.rand() * 2 - 1;
    const r = 0.22 * Math.cbrt(b.rand());
    const s = Math.sqrt(1 - u * u);
    b.push(hub[0] + r * s * Math.cos(a), hub[1] + r * s * Math.sin(a), hub[2] + r * u, PALETTE.cyan, 1.1);
  }
  return b.done();
}

function audit(seed: number): Formation {
  const b = new Builder(rng(seed));
  ring(b, 2000, 0, 0, 2.25, PALETTE.amber, { tiltX: 1.15, thickness: 0.05, brightness: 0.95 });
  ring(b, 1400, 0, 0, 1.55, PALETTE.teal, { tiltX: -0.95, thickness: 0.04, brightness: 1.0 });
  // the drafted claims: short evidence-linked lines at the centre
  const lines = 7;
  const perLine = Math.floor(b.remaining / lines);
  for (let l = 0; l < lines; l += 1) {
    const y = 0.75 - (l / (lines - 1)) * 1.5;
    const half = 0.55 + ((l * 37) % 5) * 0.08;
    const n = l === lines - 1 ? b.remaining : perLine;
    for (let k = 0; k < n; k += 1) {
      const x = -half + b.rand() * half * 2;
      b.push(x, y + b.jitter(0.025), b.jitter(0.05), PALETTE.paper, 0.45 + b.rand() * 0.35);
    }
  }
  return b.done();
}

function clinician(seed: number): Formation {
  const b = new Builder(rng(seed));
  const W = 1.35;
  const H = 1.75;
  const tilt = -0.28;
  const put = (x: number, y: number, z: number, c: RGB, br: number) => {
    const [rx, rz] = rotateY(x, z, tilt);
    b.push(rx, y, rz, c, br);
  };
  // page outline
  for (let k = 0; k < 900; k += 1) {
    const t = b.rand() * 4;
    const side = Math.floor(t);
    const f = t - side;
    const [x, y] =
      side === 0 ? [-W + f * 2 * W, H] : side === 1 ? [W, H - f * 2 * H] : side === 2 ? [W - f * 2 * W, -H] : [-W, -H + f * 2 * H];
    put(x + b.jitter(0.015), y + b.jitter(0.015), b.jitter(0.02), PALETTE.paper, 0.75);
  }
  // report text lines
  const lines = 9;
  for (let l = 0; l < lines; l += 1) {
    const y = H - 0.45 - l * 0.27;
    const end = l === 0 ? 0.2 : W - 0.25 - ((l * 53) % 7) * 0.12;
    for (let k = 0; k < 230; k += 1) {
      const x = -W + 0.25 + b.rand() * (end + W - 0.25);
      put(x, y + b.jitter(0.02), b.jitter(0.02), l === 0 ? PALETTE.sky : PALETTE.slate, l === 0 ? 0.9 : 0.5);
    }
  }
  // clinician seal + check mark
  const seal: [number, number] = [0.72, -1.12];
  for (let k = 0; k < 500; k += 1) {
    const a = (k / 500) * Math.PI * 2;
    put(seal[0] + Math.cos(a) * 0.45, seal[1] + Math.sin(a) * 0.45, 0.05, PALETTE.mint, 1.0);
  }
  const check: [number, number][] = [
    [seal[0] - 0.24, seal[1] + 0.0],
    [seal[0] - 0.06, seal[1] - 0.18],
    [seal[0] + 0.26, seal[1] + 0.2],
  ];
  for (let k = 0; k < 700; k += 1) {
    const seg = k < 280 ? 0 : 1;
    const t = b.rand();
    const [x0, y0] = check[seg]!;
    const [x1, y1] = check[seg + 1]!;
    put(x0 + (x1 - x0) * t + b.jitter(0.02), y0 + (y1 - y0) * t + b.jitter(0.02), 0.08, PALETTE.mint, 1.2);
  }
  // soft halo
  while (b.remaining > 0) {
    const a = b.rand() * Math.PI * 2;
    const r = 2.0 + b.rand() * 1.0;
    put(Math.cos(a) * r * 0.9, Math.sin(a) * r, b.jitter(0.6), PALETTE.teal, 0.18 + b.rand() * 0.2);
  }
  return b.done();
}

/** All story formations, in scroll order. */
export function buildFormations(seed = 7): Formation[] {
  const first = eye(seed);
  return [first, fundus(seed + 1), isolation(seed + 2), audit(seed + 3), clinician(seed + 4), first];
}

/** Where the formation sits for each chapter: away from the copy column,
 *  facing the camera (rotY) so flat shapes are never seen edge-on or from
 *  behind, and dimmer where copy sits on top of it. */
export const STAGE_LAYOUT: readonly {
  x: number;
  y: number;
  z: number;
  rotY: number;
  tiltX: number;
  opacity: number;
}[] = [
  { x: 1.9, y: 0, z: 8.4, rotY: -0.45, tiltX: 0.1, opacity: 0.95 },
  { x: 2.0, y: 0, z: 8.0, rotY: -0.18, tiltX: -0.1, opacity: 1.0 },
  { x: -2.6, y: 0.05, z: 9.6, rotY: 0.12, tiltX: 0.05, opacity: 0.95 },
  { x: 1.7, y: 0, z: 9.2, rotY: -0.25, tiltX: 0.2, opacity: 0.95 },
  { x: -2.0, y: 0, z: 8.4, rotY: 0.32, tiltX: 0.04, opacity: 0.95 },
  { x: 0, y: 0.1, z: 12.5, rotY: -0.35, tiltX: 0.1, opacity: 0.4 },
];

export const easeInOut = (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
