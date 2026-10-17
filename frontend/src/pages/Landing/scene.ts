// three.js scene for the landing page: one particle cloud that morphs between
// the story formations as the page scrolls, over a quiet warm starfield.
// Plain three.js (no React renderer): the page pushes nothing per frame — the
// scene pulls scroll progress through `getProgress` inside its own loop.
//
// The morph runs on the GPU: each point carries its current formation (A) and
// the next one (B); a single uMix uniform drives a per-point STAGGERED
// transition, a mid-flight burst (points bloom outward, then lock into the
// next shape) and a scroll-velocity scatter that settles when scrolling stops.
// Points are rendered crisp (hard core + small halo) at up to 2.5x DPR.

import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  ShaderMaterial,
  WebGLRenderer,
} from "three";

import { POINT_COUNT, STAGE_LAYOUT, buildFormations, easeInOut } from "./formations";

export interface LandingSceneOptions {
  /** Story progress in chapters (0 … STAGE_LAYOUT.length - 1). */
  getProgress: () => number;
  reducedMotion: boolean;
  /** Narrow viewport: keep the formation centred behind the copy. */
  isCompact: () => boolean;
}

export interface LandingSceneHandle {
  setPointer(x: number, y: number): void;
  dispose(): void;
}

const VERTEX = /* glsl */ `
  attribute vec3 aTarget;
  attribute vec3 aColorB;
  attribute vec3 aDir;
  attribute float aRand;
  uniform float uMix;
  uniform float uBurst;
  uniform float uScatter;
  uniform float uTime;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uTwinkle;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    // Staggered: each point starts its journey at its own moment.
    float start = aRand * 0.42;
    float t = smoothstep(start, start + 0.58, uMix);
    vec3 p = mix(position, aTarget, t);
    // Mid-flight bloom: strongest halfway through a point's own transition.
    float flight = sin(3.14159265 * t);
    p += aDir * flight * uBurst * (0.35 + aRand * 0.65);
    // Swirl a little around the vertical axis while in flight.
    float a = flight * (aRand - 0.5) * 1.4;
    p.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xz;
    // Scroll-velocity scatter, settles back when scrolling stops.
    p += aDir * uScatter * (0.25 + aRand * 0.75);
    // Idle shimmer.
    p += aDir * 0.012 * sin(uTime * 1.3 + aRand * 40.0);

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = 8.0 / -mv.z;
    gl_PointSize = uSize * (0.6 + aRand * 0.7) * uPixelRatio * depth;
    vColor = mix(color, aColorB, t);
    float twinkle = 1.0 - uTwinkle * (0.25 + 0.25 * sin(uTime * 1.7 + aRand * 6.2831));
    vAlpha = twinkle * clamp(depth * 1.15, 0.35, 1.0) * (1.0 - 0.35 * flight * uBurst);
  }
`;

const FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    // Crisp core + soft halo: sharp at any resolution, still luminous.
    float core = smoothstep(0.5, 0.3, d);
    float halo = pow(smoothstep(0.5, 0.0, d), 3.0) * 0.5;
    vec3 col = mix(vColor, vec3(1.0, 0.97, 0.92), core * 0.18);
    gl_FragColor = vec4(col, (core * 0.9 + halo) * vAlpha * uOpacity);
  }
`;

export function createLandingScene(canvas: HTMLCanvasElement, opts: LandingSceneOptions): LandingSceneHandle {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.5));
  renderer.setClearColor(new Color(0x000000), 0);

  const scene = new Scene();
  const camera = new PerspectiveCamera(40, 1, 0.1, 120);
  camera.position.set(0, 0, STAGE_LAYOUT[0]!.z);

  // ---- story cloud (A = current formation, B = next formation)
  const formations = buildFormations();
  const posA = new Float32Array(formations[0]!.positions);
  const posB = new Float32Array(formations[0]!.positions);
  const colA = new Float32Array(formations[0]!.colors);
  const colB = new Float32Array(formations[0]!.colors);
  const rand = new Float32Array(POINT_COUNT);
  const dir = new Float32Array(POINT_COUNT * 3);
  let seed = 0x9e3779b9;
  const random = () => {
    seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x6d2b79f5) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < POINT_COUNT; i += 1) {
    rand[i] = random();
    const th = random() * Math.PI * 2;
    const u = random() * 2 - 1;
    const s = Math.sqrt(1 - u * u);
    dir[i * 3] = s * Math.cos(th);
    dir[i * 3 + 1] = u;
    dir[i * 3 + 2] = s * Math.sin(th);
  }
  const geometry = new BufferGeometry();
  const aPos = new BufferAttribute(posA, 3);
  const aTarget = new BufferAttribute(posB, 3);
  const aCol = new BufferAttribute(colA, 3);
  const aColB = new BufferAttribute(colB, 3);
  geometry.setAttribute("position", aPos);
  geometry.setAttribute("aTarget", aTarget);
  geometry.setAttribute("color", aCol);
  geometry.setAttribute("aColorB", aColB);
  geometry.setAttribute("aDir", new BufferAttribute(dir, 3));
  geometry.setAttribute("aRand", new BufferAttribute(rand, 1));
  const uniforms = {
    uMix: { value: 0 },
    uBurst: { value: opts.reducedMotion ? 0 : 0.85 },
    uScatter: { value: 0 },
    uTime: { value: 0 },
    uSize: { value: 4.4 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uOpacity: { value: 0.95 },
    uTwinkle: { value: opts.reducedMotion ? 0 : 1 },
  };
  const material = new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const cloud = new Points(geometry, material);
  cloud.frustumCulled = false; // vertices move in the shader
  const group = new Group();
  group.add(cloud);
  scene.add(group);

  // ---- warm starfield
  const starCount = 1600;
  const stars = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i += 1) {
    const a = random() * Math.PI * 2;
    const u = random() * 2 - 1;
    const r = 22 + random() * 18;
    const s = Math.sqrt(1 - u * u);
    stars[i * 3] = r * s * Math.cos(a);
    stars[i * 3 + 1] = r * u;
    stars[i * 3 + 2] = r * s * Math.sin(a) - 10;
  }
  const starGeometry = new BufferGeometry();
  starGeometry.setAttribute("position", new BufferAttribute(stars, 3));
  const starMaterial = new PointsMaterial({
    color: 0xb9a68c,
    size: 0.055,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
  });
  const starfield = new Points(starGeometry, starMaterial);
  scene.add(starfield);

  // ---- state
  const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  const pose = { rotY: STAGE_LAYOUT[0]!.rotY, tiltX: STAGE_LAYOUT[0]!.tiltX };
  let current = opts.getProgress();
  let previousTarget = current;
  let velocity = 0;
  let loadedStage = -1;
  let frame = 0;
  let running = true;
  let last = performance.now();
  const startedAt = last;
  const lastStage = STAGE_LAYOUT.length - 1;

  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }

  /** Upload formations A=stage, B=stage+1 only when the stage changes. */
  function loadStage(stage: number) {
    if (stage === loadedStage) return;
    const a = formations[stage]!;
    const b = formations[Math.min(stage + 1, lastStage)]!;
    posA.set(a.positions);
    colA.set(a.colors);
    posB.set(b.positions);
    colB.set(b.colors);
    aPos.needsUpdate = true;
    aCol.needsUpdate = true;
    aTarget.needsUpdate = true;
    aColB.needsUpdate = true;
    loadedStage = stage;
  }

  function place(p: number, time: number) {
    const stage = Math.min(Math.floor(p), lastStage);
    const t = stage >= lastStage ? 0 : p - stage;
    loadStage(stage);
    uniforms.uMix.value = t;
    const e = easeInOut(t);
    const la = STAGE_LAYOUT[stage]!;
    const lb = STAGE_LAYOUT[Math.min(stage + 1, lastStage)]!;
    const compact = opts.isCompact();
    // Travel between sides on an arc (dips towards the viewer mid-way).
    const arc = Math.sin(Math.PI * e);
    group.position.set(
      compact ? 0 : la.x + (lb.x - la.x) * e,
      la.y + (lb.y - la.y) * e - arc * 0.25,
      arc * (opts.reducedMotion ? 0 : 0.9),
    );
    pose.tiltX = la.tiltX + (lb.tiltX - la.tiltX) * e;
    pose.rotY = la.rotY + (lb.rotY - la.rotY) * e;
    const breathe = opts.reducedMotion ? 0 : Math.sin(time * 0.6) * 0.012;
    group.scale.setScalar(1 + breathe);
    camera.position.z = la.z + (lb.z - la.z) * e + (compact ? 3.5 : 0);
    const opacity = la.opacity + (lb.opacity - la.opacity) * e;
    uniforms.uOpacity.value = compact ? opacity * 0.38 : opacity;
  }

  function tick(now: number) {
    if (!running) return;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    const time = (now - startedAt) / 1000;
    const target = Math.max(0, Math.min(lastStage, opts.getProgress()));
    // Scroll velocity (chapters / s), smoothed, drives the scatter.
    const v = dt > 0 ? Math.abs(target - previousTarget) / dt : 0;
    previousTarget = target;
    velocity += (v - velocity) * Math.min(1, dt * 6);
    uniforms.uScatter.value = opts.reducedMotion ? 0 : Math.min(velocity * 0.35, 0.9);
    current = opts.reducedMotion ? target : current + (target - current) * Math.min(1, dt * 4.2);
    place(current, time);

    // The eye follows the cursor (eased), with a slow sway; never a full spin.
    pointer.sx += (pointer.x - pointer.sx) * Math.min(1, dt * 3.5);
    pointer.sy += (pointer.y - pointer.sy) * Math.min(1, dt * 3.5);
    const sway = opts.reducedMotion ? 0 : Math.sin(time * 0.35) * 0.1;
    group.rotation.y = pose.rotY + sway + pointer.sx * 0.38;
    group.rotation.x = pose.tiltX + pointer.sy * 0.22;
    starfield.rotation.y = opts.reducedMotion ? 0 : time * 0.004;
    starfield.position.y = opts.reducedMotion ? 0 : current * 0.6; // parallax drift with the story
    uniforms.uTime.value = time;
    renderer.render(scene, camera);
    frame = requestAnimationFrame(tick);
  }

  function onVisibility() {
    if (document.hidden) {
      running = false;
      cancelAnimationFrame(frame);
    } else if (!running) {
      running = true;
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }
  }

  resize();
  place(current, 0);
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", onVisibility);
  frame = requestAnimationFrame(tick);

  return {
    setPointer(x, y) {
      pointer.x = x;
      pointer.y = y;
    },
    dispose() {
      running = false;
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      geometry.dispose();
      material.dispose();
      starGeometry.dispose();
      starMaterial.dispose();
      renderer.dispose();
    },
  };
}
