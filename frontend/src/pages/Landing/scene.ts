// three.js scene for the landing page: one particle cloud that morphs between
// the story formations as the page scrolls, over a quiet starfield. Plain
// three.js (no React renderer): the page pushes nothing per frame — the scene
// pulls scroll progress through `getProgress` inside its own render loop.

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
  attribute float aRand;
  uniform float uTime;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform float uTwinkle;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * (0.55 + aRand * 0.9) * uPixelRatio * (8.0 / -mv.z);
    vColor = color;
    vAlpha = 1.0 - uTwinkle * (0.35 + 0.35 * sin(uTime * 1.7 + aRand * 6.2831));
  }
`;

const FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float glow = pow(smoothstep(0.5, 0.0, d), 1.7);
    gl_FragColor = vec4(vColor * 1.2, glow * vAlpha * uOpacity);
  }
`;

export function createLandingScene(canvas: HTMLCanvasElement, opts: LandingSceneOptions): LandingSceneHandle {
  const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(new Color(0x000000), 0);

  const scene = new Scene();
  const camera = new PerspectiveCamera(42, 1, 0.1, 120);
  camera.position.set(0, 0, STAGE_LAYOUT[0]!.z);

  // ---- story cloud
  const formations = buildFormations();
  const positions = new Float32Array(formations[0]!.positions);
  const colors = new Float32Array(formations[0]!.colors);
  const rand = new Float32Array(POINT_COUNT);
  for (let i = 0; i < POINT_COUNT; i += 1) rand[i] = ((i * 2654435761) % 1000) / 1000;
  const geometry = new BufferGeometry();
  const posAttr = new BufferAttribute(positions, 3);
  const colAttr = new BufferAttribute(colors, 3);
  geometry.setAttribute("position", posAttr);
  geometry.setAttribute("color", colAttr);
  geometry.setAttribute("aRand", new BufferAttribute(rand, 1));
  const material = new ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: {
      uTime: { value: 0 },
      uSize: { value: 5.2 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uOpacity: { value: 0.95 },
      uTwinkle: { value: opts.reducedMotion ? 0 : 1 },
    },
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: AdditiveBlending,
  });
  const cloud = new Points(geometry, material);
  const group = new Group();
  group.add(cloud);
  scene.add(group);

  // ---- starfield
  const starCount = 1400;
  const stars = new Float32Array(starCount * 3);
  let seed = 0x9e3779b9;
  const random = () => {
    seed = (Math.imul(seed ^ (seed >>> 15), 0x2c1b3c6d) + 0x6d2b79f5) >>> 0;
    return seed / 4294967296;
  };
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
    color: 0x8fa3c7,
    size: 0.06,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  const starfield = new Points(starGeometry, starMaterial);
  scene.add(starfield);

  // ---- state
  const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  const pose = { rotY: STAGE_LAYOUT[0]!.rotY, tiltX: STAGE_LAYOUT[0]!.tiltX };
  let current = opts.getProgress();
  let drawn = Number.NaN;
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

  function morph(p: number) {
    const stage = Math.min(Math.floor(p), lastStage);
    const t = stage >= lastStage ? 0 : easeInOut(p - stage);
    const a = formations[stage]!;
    const b = formations[Math.min(stage + 1, lastStage)]!;
    for (let i = 0; i < positions.length; i += 1) {
      positions[i] = a.positions[i]! + (b.positions[i]! - a.positions[i]!) * t;
      colors[i] = a.colors[i]! + (b.colors[i]! - a.colors[i]!) * t;
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    const la = STAGE_LAYOUT[stage]!;
    const lb = STAGE_LAYOUT[Math.min(stage + 1, lastStage)]!;
    const compact = opts.isCompact();
    group.position.set(compact ? 0 : la.x + (lb.x - la.x) * t, la.y + (lb.y - la.y) * t, 0);
    pose.tiltX = la.tiltX + (lb.tiltX - la.tiltX) * t;
    pose.rotY = la.rotY + (lb.rotY - la.rotY) * t;
    camera.position.z = la.z + (lb.z - la.z) * t + (compact ? 3.5 : 0);
    const opacity = la.opacity + (lb.opacity - la.opacity) * t;
    // narrow screens: the copy sits on top of the cloud, so keep it in the background
    material.uniforms.uOpacity!.value = compact ? opacity * 0.38 : opacity;
  }

  function tick(now: number) {
    if (!running) return;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    const target = Math.max(0, Math.min(lastStage, opts.getProgress()));
    current = opts.reducedMotion ? target : current + (target - current) * Math.min(1, dt * 5);
    if (Math.abs(current - drawn) > 1e-4) {
      morph(current);
      drawn = current;
    }
    const time = (now - startedAt) / 1000;
    pointer.sx += (pointer.x - pointer.sx) * Math.min(1, dt * 3);
    pointer.sy += (pointer.y - pointer.sy) * Math.min(1, dt * 3);
    // gentle sway, never a full spin: flat shapes keep facing the viewer
    const sway = opts.reducedMotion ? 0 : Math.sin(time * 0.35) * 0.16;
    group.rotation.y = pose.rotY + sway + pointer.sx * 0.22;
    group.rotation.x = pose.tiltX + pointer.sy * 0.12;
    starfield.rotation.y = opts.reducedMotion ? 0 : time * 0.004;
    material.uniforms.uTime!.value = time;
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
  morph(current);
  drawn = current;
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
