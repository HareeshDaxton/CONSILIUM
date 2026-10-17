// Single GSAP entry point: plugins are registered once, here.
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

/** True when the user asked the OS for less motion (or the env has no matchMedia). */
export function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? true;
}

/** Shared easing language: one "expo" family keeps every surface consistent. */
export const EASE = {
  out: "expo.out",
  inOut: "expo.inOut",
  soft: "power3.out",
} as const;

export { gsap, ScrollTrigger };
