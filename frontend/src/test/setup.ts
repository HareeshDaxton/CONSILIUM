import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// jsdom has no object URLs; the overlay viewer turns server PNG blobs into them.
let objectUrls = 0;
if (!("createObjectURL" in URL)) {
  Object.defineProperty(URL, "createObjectURL", { value: () => `blob:test/${++objectUrls}`, writable: true });
  Object.defineProperty(URL, "revokeObjectURL", { value: () => undefined, writable: true });
}

afterEach(() => {
  cleanup();
});

// jsdom has no matchMedia; GSAP and the motion layer query it. Report
// "reduced motion" so animations are skipped and tests stay deterministic.
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
}

// jsdom does not implement scrolling; the router resets scroll on navigation.
window.scrollTo = () => undefined;

// jsdom (without pretendToBeVisual) has no animation frames.
if (typeof window.requestAnimationFrame !== "function") {
  window.requestAnimationFrame = (cb: FrameRequestCallback) => window.setTimeout(() => cb(performance.now()), 16);
  window.cancelAnimationFrame = (id: number) => window.clearTimeout(id);
}
