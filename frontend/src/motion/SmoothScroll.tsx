import Lenis from "lenis";
import { useEffect } from "react";
import { useLocation } from "react-router-dom";

import { ScrollTrigger, gsap, prefersReducedMotion } from "./gsap";

// Inertial smooth scrolling for the whole app (Lenis), driven by GSAP's ticker
// so ScrollTrigger scenes and the scroll position never drift apart. Native
// scrolling is kept when the user prefers reduced motion or in test envs.
export function SmoothScroll() {
  const { pathname } = useLocation();

  useEffect(() => {
    if (prefersReducedMotion() || typeof requestAnimationFrame !== "function") return undefined;
    const lenis = new Lenis({ duration: 1.15, easing: (t) => 1 - Math.pow(1 - t, 4), smoothWheel: true });
    lenis.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    document.documentElement.classList.add("lenis-on");
    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
      document.documentElement.classList.remove("lenis-on");
    };
  }, []);

  // New route: start at the top and let ScrollTrigger re-measure the new DOM.
  useEffect(() => {
    window.scrollTo?.(0, 0);
    const id = window.setTimeout(() => ScrollTrigger.refresh(), 60);
    return () => window.clearTimeout(id);
  }, [pathname]);

  return null;
}
