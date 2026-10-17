import { useLayoutEffect } from "react";
import type { RefObject } from "react";

import { EASE, ScrollTrigger, gsap, prefersReducedMotion } from "./gsap";

/**
 * Declarative scroll choreography inside `root`:
 *   data-reveal            fade + rise when the element enters the viewport
 *   data-reveal="mask"     each child `.line > span` slides up out of a mask
 *   data-stagger           direct children reveal one after another
 *   data-words             words brighten one by one, scrubbed to scroll
 *   data-parallax="0.2"    drifts at a fraction of scroll speed
 * Everything is reverted on unmount (gsap.context) and skipped entirely when
 * the user prefers reduced motion: content is simply visible.
 */
export function useScrollReveal(root: RefObject<HTMLElement>, deps: readonly unknown[] = []) {
  useLayoutEffect(() => {
    const host = root.current;
    if (!host) return undefined;
    if (prefersReducedMotion()) {
      host.classList.add("motion-off");
      return undefined;
    }
    const ctx = gsap.context(() => {
      host.querySelectorAll<HTMLElement>("[data-reveal]").forEach((el) => {
        const mask = el.dataset.reveal === "mask";
        const targets = mask ? el.querySelectorAll(".line > span") : [el];
        gsap.fromTo(
          targets,
          mask ? { yPercent: 110 } : { autoAlpha: 0, y: 36 },
          {
            ...(mask ? { yPercent: 0 } : { autoAlpha: 1, y: 0 }),
            duration: mask ? 1.25 : 1.1,
            ease: EASE.out,
            stagger: 0.09,
            delay: Number(el.dataset.delay ?? 0),
            scrollTrigger: { trigger: el, start: "top 88%", once: true },
          },
        );
      });

      host.querySelectorAll<HTMLElement>("[data-stagger]").forEach((el) => {
        gsap.fromTo(
          el.children,
          { autoAlpha: 0, y: 28 },
          {
            autoAlpha: 1,
            y: 0,
            duration: 0.9,
            ease: EASE.soft,
            stagger: 0.08,
            scrollTrigger: { trigger: el, start: "top 85%", once: true },
          },
        );
      });

      host.querySelectorAll<HTMLElement>("[data-words]").forEach((el) => {
        const words = el.querySelectorAll(".word");
        gsap.fromTo(
          words,
          { opacity: 0.16 },
          {
            opacity: 1,
            ease: "none",
            stagger: 0.05,
            scrollTrigger: { trigger: el, start: "top 78%", end: "bottom 45%", scrub: 0.6 },
          },
        );
      });

      host.querySelectorAll<HTMLElement>("[data-parallax]").forEach((el) => {
        const speed = Number(el.dataset.parallax) || 0.15;
        gsap.to(el, {
          yPercent: -speed * 100,
          ease: "none",
          scrollTrigger: { trigger: el, start: "top bottom", end: "bottom top", scrub: true },
        });
      });
    }, host);
    ScrollTrigger.refresh();
    return () => ctx.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
