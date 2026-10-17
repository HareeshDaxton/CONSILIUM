import { Suspense, lazy, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useSession } from "../../auth/session";
import { Brand } from "../../components/BrandMark";
import { EASE, ScrollTrigger, gsap, prefersReducedMotion } from "../../motion/gsap";
import { useScrollReveal } from "../../motion/reveal";
import { MaskLines, Words } from "../../motion/Text";

const LandingScene = lazy(() => import("./LandingScene"));

// Public landing page: a scroll-driven story of the pipeline. Copy follows the
// project contract — screening support, not diagnosis (I-12), and no
// performance claim that is not backed by a stored evaluation run (I-11).

const CHAPTERS: readonly { id: string; label: string }[] = [
  { id: "top", label: "Overview" },
  { id: "vision", label: "Vision" },
  { id: "isolation", label: "Isolation" },
  { id: "audit", label: "Audit" },
  { id: "clinician", label: "Clinician" },
  { id: "start", label: "Start" },
];

const ROLES = [
  {
    n: "01",
    name: "Ophthalmologist",
    sees: "Cup-to-disc measurements, screening tier, image quality, the referral findings.",
    does: "Interprets structure and risk. Never states a stage unless the note does.",
  },
  {
    n: "02",
    name: "Optometrist",
    sees: "Tier, vertical CDR, quality flags, pressures and exam findings.",
    does: "Proposes the examination pathway: visual field, OCT, gonioscopy, timing.",
  },
  {
    n: "03",
    name: "Pharmacist",
    sees: "Medications, intolerances, comorbidities — never the segmentation.",
    does: "Flags drug-class safety for prescriber judgement. No doses, no regimens.",
  },
] as const;

const AUDIT = [
  ["Verifier", "Every number is compared to the measurement it cites. Every reference must resolve. Unsafe wording is refused."],
  ["Critic", "A separate model asks one question per claim: is this supported by its evidence?"],
  ["Bounded loop", "Up to three revisions. A draft that still fails is shown as UNVERIFIED — never quietly passed."],
] as const;

export function LandingPage() {
  const { me } = useSession();
  const root = useRef<HTMLDivElement>(null);
  const sections = useRef<(HTMLElement | null)[]>([]);
  const progress = useRef(0);
  const [active, setActive] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  useScrollReveal(root);

  // Chapter progress for the 3D scene + the rail: i + fraction through section i.
  useEffect(() => {
    let raf = 0;
    const measure = () => {
      raf = 0;
      const mid = window.scrollY + window.innerHeight * 0.5;
      const centres = sections.current.map((el) =>
        el ? el.getBoundingClientRect().top + window.scrollY + el.offsetHeight * 0.5 : 0,
      );
      let p = 0;
      for (let i = 0; i < centres.length - 1; i += 1) {
        const a = centres[i]!;
        const b = centres[i + 1]!;
        if (mid >= b) p = i + 1;
        else if (mid > a) {
          p = i + (mid - a) / Math.max(b - a, 1);
          break;
        } else break;
      }
      progress.current = p;
      setActive(Math.round(p));
      setScrolled(window.scrollY > 24);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  // Page-specific choreography (shared reveals come from useScrollReveal).
  useLayoutEffect(() => {
    const host = root.current;
    if (!host || prefersReducedMotion()) return undefined;
    let settle: (() => void) | null = null;
    const ctx = gsap.context(() => {
      // Hero: intro on load, then lifts away as the story begins.
      gsap.from(".hero-meta > *", { autoAlpha: 0, y: 24, duration: 1.2, ease: EASE.out, stagger: 0.1, delay: 0.5 });
      gsap.to(".hero-inner", {
        yPercent: -18,
        autoAlpha: 0,
        ease: "none",
        scrollTrigger: { trigger: "#top", start: "top top", end: "bottom 15%", scrub: true },
      });

      // Reading progress: a hairline across the top of the page.
      gsap.fromTo(
        ".scroll-progress",
        { scaleX: 0 },
        { scaleX: 1, ease: "none", scrollTrigger: { start: 0, end: "max", scrub: 0.3 } },
      );

      // Chapter cards open from a slit as they arrive (scrubbed, reversible).
      host.querySelectorAll<HTMLElement>(".chapter-body").forEach((card) => {
        gsap.fromTo(
          card,
          { clipPath: "inset(18% 6% 18% 6% round 28px)", autoAlpha: 0.2, y: 60 },
          {
            clipPath: "inset(0% 0% 0% 0% round 28px)",
            autoAlpha: 1,
            y: 0,
            ease: "none",
            scrollTrigger: { trigger: card, start: "top 92%", end: "top 45%", scrub: 0.7 },
          },
        );
      });

      // Velocity skew: content leans with scroll speed, settles when it stops.
      const skewTargets = host.querySelectorAll(".chapter-body, .role-card, .manifesto-text");
      const skewTo = gsap.quickTo(skewTargets, "skewY", { duration: 0.6, ease: "power3.out" });
      const clamp = gsap.utils.clamp(-3.5, 3.5);
      ScrollTrigger.create({ onUpdate: (self) => skewTo(clamp(self.getVelocity() / -450)) });
      settle = () => skewTo(0);
      ScrollTrigger.addEventListener("scrollEnd", settle);

      const mm = gsap.matchMedia();
      // Isolation: a pinned horizontal track on wide screens.
      mm.add("(min-width: 900px)", () => {
        const track = host.querySelector<HTMLElement>(".roles-track");
        if (!track) return;
        const distance = () => track.scrollWidth - track.clientWidth;
        gsap.to(track, {
          x: () => -distance(),
          ease: "none",
          scrollTrigger: {
            trigger: "#isolation",
            start: "top top",
            end: () => `+=${distance() + window.innerHeight * 0.4}`,
            pin: true,
            scrub: 0.8,
            invalidateOnRefresh: true,
          },
        });
        // Depth: each card drifts at its own rate while the track is pinned.
        gsap.utils.toArray<HTMLElement>(".role-card").forEach((card, i) => {
          gsap.fromTo(
            card,
            { y: 40 + i * 30, rotate: i % 2 ? 1.5 : -1.5 },
            {
              y: -(i * 18),
              rotate: 0,
              ease: "none",
              scrollTrigger: {
                trigger: "#isolation",
                start: "top top",
                end: () => `+=${distance() + window.innerHeight * 0.4}`,
                scrub: 1,
                invalidateOnRefresh: true,
              },
            },
          );
        });
      });

      // Audit: the spine fills as the steps pass.
      gsap.fromTo(
        ".audit-spine-fill",
        { scaleY: 0 },
        {
          scaleY: 1,
          ease: "none",
          scrollTrigger: { trigger: ".audit-steps", start: "top 70%", end: "bottom 55%", scrub: true },
        },
      );

      // Clinician: counters roll in once.
      host.querySelectorAll<HTMLElement>("[data-count]").forEach((el) => {
        const target = Number(el.dataset.count);
        const state = { v: 0 };
        gsap.to(state, {
          v: target,
          duration: 1.6,
          ease: EASE.soft,
          onUpdate: () => {
            el.textContent = String(Math.round(state.v));
          },
          scrollTrigger: { trigger: el, start: "top 85%", once: true },
        });
      });
    }, host);
    return () => {
      if (settle) ScrollTrigger.removeEventListener("scrollEnd", settle);
      ctx.revert();
    };
  }, []);

  const getProgress = useCallback(() => progress.current, []);
  const register = (i: number) => (el: HTMLElement | null) => {
    sections.current[i] = el;
  };
  const go = (id: string) => document.getElementById(id)?.scrollIntoView?.({ behavior: "smooth" });

  const cta = me ? (
    <Link to="/app" className="btn btn-primary btn-lg">
      Start a screening
    </Link>
  ) : (
    <Link to="/login" className="btn btn-primary btn-lg">
      Sign in with Google
    </Link>
  );

  return (
    <div className="landing" ref={root}>
      <Suspense fallback={<div className="landing-canvas landing-canvas-fallback" aria-hidden="true" />}>
        <LandingScene getProgress={getProgress} />
      </Suspense>
      <div className="landing-vignette" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <div className="scroll-progress" aria-hidden="true" />

      <header className={`landing-nav${scrolled ? " is-scrolled" : ""}`}>
        <Link to="/" className="landing-brand" aria-label="CONSILIUM home">
          <Brand />
        </Link>
        <nav aria-label="Sections" className="landing-links">
          <button type="button" onClick={() => go("vision")}>
            Pipeline
          </button>
          <button type="button" onClick={() => go("audit")}>
            Safety
          </button>
          <button type="button" onClick={() => go("clinician")}>
            Clinicians
          </button>
        </nav>
        {me ? (
          <Link to="/app" className="btn btn-glass">
            Open CONSILIUM
          </Link>
        ) : (
          <Link to="/login" className="btn btn-glass">
            Sign in
          </Link>
        )}
      </header>

      <nav className="landing-rail" aria-label="Story progress">
        {CHAPTERS.map((c, i) => (
          <button
            key={c.id}
            type="button"
            className={i === active ? "is-active" : ""}
            aria-current={i === active ? "step" : undefined}
            onClick={() => go(c.id)}
          >
            <span className="rail-dot" />
            <span className="rail-label">{c.label}</span>
          </button>
        ))}
      </nav>

      <main>
        <section id="top" ref={register(0)} className="hero">
          <div className="hero-inner">
            <p className="kicker" data-reveal>
              Glaucoma screening support · research prototype
            </p>
            <h1 className="display-xl" data-reveal="mask">
              <MaskLines lines={["Screening support", <em key="e">that shows its work.</em>]} />
            </h1>
            <div className="hero-meta">
              <p className="lede">
                One fundus photograph. One referral note. Three specialists who never read each other, an audit of
                every claim — and a clinician who signs the report.
              </p>
              <div className="cta-row">
                {cta}
                <button type="button" className="btn btn-glass btn-lg" onClick={() => go("vision")}>
                  See how it works
                </button>
              </div>
            </div>
          </div>
          <div className="scroll-cue" aria-hidden="true">
            <span />
          </div>
        </section>

        <section className="manifesto" aria-label="Approach">
          <p className="manifesto-text" data-words>
            <Words text="When several AI agents read the same notes, they tend to agree with each other — and their mistakes agree too. CONSILIUM is built the other way round: separate views, audited claims, and a human who is accountable for every word." />
          </p>
        </section>

        <section id="vision" ref={register(1)} className="chapter chapter-left">
          <div className="chapter-grid">
            <p className="chapter-num" data-parallax="0.25">
              01
            </p>
            <div className="chapter-body">
              <p className="kicker" data-reveal>
                Vision
              </p>
              <h2 className="display-l" data-reveal="mask">
                <MaskLines lines={["From photograph", "to measurement."]} />
              </h2>
              <p className="body-l" data-reveal>
                Two vision models outline the optic disc and cup and estimate risk on a private GPU server. Every
                number in the report — the cup-to-disc ratio, the screening tier, the quality checks — is then computed
                in CONSILIUM&apos;s own, versioned code.
              </p>
              <ul className="feature-grid" data-stagger>
                <li>
                  <span className="feature-dot is-disc" />
                  <strong>Disc &amp; cup outlined</strong>
                  <span>Measured from the masks, not described by an LLM.</span>
                </li>
                <li>
                  <span className="feature-dot is-cup" />
                  <strong>Vertical CDR</strong>
                  <span>The clinically used height ratio, computed exactly.</span>
                </li>
                <li>
                  <span className="feature-dot is-amber" />
                  <strong>Quality gate</strong>
                  <span>Ungradable photographs are stopped with a clear reason.</span>
                </li>
              </ul>
              <p className="footnote" data-reveal>
                Language models never see the image and never produce a number.
              </p>
            </div>
          </div>
        </section>

        <section id="isolation" ref={register(2)} className="chapter chapter-roles">
          <div className="roles-head">
            <p className="kicker" data-reveal>
              02 · Isolation
            </p>
            <h2 className="display-l" data-reveal="mask">
              <MaskLines lines={["Three specialists.", "Three separate views."]} />
            </h2>
          </div>
          <div className="roles-track">
            {ROLES.map((r) => (
              <article key={r.name} className="role-card">
                <span className="role-n">{r.n}</span>
                <h3>{r.name}</h3>
                <dl>
                  <dt>Sees</dt>
                  <dd>{r.sees}</dd>
                  <dt>Does</dt>
                  <dd>{r.does}</dd>
                </dl>
              </article>
            ))}
            <article className="role-card role-card-note">
              <p>
                A deterministic router decides who sees what. They work in parallel and never read each other — a
                structural guard against consensus collapse, tested by ablation rather than assumed.
              </p>
            </article>
          </div>
        </section>

        <section id="audit" ref={register(3)} className="chapter chapter-left">
          <div className="chapter-grid">
            <p className="chapter-num" data-parallax="0.25">
              03
            </p>
            <div className="chapter-body">
              <p className="kicker" data-reveal>
                Audit
              </p>
              <h2 className="display-l" data-reveal="mask">
                <MaskLines lines={["Every claim is checked", "before anyone reads it."]} />
              </h2>
              <ol className="audit-steps">
                <span className="audit-spine" aria-hidden="true">
                  <span className="audit-spine-fill" />
                </span>
                {AUDIT.map(([title, text], i) => (
                  <li key={title} data-reveal>
                    <span className="audit-n">0{i + 1}</span>
                    <div>
                      <strong>{title}</strong>
                      <p>{text}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        <section id="clinician" ref={register(4)} className="chapter chapter-right">
          <div className="chapter-grid">
            <p className="chapter-num" data-parallax="0.25">
              04
            </p>
            <div className="chapter-body">
              <p className="kicker" data-reveal>
                Clinician
              </p>
              <h2 className="display-l" data-reveal="mask">
                <MaskLines lines={["Nothing is final", "until a clinician signs."]} />
              </h2>
              <p className="body-l" data-reveal>
                Review the outlines and the evidence, then approve, edit or reject. Every edit carries a reason; a
                signed report is immutable and carries the name of the clinician who signed it.
              </p>
              <dl className="counters" data-stagger>
                <div>
                  <dt data-count="3">3</dt>
                  <dd>isolated specialists</dd>
                </div>
                <div>
                  <dt data-count="2">2</dt>
                  <dd>verification layers</dd>
                </div>
                <div>
                  <dt data-count="1">1</dt>
                  <dd>accountable clinician</dd>
                </div>
              </dl>
            </div>
          </div>
        </section>

        <section id="start" ref={register(5)} className="finale">
          <h2 className="display-xl" data-reveal="mask">
            <MaskLines lines={["Built for", <em key="e">accountable</em>, "screening."]} />
          </h2>
          <p className="lede" data-reveal>
            Sign in with your Google account. Access is limited to allowlisted clinicians.
          </p>
          <div className="cta-row cta-center" data-reveal>
            {cta}
          </div>
        </section>
      </main>

      <footer className="landing-footer">
        <Brand size={22} />
        <p>
          Research prototype for glaucoma screening support. It does not diagnose, stage disease or recommend doses.
          Performance figures are published only from stored evaluation runs.
        </p>
      </footer>
    </div>
  );
}
