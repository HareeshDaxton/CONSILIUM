import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { useSession } from "../../auth/session";
import { Brand } from "../../components/BrandMark";

const LandingScene = lazy(() => import("./LandingScene"));

// Public landing page: a scroll-driven 3D story of the pipeline. Copy follows
// the project contract — screening support, not diagnosis (I-12), and no
// performance claim that is not backed by a stored evaluation run (I-11).

const CHAPTERS: readonly { id: string; label: string }[] = [
  { id: "top", label: "Overview" },
  { id: "vision", label: "Vision" },
  { id: "isolation", label: "Isolation" },
  { id: "audit", label: "Audit" },
  { id: "clinician", label: "Clinician" },
  { id: "start", label: "Start" },
];

function useReveal(root: React.RefObject<HTMLElement>) {
  useEffect(() => {
    const host = root.current;
    if (!host) return undefined;
    const targets = host.querySelectorAll<HTMLElement>("[data-reveal]");
    if (!("IntersectionObserver" in window)) {
      targets.forEach((t) => t.classList.add("is-visible"));
      return undefined;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) e.target.classList.add("is-visible");
      },
      { threshold: 0.2 },
    );
    targets.forEach((t) => io.observe(t));
    return () => io.disconnect();
  }, [root]);
}

export function LandingPage() {
  const { me } = useSession();
  const root = useRef<HTMLDivElement>(null);
  const sections = useRef<(HTMLElement | null)[]>([]);
  const progress = useRef(0);
  const [active, setActive] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  useReveal(root);

  // Chapter progress: i + fraction through section i (measured, so sections
  // may be taller than the viewport on small screens).
  useEffect(() => {
    let raf = 0;
    const measure = () => {
      raf = 0;
      const mid = window.scrollY + window.innerHeight * 0.5;
      const tops = sections.current.map((el) => (el ? el.offsetTop + el.offsetHeight * 0.5 : 0));
      let p = 0;
      for (let i = 0; i < tops.length - 1; i += 1) {
        const a = tops[i]!;
        const b = tops[i + 1]!;
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
        <section id="top" ref={register(0)} className="chapter chapter-hero chapter-left">
          <div className="chapter-copy">
            <p className="eyebrow" data-reveal>
              Glaucoma screening support · research prototype
            </p>
            <h1 data-reveal style={{ "--d": 1 } as React.CSSProperties}>
              Screening support
              <br />
              <span className="gradient-text">that shows its work.</span>
            </h1>
            <p className="lede" data-reveal style={{ "--d": 2 } as React.CSSProperties}>
              CONSILIUM reads a retinal fundus image and the referral note, lets three isolated specialist agents
              reason separately, audits every claim against the measured evidence — and hands the draft to a
              clinician, who signs the report.
            </p>
            <div className="cta-row" data-reveal style={{ "--d": 3 } as React.CSSProperties}>
              {cta}
              <button type="button" className="btn btn-glass btn-lg" onClick={() => go("vision")}>
                See how it works
              </button>
            </div>
            <ul className="pill-row" data-reveal style={{ "--d": 4 } as React.CSSProperties}>
              <li>Clinician-in-the-loop</li>
              <li>Evidence-linked claims</li>
              <li>Fails closed</li>
            </ul>
          </div>
          <div className="scroll-cue" aria-hidden="true">
            <span />
          </div>
        </section>

        <section id="vision" ref={register(1)} className="chapter chapter-left">
          <div className="chapter-copy glass" data-reveal>
            <p className="chapter-index">01 · Vision</p>
            <h2>From fundus to measurements.</h2>
            <p>
              SwinV2 and SegFormer run behind a private endpoint and return raw outputs only. Calibration, the
              screening tier, the vertical cup-to-disc ratio and mask quality checks are computed in our own code —
              versioned and testable.
            </p>
            <ul className="feature-list">
              <li>
                <span className="dot dot-cyan" /> Optic disc and cup segmented, then measured
              </li>
              <li>
                <span className="dot dot-magenta" /> Vertical CDR from mask extents, not an LLM
              </li>
              <li>
                <span className="dot dot-amber" /> Ungradable images stopped at the quality gate
              </li>
            </ul>
            <p className="footnote">Language models never see the image and never produce a number.</p>
          </div>
        </section>

        <section id="isolation" ref={register(2)} className="chapter chapter-right">
          <div className="chapter-copy glass" data-reveal>
            <p className="chapter-index">02 · Isolation</p>
            <h2>Three specialists. Three separate views.</h2>
            <p>
              A deterministic allowlist router gives the ophthalmologist, optometrist and pharmacist only the facts
              their role needs. They work in parallel and never read each other — a structural guard against
              consensus collapse, to be tested by ablation rather than assumed.
            </p>
            <div className="role-grid">
              <div>
                <span className="dot dot-teal" />
                <strong>Ophthalmologist</strong>
                <span>Structure and risk tier</span>
              </div>
              <div>
                <span className="dot dot-sky" />
                <strong>Optometrist</strong>
                <span>Tests and follow-up</span>
              </div>
              <div>
                <span className="dot dot-violet" />
                <strong>Pharmacist</strong>
                <span>Medication safety, no doses</span>
              </div>
            </div>
          </div>
        </section>

        <section id="audit" ref={register(3)} className="chapter chapter-left">
          <div className="chapter-copy glass" data-reveal>
            <p className="chapter-index">03 · Audit</p>
            <h2>Every claim is checked before anyone reads it.</h2>
            <p>
              The Director drafts claims that cite fact IDs and keeps real disagreements visible. A deterministic
              verifier checks numbers, references, coverage and unsafe wording first; an independent critic model
              checks entailment second.
            </p>
            <ol className="step-list">
              <li>
                <strong>Verifier</strong> exact numeric and reference checks
              </li>
              <li>
                <strong>Critic</strong> is each claim supported by its evidence?
              </li>
              <li>
                <strong>Bounded loop</strong> unresolved drafts are flagged <em>UNVERIFIED</em>, never passed
              </li>
            </ol>
          </div>
        </section>

        <section id="clinician" ref={register(4)} className="chapter chapter-right">
          <div className="chapter-copy glass" data-reveal>
            <p className="chapter-index">04 · Clinician</p>
            <h2>Nothing is final until a clinician signs.</h2>
            <p>
              Review the overlay and the evidence, then approve, edit or reject. Every edit carries a reason code;
              approved reports are immutable and attributed to the clinician who signed them.
            </p>
            <dl className="fact-row">
              <div>
                <dt>3</dt>
                <dd>isolated specialist agents</dd>
              </div>
              <div>
                <dt>2</dt>
                <dd>verification layers</dd>
              </div>
              <div>
                <dt>1</dt>
                <dd>accountable clinician</dd>
              </div>
            </dl>
          </div>
        </section>

        <section id="start" ref={register(5)} className="chapter chapter-center">
          <div className="chapter-copy" data-reveal>
            <h2 className="display">Built for accountable screening.</h2>
            <p className="lede">Sign in with your Google account. Access is limited to allowlisted clinicians.</p>
            <div className="cta-row cta-center">{cta}</div>
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
