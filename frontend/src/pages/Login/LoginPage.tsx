import { useQuery } from "@tanstack/react-query";
import { Suspense, lazy, useLayoutEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../../api/client";
import { useApi, useSession } from "../../auth/session";
import { Brand } from "../../components/BrandMark";
import { GoogleSignInButton } from "../../components/GoogleSignInButton";
import { ProblemMessage } from "../../components/ProblemMessage";
import { EASE, gsap, prefersReducedMotion } from "../../motion/gsap";

const LandingScene = lazy(() => import("../Landing/LandingScene"));
const EYE_STAGE = () => 0; // the opening "eye" formation, following the cursor

// Sign-in. Primary: Google (the server verifies the Google ID token, checks the
// role allowlist and issues a CONSILIUM session token). Secondary: paste an
// access token (IdP-issued, or `scripts/dev_token.py` on the dev stack).
// Either way the session lives in memory only.
export function LoginPage() {
  const api = useApi();
  const { login, signInWithGoogle, endedReason } = useSession();
  const navigate = useNavigate();
  const location = useLocation();
  const [token, setToken] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const host = root.current;
    if (!host || prefersReducedMotion()) return undefined;
    const ctx = gsap.context(() => {
      gsap.from(".auth-title .line > span", { yPercent: 110, duration: 1.3, ease: EASE.out, delay: 0.15 });
      gsap.from(".auth-reveal", { autoAlpha: 0, y: 24, duration: 1.1, ease: EASE.out, stagger: 0.08, delay: 0.35 });
    }, host);
    return () => ctx.revert();
  }, []);
  const config = useQuery({
    queryKey: ["auth", "config"],
    queryFn: () => api.authConfig(),
    staleTime: Infinity,
    retry: false, // fail fast; the page offers a Retry button
  });
  // Network failure or 5xx (e.g. the Vite proxy with no backend behind it) =
  // unreachable; a 4xx is an answer from a running server.
  const unreachable =
    config.error != null && (!(config.error instanceof ApiError) || config.error.status === 0 || config.error.status >= 500);

  const from = (location.state as { from?: string } | null)?.from ?? "/app";

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      setToken("");
      navigate(from, { replace: true });
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void run(() => login(token));
  }

  const clientId = config.data?.google_client_id ?? null;

  return (
    <div className="auth-page" ref={root}>
      <Suspense fallback={<div className="landing-canvas landing-canvas-fallback" aria-hidden="true" />}>
        <LandingScene getProgress={EYE_STAGE} />
      </Suspense>
      <div className="auth-veil" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />

      <header className="auth-top">
        <Link to="/" className="auth-home" aria-label="CONSILIUM home">
          <Brand />
        </Link>
        <Link to="/" className="auth-back">
          Back to overview
        </Link>
      </header>

      <main className="auth-main">
        <section className="auth-card">
          <p className="kicker auth-reveal">Clinician workspace</p>
          <h1 className="auth-title">
            <span className="line">
              <span>Sign in</span>
            </span>
          </h1>
          <p className="auth-sub auth-reveal">
            Glaucoma screening support — <em>every report is signed by a clinician.</em>
          </p>
          {endedReason ? (
            <p className="notice" role="status">
              Session ended (<code>{endedReason}</code>). Sign in again.
            </p>
          ) : null}

          <div className="auth-google auth-reveal">
            {clientId ? (
              <GoogleSignInButton clientId={clientId} onCredential={(c) => void run(() => signInWithGoogle(c))} />
            ) : config.isLoading ? (
              <p className="muted">Loading sign-in options…</p>
            ) : unreachable ? (
              <div className="problem" role="alert" data-testid="auth-unreachable">
                Can’t reach the CONSILIUM server — is the backend running?{" "}
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => void config.refetch()}>
                  Retry
                </button>
              </div>
            ) : (
              <p className="muted" data-testid="google-unavailable">
                Google sign-in is not configured on this server.
              </p>
            )}
            {busy ? <p className="muted">Signing you in…</p> : null}
          </div>

          {error ? <ProblemMessage error={error} context="Sign-in failed" /> : null}

          <details className="auth-token auth-reveal">
            <summary>Use an access token</summary>
            <form onSubmit={onSubmit}>
              <label htmlFor="token">Bearer token</label>
              <textarea
                id="token"
                name="token"
                rows={4}
                autoComplete="off"
                spellCheck={false}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="eyJhbGciOi…"
              />
              <button type="submit" className="btn btn-glass" disabled={busy || !token.trim()}>
                Sign in with token
              </button>
            </form>
          </details>

          <p className="hint auth-reveal">
            Access is limited to allowlisted accounts. Your session is kept in memory only — closing the tab signs
            you out.
          </p>
        </section>
        <ul className="auth-points auth-reveal">
          <li>Every claim cites the evidence it rests on</li>
          <li>Numbers are checked against the measurements</li>
          <li>Nothing is final without your signature</li>
        </ul>
      </main>
      <p className="auth-legal">Screening support only — not a diagnosis.</p>
    </div>
  );
}
