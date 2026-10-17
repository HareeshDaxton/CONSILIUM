import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../../api/client";
import { useApi, useSession } from "../../auth/session";
import { Brand, BrandMark } from "../../components/BrandMark";
import { GoogleSignInButton } from "../../components/GoogleSignInButton";
import { ProblemMessage } from "../../components/ProblemMessage";

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
    <div className="auth-page">
      <aside className="auth-aside" aria-hidden="true">
        <div className="auth-orbit">
          <span className="orbit orbit-1" />
          <span className="orbit orbit-2" />
          <span className="orbit orbit-3" />
          <BrandMark size={96} />
        </div>
        <blockquote>
          “Isolate the reasoning, audit the synthesis, keep a clinician accountable.”
        </blockquote>
        <ul>
          <li>Every claim cites the evidence it rests on</li>
          <li>Numbers are checked against the measurements</li>
          <li>Nothing is final without your signature</li>
        </ul>
      </aside>

      <main className="auth-main">
        <Link to="/" className="auth-home" aria-label="CONSILIUM home">
          <Brand />
        </Link>
        <section className="auth-card">
          <h1>Sign in</h1>
          <p className="muted">Clinician review workspace for glaucoma screening support.</p>
          {endedReason ? (
            <p className="notice" role="status">
              Session ended (<code>{endedReason}</code>). Sign in again.
            </p>
          ) : null}

          <div className="auth-google">
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

          <details className="auth-token">
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

          <p className="hint">
            Access is limited to allowlisted accounts. Your session is kept in memory only — closing the tab signs
            you out.
          </p>
        </section>
        <p className="auth-legal">Screening support only — not a diagnosis.</p>
      </main>
    </div>
  );
}
