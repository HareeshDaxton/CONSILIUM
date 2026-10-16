import { useState } from "react";
import type { FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { useSession } from "../../auth/session";
import { ProblemMessage } from "../../components/ProblemMessage";

// Bearer-token sign-in. The token is validated by the server (GET /v1/me) and
// then kept in memory only. Production tokens come from the IdP; in dev, mint
// one with `uv run python scripts/dev_token.py --sub clin-1 --role clinician`.
export function LoginPage() {
  const { login, endedReason } = useSession();
  const navigate = useNavigate();
  const location = useLocation();
  const [token, setToken] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const from = (location.state as { from?: string } | null)?.from ?? "/cases";

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(token);
      setToken("");
      navigate(from, { replace: true });
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="login card">
      <h1>Sign in</h1>
      {endedReason ? (
        <p className="notice" role="status">
          Session ended (<code>{endedReason}</code>). Sign in again.
        </p>
      ) : null}
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
        <p className="hint">Kept in memory only — closing or reloading the tab signs you out.</p>
        <button type="submit" className="btn btn-primary" disabled={busy || !token.trim()}>
          {busy ? "Checking…" : "Sign in"}
        </button>
      </form>
      {error ? <ProblemMessage error={error} context="Sign-in failed" /> : null}
    </section>
  );
}
