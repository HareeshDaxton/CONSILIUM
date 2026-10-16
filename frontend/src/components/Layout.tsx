import { Link, Outlet } from "react-router-dom";

import { useSession } from "../auth/session";

export const SCREENING_FOOTER =
  "Screening support only — not a diagnosis. Every report requires clinician review and attribution.";

export function Layout() {
  const { me, logout } = useSession();
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/cases" className="brand">
          CONSILIUM <span className="brand-sub">clinician review</span>
        </Link>
        {me ? (
          <div className="whoami">
            <span data-testid="whoami">
              {me.sub} · {me.roles.join(", ")}
            </span>
            <button type="button" className="btn btn-ghost" onClick={() => logout()}>
              Sign out
            </button>
          </div>
        ) : null}
      </header>
      <main className="content">
        <Outlet />
      </main>
      <footer className="footer">{SCREENING_FOOTER}</footer>
    </div>
  );
}
