import { Link, Outlet, useLocation } from "react-router-dom";

import { useSession } from "../auth/session";
import { QUEUE_FILTERS } from "../pages/CaseList/filters";
import { Brand } from "./BrandMark";

export const SCREENING_FOOTER =
  "Screening support only — not a diagnosis. Every report requires clinician review and attribution.";

const NAV_QUEUES = ["review", "attention", "approved", "failed", "all"] as const;

function Avatar({ name, picture }: { name: string; picture?: string | null | undefined }) {
  const initials = name
    .replace(/^google:/, "")
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]!.toUpperCase())
    .join("");
  return picture ? (
    <img className="avatar" src={picture} alt="" referrerPolicy="no-referrer" />
  ) : (
    <span className="avatar avatar-initials" aria-hidden="true">
      {initials || "?"}
    </span>
  );
}

// Signed-in workspace: sidebar navigation + content. Navigation targets are
// server-side queue filters; the shell holds no case logic.
export function AppShell() {
  const { me, logout } = useSession();
  const location = useLocation();
  const queue = new URLSearchParams(location.search).get("queue") ?? "review";
  const onList = location.pathname === "/cases";
  const display = me?.name ?? me?.email ?? me?.sub ?? "";

  return (
    <div className="shell">
      <aside className="sidebar">
        <Link to="/" className="sidebar-brand" aria-label="CONSILIUM home">
          <Brand />
        </Link>
        <Link to="/app" className={`btn btn-primary sidebar-cta${location.pathname === "/app" ? " is-active" : ""}`}>
          + New screening
        </Link>
        <nav aria-label="Workspace" className="sidebar-nav">
          <p className="sidebar-heading">Cases</p>
          {NAV_QUEUES.map((id) => {
            const f = QUEUE_FILTERS.find((q) => q.id === id)!;
            const active = onList && queue === id;
            return (
              <Link
                key={id}
                to={id === "review" ? "/cases" : `/cases?queue=${id}`}
                className={`nav-item${active ? " is-active" : ""}`}
                aria-current={active ? "page" : undefined}
              >
                <span className={`nav-dot nav-dot-${id}`} aria-hidden="true" />
                {f.label}
              </Link>
            );
          })}
        </nav>
        {me ? (
          <div className="sidebar-user">
            <Avatar name={display} picture={me.picture} />
            <div className="sidebar-user-text">
              <span data-testid="whoami">
                {display} · {me.roles.join(", ")}
              </span>
              {me.email && me.name ? <span className="muted">{me.email}</span> : null}
            </div>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => logout()}>
              Sign out
            </button>
          </div>
        ) : null}
      </aside>
      <div className="shell-main">
        <main className="content">
          <Outlet />
        </main>
        <footer className="footer">{SCREENING_FOOTER}</footer>
      </div>
    </div>
  );
}
