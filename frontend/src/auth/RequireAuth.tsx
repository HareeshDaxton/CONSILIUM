import { Navigate, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { useSession } from "./session";

export function RequireAuth({ children }: { children: ReactNode }) {
  const { me } = useSession();
  const location = useLocation();
  // Keep the query string too (e.g. /cases?queue=approved) for after sign-in.
  if (!me) return <Navigate to="/login" replace state={{ from: `${location.pathname}${location.search}` }} />;
  return <>{children}</>;
}
