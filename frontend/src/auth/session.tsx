// In-memory session (ARCHITECTURE.md §5.1): the bearer token lives in a React
// ref only — never localStorage/sessionStorage/cookies — so closing or
// reloading the tab ends the session. Identity and roles come from the
// server (`GET /v1/me`, or the verified Google sign-in response), never from
// decoding a token client-side.

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { createApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import type { Me, SessionView } from "../api/types";
import { disableGoogleAutoSelect } from "./google";

interface Session {
  me: Me | null;
  endedReason: string | null;
  isClinician: boolean;
  login(token: string): Promise<void>;
  signInWithGoogle(credential: string): Promise<void>;
  logout(reason?: string): void;
}

const SessionContext = createContext<Session | null>(null);
const ApiContext = createContext<ApiClient | null>(null);

export function SessionProvider({
  children,
  fetchImpl,
}: {
  children: ReactNode;
  fetchImpl?: typeof fetch;
}) {
  const tokenRef = useRef<string | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [endedReason, setEndedReason] = useState<string | null>(null);

  const logout = useCallback((reason?: string) => {
    tokenRef.current = null;
    setMe(null);
    setEndedReason(reason ?? null);
    // A deliberate sign-out must not be undone by One Tap auto-select; an
    // expired session (reason given) may silently sign back in.
    if (!reason) disableGoogleAutoSelect();
  }, []);

  const api = useMemo(
    () =>
      createApiClient({
        getToken: () => tokenRef.current,
        onUnauthorized: (p) => logout(p.reason ?? "UNAUTHORIZED"),
        ...(fetchImpl ? { fetchImpl } : {}),
      }),
    [fetchImpl, logout],
  );

  const login = useCallback(
    async (token: string) => {
      const trimmed = token.trim();
      const who = await api.me(trimmed); // throws ApiError on a bad token
      tokenRef.current = trimmed;
      setEndedReason(null);
      setMe(who);
    },
    [api],
  );

  const signInWithGoogle = useCallback(
    async (credential: string) => {
      const view: SessionView = await api.googleSignIn(credential); // server verifies
      tokenRef.current = view.access_token;
      setEndedReason(null);
      setMe(view.user);
    },
    [api],
  );

  const session = useMemo<Session>(
    () => ({
      me,
      endedReason,
      isClinician: me?.roles.includes("clinician") ?? false,
      login,
      signInWithGoogle,
      logout,
    }),
    [me, endedReason, login, signInWithGoogle, logout],
  );

  return (
    <SessionContext.Provider value={session}>
      <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
    </SessionContext.Provider>
  );
}

export function useSession(): Session {
  const s = useContext(SessionContext);
  if (!s) throw new Error("useSession outside SessionProvider");
  return s;
}

export function useApi(): ApiClient {
  const api = useContext(ApiContext);
  if (!api) throw new Error("useApi outside SessionProvider");
  return api;
}
