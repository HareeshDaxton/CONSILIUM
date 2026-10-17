import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GoogleAccountsId } from "../../auth/google";
import { disableGoogleAutoSelect, loadGoogleIdentity } from "../../auth/google";
import caseList from "../../test/fixtures/case_list.json";
import me from "../../test/fixtures/me_clinician.json";
import { createMockApi, problem } from "../../test/mockApi";
import { TOKEN, renderApp, renderSignedIn, signInWithToken } from "../../test/renderApp";

vi.mock("../../auth/google", () => ({
  loadGoogleIdentity: vi.fn(),
  disableGoogleAutoSelect: vi.fn(),
}));

const CLIENT_ID = "1234567890-synthetic.apps.googleusercontent.com";
const GOOGLE_SESSION = {
  access_token: "consilium-session-token-synthetic",
  token_type: "Bearer",
  expires_in: 28800,
  user: {
    sub: "google:109876543210",
    roles: ["clinician"],
    email: "clinician.one@example.org",
    name: "Synthetic Clinician",
    picture: null,
  },
};

// A stand-in for google.accounts.id that lets the test "pick an account".
function fakeGis() {
  let callback: ((r: { credential: string }) => void) | null = null;
  const gis: GoogleAccountsId = {
    initialize: vi.fn((cfg) => {
      callback = cfg.callback;
    }),
    renderButton: vi.fn((el: HTMLElement) => {
      el.textContent = "Continue with Google";
    }),
    prompt: vi.fn(),
    cancel: vi.fn(),
    disableAutoSelect: vi.fn(),
  };
  return { gis, choose: (credential: string) => callback!({ credential }) };
}

beforeEach(() => {
  vi.mocked(loadGoogleIdentity).mockReset();
  vi.mocked(disableGoogleAutoSelect).mockReset();
});

describe("login — access token", () => {
  it("rejects a token the server refuses and stays on /login", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": () => problem(401, "TOKEN_INVALID", "unauthorized"),
    });
    const { user } = renderApp("/cases", fetchImpl);
    await signInWithToken(user, "garbage");
    expect(await screen.findByRole("alert")).toHaveTextContent("TOKEN_INVALID");
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });

  it("keeps the token in memory only and shows the server identity", async () => {
    const { fetchImpl, calls } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    await renderSignedIn("/cases", fetchImpl);
    expect(await screen.findByTestId("whoami")).toHaveTextContent("clin-1 · clinician");
    expect(await screen.findAllByTestId("case-row")).toHaveLength(3);
    const stored = [
      JSON.stringify({ ...localStorage }),
      JSON.stringify({ ...sessionStorage }),
      document.cookie,
    ];
    for (const s of stored) expect(s).not.toContain(TOKEN);
    expect(calls.at(-1)!.headers.get("Authorization")).toBe(`Bearer ${TOKEN}`);
  });

  it("returns to /login when the server says the token expired", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      "GET /v1/cases": () => problem(401, "TOKEN_EXPIRED", "unauthorized"),
    });
    await renderSignedIn("/cases", fetchImpl);
    expect(await screen.findByRole("status")).toHaveTextContent("TOKEN_EXPIRED");
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(disableGoogleAutoSelect).not.toHaveBeenCalled(); // expiry may silently re-sign
  });

  it("an explicit sign-out also stops Google auto sign-in", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    const { user } = await renderSignedIn("/cases", fetchImpl);
    await user.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument();
    expect(disableGoogleAutoSelect).toHaveBeenCalledOnce();
  });
});

describe("login — Google", () => {
  it("says so when the server has Google sign-in disabled", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/auth/config": { google_client_id: null } });
    renderApp("/login", fetchImpl);
    expect(await screen.findByTestId("google-unavailable")).toBeInTheDocument();
    expect(loadGoogleIdentity).not.toHaveBeenCalled();
  });

  it("distinguishes an unreachable server from a disabled sign-in, and can retry", async () => {
    let reachable = false;
    const { fetchImpl } = createMockApi({
      "GET /v1/auth/config": () =>
        reachable
          ? { json: { google_client_id: null } }
          : new Response("proxy error", { status: 500, statusText: "Internal Server Error" }),
    });
    const { user } = renderApp("/login", fetchImpl);
    expect(await screen.findByTestId("auth-unreachable")).toHaveTextContent("Can’t reach the CONSILIUM server");
    expect(screen.queryByTestId("google-unavailable")).not.toBeInTheDocument();
    reachable = true;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByTestId("google-unavailable")).toBeInTheDocument();
  });

  it("exchanges the Google credential for a server session and lands on the worklist", async () => {
    const { gis, choose } = fakeGis();
    vi.mocked(loadGoogleIdentity).mockResolvedValue(gis);
    const { fetchImpl, calls } = createMockApi({
      "GET /v1/auth/config": { google_client_id: CLIENT_ID },
      "POST /v1/auth/google": () => ({ json: GOOGLE_SESSION }),
      "GET /v1/cases": caseList,
    });
    renderApp("/cases", fetchImpl);
    expect(await screen.findByText("Continue with Google")).toBeInTheDocument();
    expect(gis.initialize).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: CLIENT_ID, auto_select: true }),
    );
    expect(gis.prompt).toHaveBeenCalled(); // One Tap for returning users

    choose("google-id-token-synthetic");

    expect(await screen.findByTestId("whoami")).toHaveTextContent("Synthetic Clinician · clinician");
    expect(await screen.findAllByTestId("case-row")).toHaveLength(3);
    const exchange = calls.find((c) => c.url.pathname === "/v1/auth/google")!;
    expect(exchange.body).toEqual({ credential: "google-id-token-synthetic" });
    expect(exchange.headers.get("Authorization")).toBeNull();
    // later calls carry OUR session token, never the Google credential
    const list = calls.filter((c) => c.url.pathname === "/v1/cases").at(-1)!;
    expect(list.headers.get("Authorization")).toBe(`Bearer ${GOOGLE_SESSION.access_token}`);
    expect(JSON.stringify({ ...localStorage, ...sessionStorage })).not.toContain(GOOGLE_SESSION.access_token);
  });

  it("shows the server's refusal for an account that is not allowlisted", async () => {
    const { gis, choose } = fakeGis();
    vi.mocked(loadGoogleIdentity).mockResolvedValue(gis);
    const { fetchImpl } = createMockApi({
      "GET /v1/auth/config": { google_client_id: CLIENT_ID },
      "POST /v1/auth/google": () => problem(403, "GOOGLE_ACCOUNT_NOT_AUTHORIZED", "forbidden"),
    });
    renderApp("/login", fetchImpl);
    await screen.findByText("Continue with Google");
    choose("google-id-token-synthetic");
    expect(await screen.findByRole("alert")).toHaveTextContent("403 · GOOGLE_ACCOUNT_NOT_AUTHORIZED");
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });

  it("reports when the Google script cannot load", async () => {
    vi.mocked(loadGoogleIdentity).mockRejectedValue(new Error("GIS_LOAD_FAILED"));
    const { fetchImpl } = createMockApi({ "GET /v1/auth/config": { google_client_id: CLIENT_ID } });
    renderApp("/login", fetchImpl);
    expect(await screen.findByRole("alert")).toHaveTextContent("Google sign-in could not load");
  });
});
