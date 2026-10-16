import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import caseList from "../../test/fixtures/case_list.json";
import me from "../../test/fixtures/me_clinician.json";
import { createMockApi, problem } from "../../test/mockApi";
import { TOKEN, renderApp, renderSignedIn } from "../../test/renderApp";

describe("login", () => {
  it("rejects a token the server refuses and stays on /login", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": () => problem(401, "TOKEN_INVALID", "unauthorized"),
    });
    const { user } = renderApp("/cases", fetchImpl);
    await user.type(await screen.findByLabelText("Bearer token"), "garbage");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
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
    expect(screen.getByLabelText("Bearer token")).toBeInTheDocument();
  });

  it("signs out", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    const { user } = await renderSignedIn("/cases", fetchImpl);
    await user.click(await screen.findByRole("button", { name: "Sign out" }));
    expect(await screen.findByLabelText("Bearer token")).toBeInTheDocument();
  });
});
