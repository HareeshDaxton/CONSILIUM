import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import caseList from "../../test/fixtures/case_list.json";
import me from "../../test/fixtures/me_clinician.json";
import { createMockApi } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

// The main list request (limit 50); queue-count requests use limit 200.
const listCalls = (calls: { url: URL }[]) =>
  calls.filter((c) => c.url.pathname === "/v1/cases" && c.url.searchParams.get("limit") === "50");

describe("case list", () => {
  it("renders rows with status badges and the UNVERIFIED tag", async () => {
    const { fetchImpl, calls } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    await renderSignedIn("/cases", fetchImpl);
    const rows = await screen.findAllByTestId("case-row");
    expect(rows).toHaveLength(3);
    expect(within(rows[0]!).getByTestId("status-badge")).toHaveTextContent("PENDING REVIEW");
    expect(within(rows[1]!).getByText("UNVERIFIED")).toBeInTheDocument();
    expect(within(rows[2]!).getByTestId("status-badge")).toHaveTextContent("FAILED");
    expect(within(rows[0]!).getByRole("link")).toHaveAttribute(
      "href",
      `/cases/${caseList.items[0]!.case_id}`,
    );
    // default view = review queue (a server-side filter)
    expect(listCalls(calls)[0]!.url.searchParams.getAll("status")).toEqual([
      "PENDING_REVIEW",
      "NEEDS_ATTENTION",
    ]);
  });

  it("queue filters re-query the server", async () => {
    const { fetchImpl, calls } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    const { user } = await renderSignedIn("/cases", fetchImpl);
    await screen.findAllByTestId("case-row");
    await user.click(screen.getByRole("button", { name: "Failed" }));
    await screen.findByRole("button", { name: "Failed", pressed: true });
    await screen.findAllByTestId("case-row");
    expect(listCalls(calls).at(-1)!.url.searchParams.getAll("status")).toEqual([
      "FAILED",
      "REJECTED_INPUT",
    ]);
    await user.click(screen.getByRole("button", { name: "All" }));
    await screen.findByRole("button", { name: "All", pressed: true });
    await screen.findAllByTestId("case-row");
    expect(listCalls(calls).at(-1)!.url.searchParams.getAll("status")).toEqual([]);
  });

  it("a deep link keeps its queue filter through sign-in", async () => {
    const { fetchImpl, calls } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    await renderSignedIn("/cases?queue=approved", fetchImpl);
    expect(await screen.findByRole("button", { name: "Approved", pressed: true })).toBeInTheDocument();
    expect(listCalls(calls)[0]!.url.searchParams.getAll("status")).toEqual(["APPROVED", "APPROVED_WITH_EDITS"]);
  });

  it("queue cards show the server's row count per filter", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    await renderSignedIn("/cases", fetchImpl);
    expect(await screen.findByTestId("count-review")).toHaveTextContent(String(caseList.items.length));
    expect(screen.getByTestId("count-failed")).toHaveTextContent(String(caseList.items.length));
  });
});
