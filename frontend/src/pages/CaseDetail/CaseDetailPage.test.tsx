import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import failed from "../../test/fixtures/case_failed.json";
import pending from "../../test/fixtures/case_pending.json";
import facts from "../../test/fixtures/facts.json";
import me from "../../test/fixtures/me_clinician.json";
import timeline from "../../test/fixtures/timeline.json";
import { createMockApi, problem } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

const A = pending.case_id;

describe("case detail", () => {
  it("shows CAD summary, de-identified note facts and the timeline", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/timeline`]: timeline,
      [`GET /v1/cases/${A}/facts`]: facts,
    });
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    const cad = await screen.findByTestId("cad-summary");
    expect(cad).toHaveTextContent(String(pending.cad_summary.cdr_vertical));
    expect(cad).toHaveTextContent(pending.cad_summary.tier);
    expect(await screen.findByText("note.age")).toBeInTheDocument();
    // CAD facts are not repeated in the note table
    expect(screen.queryByText("cad.cdr.vertical")).not.toBeInTheDocument();
    expect(await screen.findByTestId("timeline")).toHaveTextContent("node.persist_pending");
    expect(screen.getByRole("link", { name: "Open review" })).toHaveAttribute(
      "href",
      `/cases/${A}/review`,
    );
  });

  it("shows the failure reason of a failed case", async () => {
    const id = failed.case_id;
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${id}`]: failed,
      [`GET /v1/cases/${id}/timeline`]: [],
      [`GET /v1/cases/${id}/facts`]: () => problem(409, "FACTS_NOT_AVAILABLE"),
    });
    await renderSignedIn(`/cases/${id}`, fetchImpl);
    expect(await screen.findByTestId("failure")).toHaveTextContent("VISION_UNAVAILABLE");
    expect(await screen.findByText(/FACTS_NOT_AVAILABLE/)).toBeInTheDocument();
  });

  it("submitters get no fact-sheet request and no review links", async () => {
    const { fetchImpl, calls } = createMockApi({
      "GET /v1/me": { sub: "sub-1", roles: ["submitter"] },
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/timeline`]: timeline,
    });
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    await screen.findByTestId("cad-summary");
    expect(screen.queryByRole("link", { name: "Open review" })).not.toBeInTheDocument();
    expect(calls.some((c) => c.url.pathname.endsWith("/facts"))).toBe(false);
  });
});
