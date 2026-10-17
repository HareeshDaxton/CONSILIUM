import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import failed from "../../test/fixtures/case_failed.json";
import pending from "../../test/fixtures/case_pending.json";
import me from "../../test/fixtures/me_clinician.json";
import reportDoc from "../../test/fixtures/report_document.json";
import unverifiedDoc from "../../test/fixtures/report_document_unverified.json";
import timeline from "../../test/fixtures/timeline.json";
import { createMockApi, problem } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

const A = pending.case_id;
const png = () => ({ blob: new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }) });

function ready(extra: Record<string, unknown> = {}) {
  return {
    "GET /v1/me": me,
    [`GET /v1/cases/${A}`]: { ...pending, in_progress: false },
    [`GET /v1/cases/${A}/timeline`]: timeline,
    [`GET /v1/cases/${A}/report-document`]: reportDoc,
    [`GET /v1/cases/${A}/overlay`]: png,
    ...extra,
  };
}

describe("result page", () => {
  it("shows live pipeline progress while the server says it is running", async () => {
    const { fetchImpl, calls } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}`]: { ...pending, in_progress: true },
      [`GET /v1/cases/${A}/timeline`]: timeline.slice(0, 5),
    });
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    const progress = await screen.findByTestId("progress");
    await within(progress).findByText("Facts assembled");
    const done = progress.querySelectorAll("li.is-done");
    expect(done.length).toBe(4); // precheck, input rail, extract note, call vision
    expect(progress.querySelector("li.is-current")).toHaveTextContent("Facts assembled");
    expect(calls.some((c) => c.url.pathname.endsWith("/report-document"))).toBe(false);
  });

  it("renders the server's report document with evidence and model info", async () => {
    const { fetchImpl } = createMockApi(ready());
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    const doc = await screen.findByTestId("report-document");
    expect(within(doc).getByTestId("report-state")).toHaveTextContent("DRAFT — awaiting clinician review");
    expect(within(doc).getByTestId("image-findings")).toHaveTextContent("Vertical cup-to-disc ratio");
    expect(within(doc).getByText(reportDoc.impression)).toBeInTheDocument();
    expect(within(doc).getByText(reportDoc.footer)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Review & sign" })).toHaveAttribute("href", `/cases/${A}/review`);
  });

  it("exports PDF and Word through the server", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    const { fetchImpl, calls } = createMockApi(
      ready({
        [`GET /v1/cases/${A}/export`]: (req: { url: URL }) =>
          new Response(new Blob(["%PDF-1.7"]), {
            status: 200,
            headers: {
              "content-disposition": `attachment; filename="consilium-report-${A.slice(0, 8)}-draft.${req.url.searchParams.get("format")}"`,
            },
          }),
      }),
    );
    const { user } = await renderSignedIn(`/cases/${A}`, fetchImpl);
    await user.click(await screen.findByRole("button", { name: "Download PDF" }));
    await vi.waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    const anchor = click.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.download).toBe(`consilium-report-${A.slice(0, 8)}-draft.pdf`);
    await user.click(screen.getByRole("button", { name: "Download Word (.docx)" }));
    await vi.waitFor(() => expect(click).toHaveBeenCalledTimes(2));
    const formats = calls.filter((c) => c.url.pathname.endsWith("/export")).map((c) => c.url.searchParams.get("format"));
    expect(formats).toEqual(["pdf", "docx"]);
  });

  it("highlights open critic issues on an UNVERIFIED draft", async () => {
    const { fetchImpl } = createMockApi(ready({ [`GET /v1/cases/${A}/report-document`]: unverifiedDoc }));
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    const doc = await screen.findByTestId("report-document");
    expect(within(doc).getByTestId("report-state")).toHaveTextContent("UNVERIFIED DRAFT");
    expect(within(doc).getByText(/calibrated probability as evidence/)).toBeInTheDocument();
  });

  it("explains a vision-server failure in plain language", async () => {
    const { fetchImpl, calls } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${failed.case_id}`]: { ...failed, in_progress: false },
      [`GET /v1/cases/${failed.case_id}/timeline`]: [],
    });
    await renderSignedIn(`/cases/${failed.case_id}`, fetchImpl);
    const card = await screen.findByTestId("failure");
    expect(card).toHaveTextContent("VISION_UNAVAILABLE");
    expect(card).toHaveTextContent("Deva's server is running");
    expect(calls.some((c) => c.url.pathname.endsWith("/report-document"))).toBe(false);
  });

  it("says when the server has no report for the case", async () => {
    const { fetchImpl } = createMockApi(
      ready({ [`GET /v1/cases/${A}/report-document`]: () => problem(409, "REPORT_NOT_AVAILABLE") }),
    );
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    expect(await screen.findByText(/No report is available for this case \(REPORT_NOT_AVAILABLE\)/)).toBeInTheDocument();
  });

  it("non-clinicians are told reports are clinician-only and nothing is fetched", async () => {
    const { fetchImpl, calls } = createMockApi({
      ...ready(),
      "GET /v1/me": { sub: "sub-1", roles: ["submitter"] },
    });
    await renderSignedIn(`/cases/${A}`, fetchImpl);
    expect(await screen.findByText(/Only clinicians can view and export reports/)).toBeInTheDocument();
    expect(calls.some((c) => c.url.pathname.endsWith("/report-document"))).toBe(false);
  });
});
