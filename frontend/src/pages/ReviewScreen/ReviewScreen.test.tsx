import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import pending from "../../test/fixtures/case_pending.json";
import unverified from "../../test/fixtures/draft_unverified.json";
import verified from "../../test/fixtures/draft_verified.json";
import me from "../../test/fixtures/me_clinician.json";
import reasonCodes from "../../test/fixtures/reason_codes.json";
import report from "../../test/fixtures/report.json";
import reviewResult from "../../test/fixtures/review_result.json";
import { createMockApi, problem } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

const A = verified.case_id;
const B = unverified.case_id;

describe("review screen", () => {
  it("renders a verified draft for the case", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/draft`]: verified,
    });
    await renderSignedIn(`/cases/${A}/review`, fetchImpl);
    expect(await screen.findByTestId("draft-view")).toBeInTheDocument();
    expect(screen.queryByTestId("unverified-banner")).not.toBeInTheDocument();
    expect(await screen.findByText(/· OD/)).toBeInTheDocument();
  });

  it("renders an unverified draft with the banner", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${B}`]: { ...pending, case_id: B, status: "NEEDS_ATTENTION", unverified: true },
      [`GET /v1/cases/${B}/draft`]: unverified,
    });
    await renderSignedIn(`/cases/${B}/review`, fetchImpl);
    expect(await screen.findByTestId("unverified-banner")).toHaveTextContent(
      "UNVERIFIED — critic did not approve",
    );
  });

  it("shows the server's reason when no draft exists yet", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/draft`]: () => problem(409, "DRAFT_NOT_AVAILABLE"),
    });
    await renderSignedIn(`/cases/${A}/review`, fetchImpl);
    expect(await screen.findByText(/DRAFT_NOT_AVAILABLE/)).toBeInTheDocument();
    expect(screen.queryByTestId("draft-view")).not.toBeInTheDocument();
  });

  it("shows the server's refusal for a non-clinician", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": { sub: "sub-1", roles: ["submitter"] },
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/draft`]: () => problem(403, "ROLE_CLINICIAN_REQUIRED", "forbidden"),
    });
    await renderSignedIn(`/cases/${A}/review`, fetchImpl);
    expect(await screen.findByText(/ROLE_CLINICIAN_REQUIRED/)).toBeInTheDocument();
  });

  it("composes overlay + draft + edit form; approve-with-edits lands on the report", async () => {
    const png = () => ({ blob: new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }) });
    const { fetchImpl, calls } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}`]: pending,
      [`GET /v1/cases/${A}/draft`]: verified,
      [`GET /v1/cases/${A}/overlay`]: png,
      "GET /v1/meta/reason-codes": reasonCodes,
      [`POST /v1/cases/${A}/review`]: () => ({ json: reviewResult }),
      [`GET /v1/cases/${A}/report`]: report,
    });
    const { user } = await renderSignedIn(`/cases/${A}/review`, fetchImpl);
    expect(await screen.findByTestId("overlay-viewer")).toBeInTheDocument();
    const box = await screen.findByLabelText(/Claim c2/);
    await user.clear(box);
    await user.type(box, report.report.claims[1]!.text);
    await user.click(await screen.findByRole("checkbox", { name: /TONE_OR_WORDING/ }));
    await user.click(screen.getByRole("button", { name: "Approve with edits" }));
    expect(await screen.findByTestId("screening-notice")).toBeInTheDocument();
    expect(screen.getByTestId("attribution")).toHaveTextContent(report.approved_by);
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(1);
  });
});
