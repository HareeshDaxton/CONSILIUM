import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import me from "../../test/fixtures/me_clinician.json";
import report from "../../test/fixtures/report.json";
import { createMockApi, problem } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

const A = report.case_id;

describe("approved report", () => {
  it("shows clinician attribution, version and the screening notice", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/me": me, [`GET /v1/cases/${A}/report`]: report });
    await renderSignedIn(`/cases/${A}/report`, fetchImpl);
    expect(await screen.findByTestId("screening-notice")).toHaveTextContent(report.screening_notice);
    const attribution = screen.getByTestId("attribution");
    expect(attribution).toHaveTextContent(report.approved_by);
    expect(attribution).toHaveTextContent("APPROVE_WITH_EDITS");
    expect(attribution).toHaveTextContent("TONE_OR_WORDING");
    expect(screen.getByText(`Version ${report.version}`)).toBeInTheDocument();
    expect(screen.getByText(report.report.claims[1]!.text)).toBeInTheDocument();
  });

  it("shows the server's refusal before approval", async () => {
    const { fetchImpl } = createMockApi({
      "GET /v1/me": me,
      [`GET /v1/cases/${A}/report`]: () => problem(409, "REPORT_NOT_APPROVED"),
    });
    await renderSignedIn(`/cases/${A}/report`, fetchImpl);
    expect(await screen.findByText(/REPORT_NOT_APPROVED/)).toBeInTheDocument();
    expect(screen.queryByTestId("report")).not.toBeInTheDocument();
  });
});
