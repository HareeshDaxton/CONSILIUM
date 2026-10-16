import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DraftView, ReviewCommand } from "../../api/types";
import unverifiedJson from "../../test/fixtures/draft_unverified.json";
import reasonCodes from "../../test/fixtures/reason_codes.json";
import reviewResult from "../../test/fixtures/review_result.json";
import verifiedJson from "../../test/fixtures/draft_verified.json";
import { createMockApi, problem } from "../../test/mockApi";
import type { Handler } from "../../test/mockApi";
import { renderWithApi } from "../../test/renderApp";
import { EditDiffForm } from "./EditDiffForm";

const verified = verifiedJson as unknown as DraftView;
const unverified = unverifiedJson as unknown as DraftView;

function setup(view: DraftView, reviewHandler: Handler = () => ({ json: reviewResult })) {
  const { fetchImpl, calls } = createMockApi({
    "GET /v1/meta/reason-codes": reasonCodes,
    [`POST /v1/cases/${view.case_id}/review`]: reviewHandler,
  });
  const onSubmitted = vi.fn();
  const r = renderWithApi(<EditDiffForm view={view} onSubmitted={onSubmitted} />, fetchImpl);
  const posted = () => calls.filter((c) => c.method === "POST").map((c) => c.body as ReviewCommand);
  return { ...r, onSubmitted, posted };
}

const btn = (name: string) => screen.getByRole("button", { name });

describe("EditDiffForm — edit and diff", () => {
  it("shows AI text beside the clinician edit for each changed field", async () => {
    const { user } = setup(verified);
    const box = screen.getByLabelText(/Claim c2/);
    await user.clear(box);
    await user.type(box, "Screening tier is intermediate.");
    const rows = screen.getAllByTestId("diff-row");
    expect(rows).toHaveLength(1);
    expect(within(rows[0]!).getByText("claims[id=c2].text")).toBeInTheDocument();
    expect(within(rows[0]!).getByText(verified.draft.claims[1]!.text)).toBeInTheDocument();
    expect(within(rows[0]!).getByText("Screening tier is intermediate.")).toBeInTheDocument();
    await user.click(btn("Reset all edits"));
    expect(screen.queryByTestId("diff-table")).not.toBeInTheDocument();
  });

  it("submits the FULL edited draft + base revision + reason codes", async () => {
    const { user, posted, onSubmitted } = setup(verified);
    const box = screen.getByLabelText(/Claim c2/);
    await user.clear(box);
    await user.type(box, "Screening tier is intermediate.");
    expect(btn("Approve")).toBeDisabled(); // edits present → plain approve would drop them
    expect(btn("Approve with edits")).toBeDisabled(); // no reason code yet
    await user.click(await screen.findByRole("checkbox", { name: /TONE_OR_WORDING/ }));
    await user.click(btn("Approve with edits"));
    await vi.waitFor(() => expect(onSubmitted).toHaveBeenCalledWith(reviewResult));
    const [cmd] = posted();
    expect(cmd!.action).toBe("APPROVE_WITH_EDITS");
    expect(cmd!.base_revision).toBe(verified.revision);
    expect(cmd!.reason_codes).toEqual(["TONE_OR_WORDING"]);
    expect(cmd!.edited_draft!.claims.map((c) => c.text)).toEqual([
      verified.draft.claims[0]!.text,
      "Screening tier is intermediate.",
      verified.draft.claims[2]!.text,
    ]);
    expect(cmd!.edited_draft!.claims[0]).toEqual(verified.draft.claims[0]); // untouched claim intact
    expect(cmd!.edited_draft!.limitations).toEqual(verified.draft.limitations);
    expect(cmd).not.toHaveProperty("status");
  });

  it("OTHER keeps edits unsubmittable until free text is given, then sends the comment", async () => {
    const { user, posted } = setup(verified);
    await user.type(screen.getByLabelText("Impression"), " Clinician addition.");
    await user.click(await screen.findByRole("checkbox", { name: /^OTHER/ }));
    expect(btn("Approve with edits")).toBeDisabled();
    await user.type(screen.getByLabelText(/required for OTHER/), "Local referral pathway wording");
    await user.click(btn("Approve with edits"));
    await vi.waitFor(() => expect(posted()).toHaveLength(1));
    expect(posted()[0]!.comment).toBe("Local referral pathway wording");
  });

  it("blocks blank text locally (schema requires non-empty text)", async () => {
    const { user } = setup(verified);
    await user.clear(screen.getByLabelText(/Claim c1/));
    await user.click(await screen.findByRole("checkbox", { name: /OMISSION/ }));
    expect(screen.getByText(/Empty text: claim c1/)).toBeInTheDocument();
    expect(btn("Approve with edits")).toBeDisabled();
  });
});

describe("EditDiffForm — actions", () => {
  it("plain approve sends no draft and no codes", async () => {
    const { user, posted } = setup(verified, () => ({ json: { ...reviewResult, action: "APPROVE" } }));
    await screen.findAllByRole("checkbox");
    await user.click(btn("Approve"));
    await vi.waitFor(() => expect(posted()).toHaveLength(1));
    expect(posted()[0]).toEqual({ action: "APPROVE", base_revision: verified.revision });
  });

  it("reject requires a code and never carries edits", async () => {
    const { user, posted } = setup(verified, () => ({ json: { ...reviewResult, action: "REJECT" } }));
    expect(btn("Reject")).toBeDisabled();
    await user.click(await screen.findByRole("checkbox", { name: /FACTUAL_ERROR_CAD/ }));
    await user.click(btn("Reject"));
    await vi.waitFor(() => expect(posted()).toHaveLength(1));
    expect(posted()[0]).toEqual({
      action: "REJECT",
      base_revision: verified.revision,
      reason_codes: ["FACTUAL_ERROR_CAD"],
    });
  });

  it("does not pre-empt the server on an UNVERIFIED draft: the 409 is shown verbatim", async () => {
    const { user, onSubmitted } = setup(unverified, () => problem(409, "UNVERIFIED_REQUIRES_EDITS"));
    await screen.findAllByRole("checkbox");
    expect(btn("Approve")).toBeEnabled(); // no status logic in the UI
    await user.click(btn("Approve"));
    expect(await screen.findByRole("alert")).toHaveTextContent("409 · UNVERIFIED_REQUIRES_EDITS");
    expect(onSubmitted).not.toHaveBeenCalled();
  });

  it("an edit that does not resolve the flagged claim is refused by the server", async () => {
    const { user, posted } = setup(unverified, () => problem(409, "UNVERIFIED_ISSUES_OPEN"));
    await user.type(screen.getByLabelText(/Claim c1/), " Extra.");
    await user.click(await screen.findByRole("checkbox", { name: /CLINICAL_INTERPRETATION/ }));
    await user.click(btn("Approve with edits"));
    expect(await screen.findByRole("alert")).toHaveTextContent("UNVERIFIED_ISSUES_OPEN");
    expect(posted()[0]!.base_revision).toBe(unverified.revision);
  });
});
