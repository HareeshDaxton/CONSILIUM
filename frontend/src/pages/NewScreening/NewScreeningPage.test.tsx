import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import pending from "../../test/fixtures/case_pending.json";
import me from "../../test/fixtures/me_clinician.json";
import status from "../../test/fixtures/system_status.json";
import { createMockApi, problem } from "../../test/mockApi";
import { renderSignedIn } from "../../test/renderApp";

const ME = { ...me, roles: ["clinician", "submitter"] };
const NEW_ID = pending.case_id;

function routes(extra: Record<string, unknown> = {}) {
  return {
    "GET /v1/me": ME,
    "GET /v1/meta/status": status,
    "GET /v1/cases": { items: [] },
    ...extra,
  };
}

function pickFile(name = "Jane_Doe_MRN12345.png", type = "image/png") {
  const file = new File([new Uint8Array([137, 80, 78, 71, 1, 2, 3])], name, { type });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
  return file;
}

function pickReport(text: string, name = "referral_Jane_Doe.txt", type = "text/plain") {
  const file = new File([text], name, { type });
  fireEvent.change(screen.getByTestId("report-input"), { target: { files: [file] } });
}

describe("new screening (main page)", () => {
  it("shows only the photo box, the doctor's report box and Generate report", async () => {
    const { fetchImpl } = createMockApi(routes());
    await renderSignedIn("/app", fetchImpl);
    expect(await screen.findByRole("heading", { name: /Generate a glaucoma screening report/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Fundus photograph" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Doctor's report/ })).toBeInTheDocument();
    expect(screen.getByLabelText("Clinical note")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate report" })).toBeDisabled();
    // No technical panels on the main page.
    expect(screen.queryByTestId("system-status")).not.toBeInTheDocument();
    expect(screen.queryByText(/SwinV2|SegFormer|Recent screenings/)).not.toBeInTheDocument();
  });

  it("shows a plain notice only when reports cannot be generated", async () => {
    const offline = { ...status, vision: { ...status.vision, reachable: false, revision_ok: false } };
    const { fetchImpl } = createMockApi(routes({ "GET /v1/meta/status": offline }));
    await renderSignedIn("/app", fetchImpl);
    expect(await screen.findByTestId("system-notice")).toHaveTextContent("temporarily unavailable");
  });

  it("fills the notes from an uploaded report, editable, without sending the file", async () => {
    const { fetchImpl, calls } = createMockApi(
      routes({
        "POST /v1/cases": () => ({
          status: 202,
          json: { case_id: NEW_ID, status: "INPUT_RAIL_PASSED", status_url: `/v1/cases/${NEW_ID}` },
        }),
        [`GET /v1/cases/${NEW_ID}`]: { ...pending, in_progress: true },
        [`GET /v1/cases/${NEW_ID}/timeline`]: [],
      }),
    );
    const { user } = await renderSignedIn("/app", fetchImpl);
    await screen.findByRole("heading", { name: "Fundus photograph" });
    pickFile();
    expect(await screen.findByAltText("Selected fundus image")).toBeInTheDocument();
    pickReport("IOP   right eye 24 mmHg.\n\n\n\nOn latanoprost.");
    await waitFor(() =>
      expect(screen.getByLabelText("Clinical note")).toHaveValue("IOP right eye 24 mmHg.\n\nOn latanoprost."),
    );
    expect(screen.getByText(/Text from your text file/)).toBeInTheDocument();
    await user.click(screen.getByLabelText("Left eye (OS)"));
    await user.click(screen.getByRole("button", { name: "Generate report" }));

    expect(await screen.findByTestId("progress")).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST")!;
    const form = post.body as FormData;
    expect((form.get("image") as File).name).toBe("fundus.png"); // original filename never sent
    expect(form.get("laterality")).toBe("OS");
    expect(form.get("note")).toBe("IOP right eye 24 mmHg.\n\nOn latanoprost.");
    expect([...form.keys()].sort()).toEqual(["image", "laterality", "note"]); // the report file is not uploaded
  });

  it("explains unsupported report files", async () => {
    const { fetchImpl } = createMockApi(routes());
    await renderSignedIn("/app", fetchImpl);
    await screen.findByRole("heading", { name: "Fundus photograph" });
    pickReport("x", "old.doc", "application/msword");
    expect(await screen.findByText(/save it as \.docx or PDF/)).toBeInTheDocument();
  });

  it("refuses non-image photographs locally", async () => {
    const { fetchImpl } = createMockApi(routes());
    await renderSignedIn("/app", fetchImpl);
    await screen.findByRole("heading", { name: "Fundus photograph" });
    pickFile("notes.pdf", "application/pdf");
    expect(screen.getByText("Please choose a PNG or JPEG fundus photograph.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate report" })).toBeDisabled();
  });

  it("shows the server's rejection reason", async () => {
    const { fetchImpl } = createMockApi(
      routes({ "POST /v1/cases": () => problem(422, "IMAGE_TOO_SMALL", "input-rejected") }),
    );
    const { user } = await renderSignedIn("/app", fetchImpl);
    await screen.findByRole("heading", { name: "Fundus photograph" });
    pickFile();
    await user.click(screen.getByRole("button", { name: "Generate report" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("422 · IMAGE_TOO_SMALL");
  });
});
