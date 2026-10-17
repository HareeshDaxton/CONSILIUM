import { fireEvent, screen } from "@testing-library/react";
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

describe("new screening (main page)", () => {
  it("is where sign-in lands and shows the vision + AI status", async () => {
    const { fetchImpl } = createMockApi(routes());
    await renderSignedIn("/app", fetchImpl);
    expect(await screen.findByRole("heading", { name: /Generate a glaucoma screening report/ })).toBeInTheDocument();
    const pills = await screen.findByTestId("system-status");
    expect(pills).toHaveTextContent("Vision server online (GPU)");
    expect(pills).toHaveTextContent("AI agents ready");
    expect(screen.getByRole("button", { name: "Generate report" })).toBeDisabled();
  });

  it("warns when Deva's vision server is offline", async () => {
    const offline = { ...status, vision: { ...status.vision, reachable: false, revision_ok: false } };
    const { fetchImpl } = createMockApi(routes({ "GET /v1/meta/status": offline }));
    await renderSignedIn("/app", fetchImpl);
    expect(await screen.findByText("Vision server offline")).toBeInTheDocument();
  });

  it("uploads image + eye + note under a generic filename, then opens the result", async () => {
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
    await screen.findByTestId("system-status");
    pickFile();
    expect(await screen.findByAltText("Selected fundus image")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Left eye (OS)"));
    await user.type(screen.getByLabelText("Clinical note"), "IOP OD 24 mmHg, on latanoprost.");
    await user.click(screen.getByRole("button", { name: "Generate report" }));

    expect(await screen.findByTestId("progress")).toBeInTheDocument();
    const post = calls.find((c) => c.method === "POST")!;
    const form = post.body as FormData;
    expect((form.get("image") as File).name).toBe("fundus.png"); // original filename never sent
    expect(form.get("laterality")).toBe("OS");
    expect(form.get("note")).toBe("IOP OD 24 mmHg, on latanoprost.");
    expect(post.headers.get("Idempotency-Key")).toBeTruthy();
  });

  it("refuses non-image files locally", async () => {
    const { fetchImpl } = createMockApi(routes());
    await renderSignedIn("/app", fetchImpl);
    await screen.findByTestId("system-status");
    pickFile("notes.pdf", "application/pdf");
    expect(screen.getByText("Please choose a PNG or JPEG fundus photograph.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate report" })).toBeDisabled();
  });

  it("shows the server's rejection reason", async () => {
    const { fetchImpl } = createMockApi(
      routes({ "POST /v1/cases": () => problem(422, "IMAGE_TOO_SMALL", "input-rejected") }),
    );
    const { user } = await renderSignedIn("/app", fetchImpl);
    await screen.findByTestId("system-status");
    pickFile();
    await user.click(screen.getByRole("button", { name: "Generate report" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("422 · IMAGE_TOO_SMALL");
  });
});
