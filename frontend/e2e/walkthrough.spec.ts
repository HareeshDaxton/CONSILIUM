// Phase 12.4 — clinician walkthrough on the dev stack, in a real browser:
// list → open case → inspect overlay → edit → reason codes →
// approve-with-edits → approved report; reject path; unverified-draft
// fix-and-approve path; self-approval refused. Every state change goes
// through the UI → API; the database is never touched by hand. A request
// spy asserts the UI only ever POSTs review commands and never sends a status.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";
import type { Page, Request } from "@playwright/test";

interface StackInfo {
  tokens: Record<"clinician" | "clinician_2" | "submitter" | "submitter_clinician", string>;
  cases: Record<"approve" | "reject" | "unverified" | "failed", string>;
}

// Written by scripts/dev_ui_stack.py at startup (Playwright starts it first).
let cached: StackInfo | null = null;
function stack(): StackInfo {
  cached ??= JSON.parse(
    readFileSync(fileURLToPath(new URL("./.stack.json", import.meta.url)), "utf-8"),
  ) as StackInfo;
  return cached;
}
const short = (id: string) => id.slice(0, 8);
const caseLink = (page: Page, id: string) => page.getByRole("link", { name: short(id) });

const requests: Request[] = [];

test.describe.configure({ mode: "serial" });

test.beforeEach(({ page }) => {
  page.on("request", (r) => {
    if (new URL(r.url()).pathname.startsWith("/v1/")) requests.push(r);
  });
});

// The token lives in memory, so any full page load signs the user out. Open
// the target path, get sent to /login, sign in, land back on the target —
// then navigate in-app only (no page.goto after sign-in).
async function signInAt(page: Page, path: string, token: string, as: string) {
  await page.goto(path);
  await page.getByLabel("Bearer token").fill(token);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByTestId("whoami")).toContainText(as);
}

test("token lives in memory only: storage is clean and a reload signs out", async ({ page }) => {
  const { tokens } = stack();
  await signInAt(page, "/cases", tokens.clinician, "dev-clinician");
  const stored = await page.evaluate(() => [
    JSON.stringify({ ...localStorage }),
    JSON.stringify({ ...sessionStorage }),
    document.cookie,
  ]);
  for (const s of stored) expect(s).not.toContain(tokens.clinician);
  await page.reload();
  await expect(page.getByLabel("Bearer token")).toBeVisible();
});

test("queues: review queue lists the seeded cases; failed case shows its reason", async ({ page }) => {
  const { tokens, cases } = stack();
  await signInAt(page, "/cases", tokens.clinician, "dev-clinician");
  for (const name of ["approve", "reject", "unverified"] as const) {
    await expect(caseLink(page, cases[name])).toBeVisible();
  }
  const unverifiedRow = page.getByTestId("case-row").filter({ hasText: short(cases.unverified) });
  await expect(unverifiedRow).toContainText("UNVERIFIED");
  await expect(unverifiedRow).toContainText("NEEDS ATTENTION");
  await expect(caseLink(page, cases.failed)).toHaveCount(0); // not in the review queue

  await page.getByRole("button", { name: "Failed" }).click();
  await caseLink(page, cases.failed).click();
  await expect(page.getByTestId("failure")).toContainText("VISION_REVISION_MISMATCH");
});

test("self-approval is refused by the server", async ({ page }) => {
  const { tokens, cases } = stack();
  await signInAt(page, `/cases/${cases.approve}/review`, tokens.submitter_clinician, "dev-submitter");
  await expect(page.getByTestId("draft-view")).toBeVisible();
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "SELF_APPROVAL" })).toContainText("403");
});

test("list → case → overlay → edit → reason code → approve with edits → report", async ({ page }) => {
  const { tokens, cases } = stack();
  await signInAt(page, "/cases", tokens.clinician, "dev-clinician");
  await caseLink(page, cases.approve).click();

  // case detail: CAD summary, de-identified note facts, timeline
  await expect(page.getByTestId("cad-summary")).toContainText("Vertical CDR");
  await expect(page.getByText("note.age")).toBeVisible();
  await expect(page.getByTestId("timeline")).toContainText("node.persist_pending");
  await page.getByRole("link", { name: "Open review" }).click();

  // overlay: the three server-rendered layers actually decode in the browser
  for (const alt of ["Fundus image", "Optic disc outline", "Optic cup outline"]) {
    const img = page.getByAltText(alt);
    await expect.poll(() => img.evaluate((i) => (i as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  }
  await page.getByRole("checkbox", { name: /Disc outline/ }).uncheck();
  await expect(page.getByTestId("layer-disc")).toBeHidden();
  await page.getByRole("checkbox", { name: /Disc outline/ }).check();
  await expect(page.getByTestId("layer-disc")).toBeVisible();
  await page.getByRole("slider").fill("50");
  await expect(page.getByTestId("layer-cup")).toHaveCSS("opacity", "0.5");

  // edit → diff → reason code → approve with edits
  await expect(page.getByTestId("unverified-banner")).toHaveCount(0);
  const edited = "Screening tier is as computed from the calibrated probability.";
  await page.getByLabel(/Claim c2/).fill(edited);
  await expect(page.getByTestId("diff-row")).toHaveCount(1);
  await expect(page.getByTestId("diff-row")).toContainText("claims[id=c2].text");
  await page.getByRole("checkbox", { name: /TONE_OR_WORDING/ }).check();
  await page.getByRole("button", { name: "Approve with edits" }).click();

  // approved report: attribution, screening notice, edited text
  await expect(page).toHaveURL(new RegExp(`/cases/${cases.approve}/report$`));
  await expect(page.getByTestId("screening-notice")).toBeVisible();
  await expect(page.getByTestId("attribution")).toContainText("dev-clinician");
  await expect(page.getByTestId("attribution")).toContainText("TONE_OR_WORDING");
  await expect(page.getByTestId("status-badge")).toHaveText("APPROVED WITH EDITS");
  await expect(page.getByText("Version 1")).toBeVisible();
  await expect(page.getByText(edited)).toBeVisible();
});

test("reject path: reason code → REJECTED; no report; no further review", async ({ page }) => {
  const { tokens, cases } = stack();
  await signInAt(page, `/cases/${cases.reject}/review`, tokens.clinician, "dev-clinician");
  await expect(page.getByTestId("draft-view")).toBeVisible();
  await expect(page.getByRole("button", { name: "Reject" })).toBeDisabled();
  await page.getByRole("checkbox", { name: /FACTUAL_ERROR_CAD/ }).check();
  await page.getByRole("button", { name: "Reject" }).click();

  await expect(page).toHaveURL(new RegExp(`/cases/${cases.reject}$`));
  await expect(page.getByTestId("status-badge").first()).toHaveText("REJECTED");
  await expect(page.getByTestId("timeline")).toContainText("review.reject");

  await page.getByRole("link", { name: "Approved report" }).click();
  await expect(page.getByText(/REPORT_NOT_APPROVED/)).toBeVisible();

  await page.getByRole("link", { name: "← Case detail" }).click();
  await page.getByRole("link", { name: "Open review" }).click();
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "409" })).toContainText("CASE_NOT_REVIEWABLE");
});

test("unverified draft: banner → plain approve refused → wrong fix refused → real fix approved", async ({
  page,
}) => {
  const { tokens, cases } = stack();
  await signInAt(page, `/cases/${cases.unverified}/review`, tokens.clinician, "dev-clinician");
  await expect(page.getByTestId("unverified-banner")).toContainText("UNVERIFIED — critic did not approve");
  await expect(page.getByTestId("claim-c2")).toHaveAttribute("data-flagged", "true");
  await expect(page.getByTestId("claim-c1")).toHaveAttribute("data-flagged", "false");

  // 1) plain approve: the UI does not pre-empt; the server refuses
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByRole("alert").filter({ hasText: "409" })).toContainText("UNVERIFIED_REQUIRES_EDITS");

  // 2) editing the wrong claim leaves the flagged one open
  await page.getByLabel(/Claim c1/).fill("Vertical cup-to-disc ratio is as measured on this image.");
  await page.getByRole("checkbox", { name: /CLINICAL_INTERPRETATION/ }).check();
  await page.getByRole("button", { name: "Approve with edits" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "409" })).toContainText("UNVERIFIED_ISSUES_OPEN");

  // 3) fix the flagged claim (no stray numerals) → approved with edits
  await page.getByRole("button", { name: "Reset all edits" }).click();
  await page.getByLabel(/Claim c2/).fill("Screening tier is as computed; it is supported by the calibrated probability.");
  await page.getByRole("button", { name: "Approve with edits" }).click();
  await expect(page).toHaveURL(new RegExp(`/cases/${cases.unverified}/report$`));
  await expect(page.getByTestId("status-badge")).toHaveText("APPROVED WITH EDITS");
  await expect(page.getByTestId("attribution")).toContainText("CLINICAL_INTERPRETATION");
});

test("approved queue shows both approvals, seen by a second clinician", async ({ page }) => {
  const { tokens, cases } = stack();
  await signInAt(page, "/cases?queue=approved", tokens.clinician_2, "dev-clinician-2");
  for (const name of ["approve", "unverified"] as const) {
    await expect(caseLink(page, cases[name])).toBeVisible();
  }
  await expect(caseLink(page, cases.reject)).toHaveCount(0);
});

test.afterAll(() => {
  // No status logic on the wire: the UI reads with GET and only POSTs review
  // commands; no request body ever carries a status.
  const allowed = ["action", "base_revision", "edited_draft", "reason_codes", "comment"];
  const writes = requests.filter((r) => r.method() !== "GET");
  expect(writes.length).toBeGreaterThan(0);
  for (const r of writes) {
    expect(r.method()).toBe("POST");
    expect(new URL(r.url()).pathname).toMatch(/^\/v1\/cases\/[0-9a-f-]{36}\/review$/);
    const body = JSON.parse(r.postData() ?? "{}") as Record<string, unknown>;
    expect(body).not.toHaveProperty("status");
    expect(Object.keys(body).filter((k) => !allowed.includes(k))).toEqual([]);
  }
});
