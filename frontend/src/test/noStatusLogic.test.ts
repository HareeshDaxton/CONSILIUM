// Phase 12.4 exit: "no status logic anywhere in the frontend". The review
// service owns every status and transition (SKILLS.md clinician-review: "the
// React UI calls this service and never sets status itself"). This static
// guard scans every non-test source file so a regression fails CI instead of
// depending on a reviewer's eye.

import { describe, expect, it } from "vitest";

const sources = import.meta.glob(["../**/*.{ts,tsx}", "!../**/*.test.{ts,tsx}", "!./**"], {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

// Code only: comments may (and do) explain what the code must NOT do.
const stripComments = (text: string) =>
  text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

const files = Object.entries(sources).map(([path, text]) => ({
  path: path.replace(/^\.\.\//, ""),
  text: stripComments(text),
}));

const CASE_STATUSES = [
  "RECEIVED", "IMAGE_PRECHECKED", "REJECTED_INPUT", "INPUT_RAIL_PASSED", "NOTE_EXTRACTED",
  "VISION_CALLED", "CAD_COMPLETE", "CONTEXT_BUILT", "ROUTED", "SPECIALISTS_DONE", "DRAFTED",
  "AUDITING", "REVISING", "OUTPUT_RAIL_PASSED", "PENDING_REVIEW", "APPROVED",
  "APPROVED_WITH_EDITS", "REJECTED", "FAILED", "NEEDS_ATTENTION",
];
// The only places a status value may be spelled out: the wire type, the
// display mapping, and the read-only queue filters.
const STATUS_LITERAL_ALLOWED = ["api/types.ts", "components/StatusBadge.tsx", "pages/CaseList/filters.ts"];

const literal = new RegExp(`["'\`](${CASE_STATUSES.join("|")})["'\`]`);

describe("frontend has no status logic", () => {
  it("scans the real source tree", () => {
    expect(files.map((f) => f.path)).toEqual(
      expect.arrayContaining(["api/client.ts", "components/EditDiffForm/EditDiffForm.tsx", "App.tsx"]),
    );
  });

  it("case-status values appear only in the type, the badge and the queue filters", () => {
    const offenders = files
      .filter((f) => !STATUS_LITERAL_ALLOWED.includes(f.path))
      .filter((f) => literal.test(f.text))
      .map((f) => f.path);
    expect(offenders).toEqual([]);
  });

  it("nothing compares a status to a literal (no client-side status branching)", () => {
    const cmp = /status\s*[!=]==?\s*["'`]|["'`]\s*[!=]==?\s*[\w.?]*status\b/; // incl. case_status
    expect(files.filter((f) => cmp.test(f.text)).map((f) => f.path)).toEqual([]);
  });

  it("review actions are enabled from form state only — never status or the unverified flag", () => {
    const form = files.find((f) => f.path === "components/EditDiffForm/EditDiffForm.tsx")!;
    expect(form.text).not.toMatch(/case_status|\.unverified\b|unverified_banner|\.status\b/);
  });

  it("the only writes are POSTs from the API client; no PUT/PATCH/DELETE", () => {
    expect(files.filter((f) => /method:\s*["'](PUT|PATCH|DELETE)["']/.test(f.text)).map((f) => f.path)).toEqual([]);
    expect(files.filter((f) => /method:\s*["']POST["']/.test(f.text)).map((f) => f.path)).toEqual(["api/client.ts"]);
  });

  it("no browser persistence of the session and no raw-HTML rendering", () => {
    const bad = /localStorage|sessionStorage|document\.cookie|indexedDB|dangerouslySetInnerHTML|innerHTML\s*=/;
    expect(files.filter((f) => bad.test(f.text)).map((f) => f.path)).toEqual([]);
  });
});
