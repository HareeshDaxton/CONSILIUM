import { describe, expect, it } from "vitest";

import type { DirectorDraft, DraftView } from "../../api/types";
import verifiedJson from "../../test/fixtures/draft_verified.json";
import { blankFields, buildEditedDraft, diffDraft, initialEditState } from "./diff";

const base = (verifiedJson as unknown as DraftView).draft;

describe("edit model + client diff", () => {
  it("no edits → no changes, and the rebuilt draft equals the AI draft", () => {
    const state = initialEditState(base);
    expect(diffDraft(base, state)).toEqual([]);
    expect(buildEditedDraft(base, state)).toEqual(base);
  });

  it("claim edits use id paths and keep every non-text field", () => {
    const state = initialEditState(base);
    state.claims.c1 = { text: "Vertical cup-to-disc ratio as measured on this image.", removed: false };
    expect(diffDraft(base, state)).toEqual([
      {
        path: "claims[id=c1].text",
        kind: "changed",
        before: base.claims[0]!.text,
        after: "Vertical cup-to-disc ratio as measured on this image.",
      },
    ]);
    const edited = buildEditedDraft(base, state);
    const { text: _t, ...rest } = edited.claims[0]!;
    const { text: _b, ...baseRest } = base.claims[0]!;
    expect(rest).toEqual(baseRest); // evidence_refs, numeric_value/ref, kind, roles untouched
    expect(edited.claims.slice(1)).toEqual(base.claims.slice(1));
  });

  it("removing a claim drops it from the submitted draft", () => {
    const state = initialEditState(base);
    state.claims.c2 = { ...state.claims.c2!, removed: true };
    expect(diffDraft(base, state)).toEqual([
      { path: "claims[id=c2]", kind: "removed", before: base.claims[1]!.text, after: null },
    ]);
    expect(buildEditedDraft(base, state).claims.map((c) => c.id)).toEqual(["c1", "c3"]);
  });

  it("impression, recommendation and limitation edits; added limitations append", () => {
    const state = initialEditState(base);
    state.impression = "Findings raise suspicion of glaucoma and warrant evaluation.";
    state.recommendations[0] = { ...state.recommendations[0]!, removed: true };
    state.limitations[0] = { text: "Screening support only.", removed: false };
    state.addedLimitations.push("No visual field or OCT available.");
    const changes = diffDraft(base, state);
    expect(changes.map((c) => [c.path, c.kind])).toEqual([
      ["impression", "changed"],
      ["recommendations[0].text", "removed"],
      ["limitations[0]", "changed"],
      ["limitations[+]", "added"],
    ]);
    const edited: DirectorDraft = buildEditedDraft(base, state);
    expect(edited.recommendations).toEqual([]);
    expect(edited.limitations).toEqual([
      "Screening support only.",
      ...base.limitations.slice(1),
      "No visual field or OCT available.",
    ]);
    expect(edited.revision).toBe(base.revision);
    expect(edited.disagreements).toEqual(base.disagreements);
  });

  it("flags blank kept fields but ignores blank removed ones", () => {
    const state = initialEditState(base);
    state.claims.c1 = { text: "  ", removed: false };
    state.claims.c2 = { text: "", removed: true };
    state.addedLimitations.push("");
    expect(blankFields(state)).toEqual(["claim c1", "new limitation 1"]);
  });
});
