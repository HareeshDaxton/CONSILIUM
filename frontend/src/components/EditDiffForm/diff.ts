// Field-level edit model + client-side diff (UX only). The server recomputes
// the authoritative diff (review/diff.py) from the full edited draft and the
// base revision we submit; nothing here decides whether an edit is allowed.
//
// Editable: impression, claim text (or removal), recommendation text (or
// removal), limitations (edit, remove, add). Evidence refs, numeric values,
// claim kinds and disagreements are never edited here — CAD numbers stay as
// computed (I-1).

import type { DirectorDraft } from "../../api/types";

export interface EditableText {
  text: string;
  removed: boolean;
}

export interface EditState {
  impression: string;
  claims: Record<string, EditableText>;
  recommendations: EditableText[];
  limitations: EditableText[];
  addedLimitations: string[];
}

export type ChangeKind = "changed" | "removed" | "added";

export interface Change {
  path: string;
  kind: ChangeKind;
  before: string | null;
  after: string | null;
}

export function initialEditState(draft: DirectorDraft): EditState {
  return {
    impression: draft.impression,
    claims: Object.fromEntries(draft.claims.map((c) => [c.id, { text: c.text, removed: false }])),
    recommendations: draft.recommendations.map((r) => ({ text: r.text, removed: false })),
    limitations: draft.limitations.map((l) => ({ text: l, removed: false })),
    addedLimitations: [],
  };
}

/** The full edited draft to submit (order and all non-text fields preserved). */
export function buildEditedDraft(base: DirectorDraft, state: EditState): DirectorDraft {
  return {
    ...base,
    impression: state.impression,
    claims: base.claims
      .filter((c) => !state.claims[c.id]?.removed)
      .map((c) => ({ ...c, text: state.claims[c.id]?.text ?? c.text })),
    recommendations: base.recommendations
      .map((r, i) => ({ r, edit: state.recommendations[i] }))
      .filter(({ edit }) => !edit?.removed)
      .map(({ r, edit }) => ({ ...r, text: edit?.text ?? r.text })),
    limitations: [
      ...base.limitations
        .map((l, i) => ({ l, edit: state.limitations[i] }))
        .filter(({ edit }) => !edit?.removed)
        .map(({ l, edit }) => edit?.text ?? l),
      ...state.addedLimitations,
    ],
  };
}

/** Changes relative to the AI draft, with server-style paths (claims by id). */
export function diffDraft(base: DirectorDraft, state: EditState): Change[] {
  const changes: Change[] = [];
  const text = (path: string, before: string, edit: EditableText | undefined) => {
    if (!edit) return;
    if (edit.removed) changes.push({ path, kind: "removed", before, after: null });
    else if (edit.text !== before) changes.push({ path, kind: "changed", before, after: edit.text });
  };
  if (state.impression !== base.impression) {
    changes.push({ path: "impression", kind: "changed", before: base.impression, after: state.impression });
  }
  for (const c of base.claims) {
    const edit = state.claims[c.id];
    if (edit?.removed) changes.push({ path: `claims[id=${c.id}]`, kind: "removed", before: c.text, after: null });
    else text(`claims[id=${c.id}].text`, c.text, edit);
  }
  base.recommendations.forEach((r, i) => text(`recommendations[${i}].text`, r.text, state.recommendations[i]));
  base.limitations.forEach((l, i) => text(`limitations[${i}]`, l, state.limitations[i]));
  state.addedLimitations.forEach((l) => changes.push({ path: "limitations[+]", kind: "added", before: null, after: l }));
  return changes;
}

/** Kept text fields that are blank (the server's schema requires non-empty text). */
export function blankFields(state: EditState): string[] {
  const blank: string[] = [];
  if (!state.impression.trim()) blank.push("impression");
  for (const [id, e] of Object.entries(state.claims)) if (!e.removed && !e.text.trim()) blank.push(`claim ${id}`);
  state.recommendations.forEach((e, i) => {
    if (!e.removed && !e.text.trim()) blank.push(`recommendation ${i + 1}`);
  });
  state.limitations.forEach((e, i) => {
    if (!e.removed && !e.text.trim()) blank.push(`limitation ${i + 1}`);
  });
  state.addedLimitations.forEach((l, i) => {
    if (!l.trim()) blank.push(`new limitation ${i + 1}`);
  });
  return blank;
}
