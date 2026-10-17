import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useRecentCases, useSystemStatus } from "../../api/queries";
import type { Laterality } from "../../api/types";
import { useApi } from "../../auth/session";
import { ProblemMessage } from "../../components/ProblemMessage";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDateTime, shortId } from "../../lib/format";

const NOTE_MAX = 4000; // matches the server's input-rail cap
const ACCEPT = "image/png,image/jpeg";
const EYES: { value: Laterality; label: string }[] = [
  { value: "OD", label: "Right eye (OD)" },
  { value: "OS", label: "Left eye (OS)" },
  { value: "UNKNOWN", label: "Not specified" },
];

const PIPELINE = [
  ["Vision", "SwinV2 estimates glaucoma probability; SegFormer outlines the optic disc and cup (GPU server)."],
  ["Measurements", "Cup-to-disc ratio, risk tier and image-quality checks are computed in CONSILIUM's own code."],
  ["Note", "Your note is de-identified and safety-checked before any AI reads it."],
  ["Specialists", "Ophthalmologist, optometrist and pharmacist agents each see only the facts their role needs."],
  ["Audit", "The Director drafts the report; a verifier and an adversarial critic check every claim."],
  ["Report", "You get an evidence-linked draft to review, sign off and export as PDF or Word."],
] as const;

function VisionStatusPill() {
  const { data, error } = useSystemStatus();
  if (error) return <span className="pill pill-muted">Status unavailable</span>;
  if (!data) return <span className="pill pill-muted">Checking servers…</span>;
  const v = data.vision;
  const vision = !v.reachable
    ? { tone: "bad", text: "Vision server offline" }
    : !v.revision_ok
      ? { tone: "warn", text: "Vision server: model version mismatch" }
      : { tone: "ok", text: `Vision server online${v.device ? ` (${v.device === "cuda" ? "GPU" : v.device})` : ""}` };
  return (
    <div className="status-pills" data-testid="system-status">
      <span className={`pill pill-${vision.tone}`}>
        <span className="pill-dot" />
        {vision.text}
      </span>
      <span className={`pill pill-${data.llm_configured ? "ok" : "bad"}`}>
        <span className="pill-dot" />
        {data.llm_configured ? "AI agents ready" : "AI agents not configured"}
      </span>
    </div>
  );
}

// Main page: upload a fundus image (+ optional note) and generate the report.
export function NewScreeningPage() {
  const api = useApi();
  const navigate = useNavigate();
  const recent = useRecentCases();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [eye, setEye] = useState<Laterality>("UNKNOWN");
  const [note, setNote] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    if (!ACCEPT.split(",").includes(f.type)) {
      setLocalError("Please choose a PNG or JPEG fundus photograph.");
      return;
    }
    setLocalError(null);
    setFile(f);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    choose(e.dataTransfer.files[0]);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      // Generic upload name: the original filename (often a patient name/MRN)
      // never leaves the browser (I-6).
      const ext = file.type === "image/png" ? "png" : "jpg";
      const form = new FormData();
      form.append("image", new File([file], `fundus.${ext}`, { type: file.type }));
      form.append("laterality", eye);
      if (note.trim()) form.append("note", note.trim());
      const res = await api.submitCase(form, crypto.randomUUID());
      navigate(`/cases/${res.case_id}`);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="screening">
      <header className="page-head">
        <div>
          <p className="eyebrow">New screening</p>
          <h1>Generate a glaucoma screening report</h1>
          <p className="muted">
            Upload a retinal fundus photograph and, optionally, a clinical note. CONSILIUM returns an
            evidence-linked draft report for clinician review.
          </p>
        </div>
        <VisionStatusPill />
      </header>

      <div className="screening-grid">
        <form className="card upload-card" onSubmit={onSubmit} aria-label="New screening">
          <h2>1 · Fundus image</h2>
          <div
            className={`dropzone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => input.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            aria-label="Choose a fundus image"
            data-testid="dropzone"
          >
            {preview ? (
              <img src={preview} alt="Selected fundus image" className="dropzone-preview" />
            ) : (
              <div className="dropzone-empty">
                <span className="dropzone-icon" aria-hidden="true">
                  ◎
                </span>
                <strong>Drop a fundus photo here</strong>
                <span className="muted">or click to browse · PNG or JPEG · up to 20 MB</span>
              </div>
            )}
          </div>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            hidden
            data-testid="file-input"
            onChange={(e: ChangeEvent<HTMLInputElement>) => choose(e.target.files?.[0])}
          />
          {file ? (
            <p className="muted file-meta">
              {(file.size / 1024 / 1024).toFixed(2)} MB ·{" "}
              <button type="button" className="link-button" onClick={() => setFile(null)}>
                remove
              </button>
            </p>
          ) : null}
          {localError ? <p className="field-error">{localError}</p> : null}

          <h2>2 · Eye</h2>
          <div className="segmented" role="radiogroup" aria-label="Eye">
            {EYES.map((o) => (
              <label key={o.value} className={eye === o.value ? "is-active" : ""}>
                <input
                  type="radio"
                  name="eye"
                  value={o.value}
                  checked={eye === o.value}
                  onChange={() => setEye(o.value)}
                />
                {o.label}
              </label>
            ))}
          </div>

          <h2>
            3 · Clinical note <span className="muted optional">(optional)</span>
          </h2>
          <textarea
            className="note-input"
            aria-label="Clinical note"
            rows={6}
            maxLength={NOTE_MAX}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="e.g. 62-year-old, IOP OD 24 mmHg, on latanoprost, asthma. Family history of glaucoma."
          />
          <p className="hint">
            Do not include names, dates of birth or record numbers. The note is checked and de-identified before
            any AI reads it. {note.length}/{NOTE_MAX}
          </p>

          {error ? <ProblemMessage error={error} context="Could not start the screening" /> : null}

          <button type="submit" className="btn btn-primary btn-lg submit" disabled={!file || busy}>
            {busy ? "Uploading…" : "Generate report"}
          </button>
          <p className="hint">Screening support only — the report is a draft until a clinician signs it.</p>
        </form>

        <aside className="screening-side">
          <div className="card">
            <h2>How the report is made</h2>
            <ol className="pipeline-list">
              {PIPELINE.map(([title, text]) => (
                <li key={title}>
                  <strong>{title}</strong>
                  <span>{text}</span>
                </li>
              ))}
            </ol>
          </div>
          <div className="card">
            <h2>Recent screenings</h2>
            {recent.data && recent.data.items.length === 0 ? <p className="empty">None yet.</p> : null}
            <ul className="recent-list">
              {recent.data?.items.map((c) => (
                <li key={c.case_id}>
                  <Link to={`/cases/${c.case_id}`} className="mono">
                    {shortId(c.case_id)}
                  </Link>
                  <StatusBadge status={c.status} />
                  <span className="muted">{formatDateTime(c.created_at)}</span>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </section>
  );
}
