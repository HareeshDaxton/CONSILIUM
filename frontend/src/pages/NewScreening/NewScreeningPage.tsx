import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { useSystemStatus } from "../../api/queries";
import type { Laterality } from "../../api/types";
import { useApi } from "../../auth/session";
import { ProblemMessage } from "../../components/ProblemMessage";
import { REPORT_ACCEPT, UnsupportedReport, extractReportText, reportKind } from "../../lib/extractText";
import { EASE, gsap, prefersReducedMotion } from "../../motion/gsap";

const NOTE_MAX = 4000; // matches the server's input-rail cap
const IMAGE_ACCEPT = "image/png,image/jpeg";
const EYES: { value: Laterality; label: string; short: string }[] = [
  { value: "OD", label: "Right eye (OD)", short: "Right" },
  { value: "OS", label: "Left eye (OS)", short: "Left" },
  { value: "UNKNOWN", label: "Not specified", short: "Not sure" },
];
const KIND_LABEL = { pdf: "PDF", docx: "Word document", image: "image", text: "text file" } as const;

/** Only one thing the user must know: reports cannot be made right now. */
function OfflineNotice() {
  const { data } = useSystemStatus();
  if (!data || (data.vision.reachable && data.vision.revision_ok && data.llm_configured)) return null;
  return (
    <p className="intake-notice" role="status" data-testid="system-notice">
      Report generation is temporarily unavailable. You can prepare the case; try again in a few minutes.
    </p>
  );
}

function useDrop(onFile: (f: File | undefined) => void) {
  const [over, setOver] = useState(false);
  return {
    over,
    props: {
      onDragOver: (e: DragEvent) => {
        e.preventDefault();
        setOver(true);
      },
      onDragLeave: () => setOver(false),
      onDrop: (e: DragEvent) => {
        e.preventDefault();
        setOver(false);
        onFile(e.dataTransfer.files[0]);
      },
    },
  };
}

// Main page: fundus photograph + optional doctor's report → Generate report.
export function NewScreeningPage() {
  const api = useApi();
  const navigate = useNavigate();
  const root = useRef<HTMLElement>(null);
  const imageInput = useRef<HTMLInputElement>(null);
  const reportInput = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [eye, setEye] = useState<Laterality>("UNKNOWN");
  const [imageError, setImageError] = useState<string | null>(null);

  const [note, setNote] = useState("");
  const [source, setSource] = useState<string | null>(null); // "Text from your PDF"
  const [reading, setReading] = useState<number | null>(null); // 0..1 while extracting
  const [reportError, setReportError] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useLayoutEffect(() => {
    const host = root.current;
    if (!host || prefersReducedMotion()) return undefined;
    const ctx = gsap.context(() => {
      gsap.from(".intake-head > *", { autoAlpha: 0, y: 26, duration: 1.1, ease: EASE.out, stagger: 0.08 });
      gsap.from(".intake-card", { autoAlpha: 0, y: 40, duration: 1.2, ease: EASE.out, stagger: 0.12, delay: 0.15 });
    }, host);
    return () => ctx.revert();
  }, []);

  function chooseImage(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    if (!IMAGE_ACCEPT.split(",").includes(f.type)) {
      setImageError("Please choose a PNG or JPEG fundus photograph.");
      return;
    }
    setImageError(null);
    setFile(f);
  }

  async function chooseReport(f: File | undefined | null) {
    if (!f) return;
    setReportError(null);
    const kind = reportKind(f);
    setReading(0);
    try {
      const text = await extractReportText(f, (p) => setReading(p));
      setNote((prev) => (prev.trim() ? `${prev.trim()}\n\n${text}` : text));
      setSource(`Text from your ${kind ? KIND_LABEL[kind] : "file"} — please check it before generating.`);
    } catch (err) {
      setReportError(
        err instanceof UnsupportedReport ? err.message : "This file could not be read. Paste the notes instead.",
      );
    } finally {
      setReading(null);
      if (reportInput.current) reportInput.current.value = "";
    }
  }

  const imageDrop = useDrop(chooseImage);
  const reportDrop = useDrop((f) => void chooseReport(f));
  const tooLong = note.length > NOTE_MAX;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || tooLong) return;
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
    <section className="intake" ref={root}>
      <header className="intake-head">
        <p className="kicker">New screening</p>
        <h1>
          Generate a glaucoma screening report
        </h1>
        <p className="intake-sub">
          Add the fundus photograph and, if you have one, the doctor&apos;s report. CONSILIUM prepares a draft for a
          clinician to review and sign.
        </p>
        <OfflineNotice />
      </header>

      <form className="intake-form" onSubmit={onSubmit} aria-label="New screening">
        <div className="intake-grid">
          {/* 1 — fundus photograph */}
          <div className="intake-card intake-image">
            <div className="intake-card-head">
              <span className="intake-step">01</span>
              <h2>Fundus photograph</h2>
            </div>
            <div
              className={`dropzone dropzone-image${imageDrop.over ? " is-over" : ""}${file ? " has-file" : ""}`}
              {...imageDrop.props}
              onClick={() => imageInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && imageInput.current?.click()}
              aria-label="Choose a fundus image"
              data-testid="dropzone"
            >
              {preview ? (
                <img src={preview} alt="Selected fundus image" className="fundus-preview" />
              ) : (
                <div className="dropzone-empty">
                  <span className="lens" aria-hidden="true">
                    <span />
                  </span>
                  <strong>Drop the photograph here</strong>
                  <span>or click to browse · PNG or JPEG</span>
                </div>
              )}
            </div>
            <input
              ref={imageInput}
              type="file"
              accept={IMAGE_ACCEPT}
              hidden
              data-testid="file-input"
              onChange={(e: ChangeEvent<HTMLInputElement>) => chooseImage(e.target.files?.[0])}
            />
            {imageError ? <p className="field-error">{imageError}</p> : null}
            <div className="intake-row">
              <div className="chips" role="radiogroup" aria-label="Eye">
                <span className="chips-label">Eye</span>
                {EYES.map((o) => (
                  <label key={o.value} className={eye === o.value ? "is-active" : ""}>
                    <input
                      type="radio"
                      name="eye"
                      value={o.value}
                      checked={eye === o.value}
                      onChange={() => setEye(o.value)}
                      aria-label={o.label}
                    />
                    {o.short}
                  </label>
                ))}
              </div>
              {file ? (
                <button type="button" className="link-button" onClick={() => setFile(null)}>
                  Replace photo
                </button>
              ) : null}
            </div>
          </div>

          {/* 2 — doctor's report (optional) */}
          <div className="intake-card intake-report">
            <div className="intake-card-head">
              <span className="intake-step">02</span>
              <h2>
                Doctor&apos;s report <span className="optional">optional</span>
              </h2>
            </div>
            <div
              className={`dropzone dropzone-report${reportDrop.over ? " is-over" : ""}${reading !== null ? " is-reading" : ""}`}
              {...reportDrop.props}
              onClick={() => reading === null && reportInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && reportInput.current?.click()}
              aria-label="Upload the doctor's report"
              data-testid="report-dropzone"
            >
              {reading !== null ? (
                <div className="reading" data-testid="report-reading">
                  <span className="reading-bar" style={{ transform: `scaleX(${Math.max(reading, 0.06)})` }} />
                  <span>Reading the report on this device…</span>
                </div>
              ) : (
                <div className="dropzone-inline">
                  <strong>Upload a report</strong>
                  <span>PDF, Word or a photo of the page</span>
                </div>
              )}
            </div>
            <input
              ref={reportInput}
              type="file"
              accept={REPORT_ACCEPT}
              hidden
              data-testid="report-input"
              onChange={(e: ChangeEvent<HTMLInputElement>) => void chooseReport(e.target.files?.[0])}
            />
            {reportError ? <p className="field-error">{reportError}</p> : null}

            <div className="or-divider" aria-hidden="true">
              <span>or paste the notes</span>
            </div>
            <textarea
              className="note-input"
              aria-label="Clinical note"
              rows={8}
              value={note}
              onChange={(e) => {
                setNote(e.target.value);
                if (!e.target.value) setSource(null);
              }}
              placeholder="e.g. 62-year-old, IOP right eye 24 mmHg, on latanoprost, asthma. Family history of glaucoma."
            />
            <div className="note-foot">
              {source ? <span className="note-source">{source}</span> : <span>Leave out names, birth dates and record numbers.</span>}
              <span className={tooLong ? "note-count is-over" : "note-count"}>
                {note.length}/{NOTE_MAX}
              </span>
            </div>
            {tooLong ? <p className="field-error">Shorten the notes to {NOTE_MAX} characters.</p> : null}
          </div>
        </div>

        <div className="intake-action">
          {error ? <ProblemMessage error={error} context="Could not start the screening" /> : null}
          <div className="intake-action-bar">
            <p>Screening support only — the report is a draft until a clinician signs it.</p>
            <button
              type="submit"
              className={`btn btn-primary btn-lg generate${busy ? " is-busy" : ""}`}
              disabled={!file || busy || tooLong || reading !== null}
            >
              {busy ? "Uploading…" : "Generate report"}
            </button>
          </div>
        </div>
      </form>
    </section>
  );
}
