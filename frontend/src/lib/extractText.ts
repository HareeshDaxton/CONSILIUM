// Doctor's-report text extraction, entirely in the browser.
//
// Privacy (I-6/I-14): the report FILE and its filename never leave this page.
// Only the extracted text is placed in the note field, where the clinician
// reads it, can edit it, and submits it through the same input rail and
// de-identification as a typed note. Parsers are lazy-loaded so they never
// weigh on the first page load. OCR runs locally (tesseract.js downloads its
// engine and English language data on first use; no image is uploaded).

export type ReportKind = "pdf" | "docx" | "image" | "text";

export const REPORT_ACCEPT =
  ".pdf,.docx,.png,.jpg,.jpeg,.txt,application/pdf,image/png,image/jpeg,text/plain," +
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export class UnsupportedReport extends Error {}

export function reportKind(file: File): ReportKind | null {
  const name = file.name.toLowerCase();
  if (file.type === "application/pdf" || name.endsWith(".pdf")) return "pdf";
  if (
    file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    name.endsWith(".docx")
  )
    return "docx";
  if (file.type === "image/png" || file.type === "image/jpeg" || /\.(png|jpe?g)$/.test(name)) return "image";
  if (file.type === "text/plain" || name.endsWith(".txt")) return "text";
  return null;
}

/** Collapse layout whitespace into readable lines. */
export function tidy(text: string): string {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t\f\v]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function readAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.readAsText(file);
  });
}

async function fromPdf(file: File): Promise<string> {
  const pdfjs = await import("pdfjs-dist");
  const worker = await import("pdfjs-dist/build/pdf.worker.min.mjs?url");
  pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i += 1) {
    const content = await (await doc.getPage(i)).getTextContent();
    pages.push(
      content.items.map((item) => ("str" in item ? item.str + (item.hasEOL ? "\n" : " ") : "")).join(""),
    );
  }
  await doc.destroy();
  return pages.join("\n\n");
}

async function fromDocx(file: File): Promise<string> {
  const mammoth = await import("mammoth");
  const result = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
  return result.value;
}

async function fromImage(file: File, onProgress?: (p: number) => void): Promise<string> {
  const { createWorker } = await import("tesseract.js");
  const worker = await createWorker("eng", 1, {
    // tesseract's own job phase (not a case status): report OCR progress only.
    logger: ({ status: phase, progress }: { status: string; progress: number }) => {
      if (phase === "recognizing text") onProgress?.(progress);
    },
  });
  try {
    const { data } = await worker.recognize(file);
    return data.text;
  } finally {
    await worker.terminate();
  }
}

export async function extractReportText(file: File, onProgress?: (p: number) => void): Promise<string> {
  const kind = reportKind(file);
  if (kind === null) {
    throw new UnsupportedReport(
      file.name.toLowerCase().endsWith(".doc")
        ? "Old .doc files can't be read here — save it as .docx or PDF, or paste the text."
        : "Use a PDF, Word (.docx), PNG or JPEG file — or paste the text.",
    );
  }
  const raw =
    kind === "pdf"
      ? await fromPdf(file)
      : kind === "docx"
        ? await fromDocx(file)
        : kind === "image"
          ? await fromImage(file, onProgress)
          : await readAsText(file);
  const text = tidy(raw);
  if (!text) {
    throw new UnsupportedReport(
      kind === "pdf"
        ? "This PDF has no readable text (it may be a scan). Upload a photo of the page instead, or paste the text."
        : "No readable text was found in this file. Paste the notes instead.",
    );
  }
  return text;
}
