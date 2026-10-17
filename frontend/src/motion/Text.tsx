import { Fragment } from "react";

/** Headline split into masked lines for `data-reveal="mask"`. The accessible
 * name stays the full sentence (lines are presentational spans). */
export function MaskLines({ lines }: { lines: readonly React.ReactNode[] }) {
  return (
    <>
      {lines.map((line, i) => (
        <span className="line" key={i}>
          <span>{line}</span>
          {i < lines.length - 1 ? " " : null}
        </span>
      ))}
    </>
  );
}

/** Paragraph split into words for the scrubbed `data-words` effect. */
export function Words({ text }: { text: string }) {
  return (
    <>
      {text.split(" ").map((w, i) => (
        <Fragment key={i}>
          <span className="word">{w}</span>{" "}
        </Fragment>
      ))}
    </>
  );
}
