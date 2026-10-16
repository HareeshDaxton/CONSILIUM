// Shown whenever the server flags the draft UNVERIFIED (critic loop
// exhausted, AGENTS.md §4). The text is the server's; the UI adds nothing that
// could soften it.
export function UnverifiedBanner({ text, openIssues }: { text: string; openIssues: number }) {
  return (
    <div className="banner-unverified" role="alert" data-testid="unverified-banner">
      <strong>{text}</strong>
      <span>
        {openIssues} open issue{openIssues === 1 ? "" : "s"} highlighted below. This draft cannot become
        final without clinician fixes that resolve every flagged claim.
      </span>
    </div>
  );
}
