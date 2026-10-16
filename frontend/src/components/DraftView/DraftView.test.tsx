import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DraftView as DraftViewData } from "../../api/types";
import unverifiedJson from "../../test/fixtures/draft_unverified.json";
import verifiedJson from "../../test/fixtures/draft_verified.json";
import { DraftView } from "./DraftView";

const verified = verifiedJson as unknown as DraftViewData;
const unverified = unverifiedJson as unknown as DraftViewData;

describe("DraftView — verified draft", () => {
  it("renders claims as blocks with evidence chips and no banner", () => {
    render(<DraftView view={verified} />);
    expect(screen.queryByTestId("unverified-banner")).not.toBeInTheDocument();
    expect(screen.getByText(verified.draft.impression)).toBeInTheDocument();
    for (const claim of verified.draft.claims) {
      const block = screen.getByTestId(`claim-${claim.id}`);
      expect(block).toHaveAttribute("data-flagged", "false");
      expect(within(block).getByText(claim.text)).toBeInTheDocument();
    }
    const cdr = verified.evidence["cad.cdr.vertical"]!;
    expect(within(screen.getByTestId("claim-c1")).getByTestId("chip-cad.cdr.vertical")).toHaveTextContent(
      `cad.cdr.vertical = ${String(cdr.value)}`,
    );
    expect(screen.queryByTestId("disagreements")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("recommendation")).toHaveLength(verified.draft.recommendations.length);
    expect(screen.getByTestId("limitations")).toHaveTextContent("Screening support only");
  });
});

describe("DraftView — unverified draft", () => {
  it("shows the red UNVERIFIED banner with the server text", () => {
    render(<DraftView view={unverified} />);
    const banner = screen.getByTestId("unverified-banner");
    expect(banner).toHaveAttribute("role", "alert");
    expect(banner).toHaveTextContent(unverified.unverified_banner!);
    expect(banner).toHaveTextContent(`${unverified.open_issues.length} open issues`);
  });

  it("highlights exactly the claims named by open issues", () => {
    render(<DraftView view={unverified} />);
    const flagged = new Set(unverified.open_issues.map((i) => i.claim_id).filter(Boolean));
    for (const claim of unverified.draft.claims) {
      expect(screen.getByTestId(`claim-${claim.id}`)).toHaveAttribute(
        "data-flagged",
        String(flagged.has(claim.id)),
      );
    }
    expect(screen.getByTestId("issues-c2")).toHaveTextContent(
      "tier claim needs the calibrated probability as evidence",
    );
  });

  it("lists claim-less issues at draft level", () => {
    render(<DraftView view={unverified} />);
    expect(screen.getByTestId("issues-draft")).toHaveTextContent("image-quality score is not mentioned");
  });

  it("renders every disagreement position with its role", () => {
    render(<DraftView view={unverified} />);
    const section = screen.getByTestId("disagreements");
    const d = unverified.draft.disagreements[0]!;
    expect(section).toHaveTextContent(d.topic);
    for (const [role, position] of d.positions) {
      expect(section).toHaveTextContent(role);
      expect(section).toHaveTextContent(position);
    }
    expect(section).toHaveTextContent(`Resolution: ${d.resolution}`);
  });
});

describe("DraftView — safety of rendering", () => {
  it("renders draft text as plain text, never markup", () => {
    const hostile = structuredClone(verified);
    hostile.draft.claims[0]!.text = '<img src=x onerror="alert(1)"> claim';
    const { container } = render(<DraftView view={hostile} />);
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText('<img src=x onerror="alert(1)"> claim')).toBeInTheDocument();
  });

  it("issues naming an unknown claim fall back to draft level, not silently dropped", () => {
    const view = structuredClone(unverified);
    view.open_issues = [{ ...view.open_issues[0]!, claim_id: "c99" }];
    render(<DraftView view={view} />);
    expect(screen.getByTestId("issues-draft")).toHaveTextContent("tier claim needs");
  });

  it("marks a cited ref missing from the evidence map", () => {
    const view = structuredClone(verified);
    view.evidence = {};
    render(<DraftView view={view} />);
    expect(within(screen.getByTestId("claim-c1")).getByText(/not in fact sheet/)).toBeInTheDocument();
  });
});
