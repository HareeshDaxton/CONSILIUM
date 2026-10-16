import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createMockApi, problem } from "../../test/mockApi";
import { renderWithApi } from "../../test/renderApp";
import { MaskOverlayViewer } from "./MaskOverlayViewer";

const A = "3f9a1c2e-7b4d-4e8a-9c1f-2d6b8e0a4c11";
const png = () => ({ blob: new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }) });

describe("MaskOverlayViewer", () => {
  it("stacks the three server-rendered layers", async () => {
    const { fetchImpl, calls } = createMockApi({ [`GET /v1/cases/${A}/overlay`]: png });
    renderWithApi(<MaskOverlayViewer caseId={A} />, fetchImpl);
    expect(await screen.findByAltText("Fundus image")).toBeInTheDocument();
    expect(await screen.findByTestId("layer-disc")).toBeVisible();
    expect(await screen.findByTestId("layer-cup")).toBeVisible();
    const layers = calls.map((c) => c.url.searchParams.get("layer")).sort();
    expect(layers).toEqual(["cup", "disc", "image"]);
  });

  it("toggles disc and cup independently", async () => {
    const { fetchImpl } = createMockApi({ [`GET /v1/cases/${A}/overlay`]: png });
    const { user } = renderWithApi(<MaskOverlayViewer caseId={A} />, fetchImpl);
    const disc = await screen.findByTestId("layer-disc");
    const cup = await screen.findByTestId("layer-cup");
    await user.click(screen.getByRole("checkbox", { name: /Disc outline/ }));
    expect(disc).not.toBeVisible();
    expect(cup).toBeVisible();
    await user.click(screen.getByRole("checkbox", { name: /Cup outline/ }));
    expect(cup).not.toBeVisible();
    await user.click(screen.getByRole("checkbox", { name: /Disc outline/ }));
    expect(disc).toBeVisible();
  });

  it("the opacity slider fades only the outline layers", async () => {
    const { fetchImpl } = createMockApi({ [`GET /v1/cases/${A}/overlay`]: png });
    renderWithApi(<MaskOverlayViewer caseId={A} />, fetchImpl);
    const disc = await screen.findByTestId("layer-disc");
    const base = await screen.findByAltText("Fundus image");
    fireEvent.change(screen.getByRole("slider"), { target: { value: "40" } });
    expect(disc).toHaveStyle({ opacity: "0.4" });
    expect(await screen.findByTestId("layer-cup")).toHaveStyle({ opacity: "0.4" });
    expect(base.getAttribute("style")).toBeNull();
    expect(screen.getByText("40%")).toBeInTheDocument();
  });

  it("does no client-side mask work (no canvas) and surfaces server refusals", async () => {
    const { fetchImpl } = createMockApi({
      [`GET /v1/cases/${A}/overlay`]: () => problem(409, "OVERLAY_NOT_AVAILABLE"),
    });
    const { container } = renderWithApi(<MaskOverlayViewer caseId={A} />, fetchImpl);
    expect(await screen.findByText(/OVERLAY_NOT_AVAILABLE/)).toBeInTheDocument();
    expect(container.querySelector("canvas")).toBeNull();
  });
});
