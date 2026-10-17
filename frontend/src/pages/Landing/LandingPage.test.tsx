import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import me from "../../test/fixtures/me_clinician.json";
import caseList from "../../test/fixtures/case_list.json";
import { createMockApi } from "../../test/mockApi";
import { renderApp, renderSignedIn } from "../../test/renderApp";

// WebGL is not available in jsdom; the 3D scene is covered by formations.test.
vi.mock("./LandingScene", () => ({ default: () => <div data-testid="landing-scene" /> }));

describe("landing page", () => {
  it("tells the pipeline story in six chapters with a sign-in call to action", async () => {
    const { fetchImpl } = createMockApi({});
    renderApp("/", fetchImpl);
    expect(
      await screen.findByRole("heading", { level: 1, name: /Screening support that shows its work/ }),
    ).toBeInTheDocument();
    expect(await screen.findByTestId("landing-scene")).toBeInTheDocument();
    for (const id of ["top", "vision", "isolation", "audit", "clinician", "start"]) {
      expect(document.getElementById(id)).not.toBeNull();
    }
    const rail = screen.getByRole("navigation", { name: "Story progress" });
    expect(within(rail).getAllByRole("button")).toHaveLength(6);
    for (const link of screen.getAllByRole("link", { name: "Sign in with Google" })) {
      expect(link).toHaveAttribute("href", "/login");
    }
  });

  it("keeps to screening-support language and makes no performance claims", async () => {
    const { fetchImpl } = createMockApi({});
    const { container } = renderApp("/", fetchImpl);
    await screen.findByTestId("landing-scene");
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/\d+(\.\d+)?\s?%/); // no accuracy/sensitivity figures (I-11)
    expect(text).not.toMatch(/\bdiagnoses\b|\bdetects glaucoma\b/i); // not a diagnosis (I-12)
    expect(text).toMatch(/does not diagnose/);
  });

  it("offers the app instead of sign-in once signed in", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/me": me, "GET /v1/cases": caseList });
    const { user } = await renderSignedIn("/cases", fetchImpl);
    await screen.findAllByTestId("case-row");
    await user.click(screen.getAllByRole("link", { name: "CONSILIUM home" })[0]!);
    expect(await screen.findAllByRole("link", { name: "Start a screening" })).not.toHaveLength(0);
    expect(screen.getByRole("link", { name: "Open CONSILIUM" })).toHaveAttribute("href", "/app");
    expect(screen.queryByRole("link", { name: "Sign in with Google" })).not.toBeInTheDocument();
  });
});
