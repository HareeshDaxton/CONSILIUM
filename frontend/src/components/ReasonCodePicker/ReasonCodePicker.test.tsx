import { screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import reasonCodes from "../../test/fixtures/reason_codes.json";
import { createMockApi } from "../../test/mockApi";
import { renderWithApi } from "../../test/renderApp";
import { ReasonCodePicker } from "./ReasonCodePicker";

let last: string[] = [];

function Harness() {
  const [codes, setCodes] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  last = codes;
  return <ReasonCodePicker selected={codes} onChange={setCodes} comment={comment} onCommentChange={setComment} />;
}

describe("ReasonCodePicker", () => {
  it("renders exactly the server's closed taxonomy", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/meta/reason-codes": reasonCodes });
    renderWithApi(<Harness />, fetchImpl);
    const boxes = await screen.findAllByRole("checkbox");
    expect(boxes.map((b) => (b as HTMLInputElement).value)).toEqual(reasonCodes.map((r) => r.code));
  });

  it("multi-selects and toggles", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/meta/reason-codes": reasonCodes });
    const { user } = renderWithApi(<Harness />, fetchImpl);
    await user.click(await screen.findByRole("checkbox", { name: /KNOWLEDGE_GAP/ }));
    await user.click(screen.getByRole("checkbox", { name: /OMISSION/ }));
    expect(last).toEqual(["KNOWLEDGE_GAP", "OMISSION"]);
    await user.click(screen.getByRole("checkbox", { name: /KNOWLEDGE_GAP/ }));
    expect(last).toEqual(["OMISSION"]);
  });

  it("OTHER asks for free text until it is given", async () => {
    const { fetchImpl } = createMockApi({ "GET /v1/meta/reason-codes": reasonCodes });
    const { user } = renderWithApi(<Harness />, fetchImpl);
    await user.click(await screen.findByRole("checkbox", { name: /^OTHER/ }));
    expect(screen.getByText("OTHER requires a free-text reason.")).toBeInTheDocument();
    expect(screen.getByLabelText(/required for OTHER/)).toHaveAttribute("aria-invalid", "true");
    await user.type(screen.getByLabelText(/required for OTHER/), "Laterality mislabelled at intake");
    expect(screen.queryByText("OTHER requires a free-text reason.")).not.toBeInTheDocument();
  });
});
