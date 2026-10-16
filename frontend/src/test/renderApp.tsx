import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { ReactElement } from "react";

import { AppRoutes, makeQueryClient } from "../App";
import { SessionProvider } from "../auth/session";

export const TOKEN = "header.payload.signature-synthetic";

export function renderApp(path: string, fetchImpl: typeof fetch) {
  const user = userEvent.setup();
  const utils = render(
    <QueryClientProvider client={makeQueryClient()}>
      <SessionProvider fetchImpl={fetchImpl}>
        <MemoryRouter initialEntries={[path]}>
          <AppRoutes />
        </MemoryRouter>
      </SessionProvider>
    </QueryClientProvider>,
  );
  return { user, ...utils };
}

// Signs in through the real Login page (RequireAuth sends the user there and
// back to the requested path afterwards).
export async function renderSignedIn(path: string, fetchImpl: typeof fetch) {
  const r = renderApp(path, fetchImpl);
  await r.user.type(await screen.findByLabelText("Bearer token"), TOKEN);
  await r.user.click(screen.getByRole("button", { name: "Sign in" }));
  return r;
}

// Component-level render with the real query client + session/API context
// (no sign-in: the mock API does not check tokens).
export function renderWithApi(ui: ReactElement, fetchImpl: typeof fetch) {
  const user = userEvent.setup();
  const utils = render(
    <QueryClientProvider client={makeQueryClient()}>
      <SessionProvider fetchImpl={fetchImpl}>
        <MemoryRouter>{ui}</MemoryRouter>
      </SessionProvider>
    </QueryClientProvider>,
  );
  return { user, ...utils };
}
