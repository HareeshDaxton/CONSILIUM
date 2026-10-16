import { describe, expect, it, vi } from "vitest";

import { createMockApi, problem } from "../test/mockApi";
import { ApiError, createApiClient } from "./client";

function client(routes: Parameters<typeof createMockApi>[0], token: string | null = "tok") {
  const mock = createMockApi(routes);
  const onUnauthorized = vi.fn();
  const api = createApiClient({ getToken: () => token, onUnauthorized, fetchImpl: mock.fetchImpl });
  return { api, onUnauthorized, calls: mock.calls };
}

describe("api client", () => {
  it("sends the in-memory bearer token", async () => {
    const { api, calls } = client({ "GET /v1/cases/abc": { case_id: "abc" } });
    await api.getCase("abc");
    expect(calls[0]!.headers.get("Authorization")).toBe("Bearer tok");
  });

  it("surfaces problem+json reason codes as ApiError", async () => {
    const { api } = client({ "GET /v1/cases/abc/draft": () => problem(409, "DRAFT_NOT_AVAILABLE") });
    const err = await api.getDraft("abc").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
    expect((err as ApiError).reason).toBe("DRAFT_NOT_AVAILABLE");
    expect((err as ApiError).problem.type).toBe("urn:consilium:problem:conflict-status");
  });

  it("synthesises a problem for non-JSON errors", async () => {
    const { api } = client({
      "GET /v1/cases/abc": () =>
        new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }),
    });
    const err = (await api.getCase("abc").catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(502);
    expect(err.reason).toBe("Bad Gateway");
  });

  it("maps a network failure to NETWORK_ERROR", async () => {
    const api = createApiClient({
      getToken: () => "tok",
      fetchImpl: () => Promise.reject(new TypeError("offline")),
    });
    const err = (await api.getCase("abc").catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(0);
    expect(err.reason).toBe("NETWORK_ERROR");
  });

  it("ends the session on 401 for session calls, but not for a login probe", async () => {
    const { api, onUnauthorized } = client({
      "GET /v1/cases/abc": () => problem(401, "TOKEN_EXPIRED", "unauthorized"),
      "GET /v1/me": () => problem(401, "TOKEN_INVALID", "unauthorized"),
    });
    await expect(api.me("candidate")).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).not.toHaveBeenCalled();
    await expect(api.getCase("abc")).rejects.toBeInstanceOf(ApiError);
    expect(onUnauthorized).toHaveBeenCalledWith(expect.objectContaining({ reason: "TOKEN_EXPIRED" }));
  });

  it("probes /v1/me with the candidate token, not the session token", async () => {
    const { api, calls } = client({ "GET /v1/me": { sub: "clin-1", roles: ["clinician"] } }, null);
    await expect(api.me("candidate")).resolves.toEqual({ sub: "clin-1", roles: ["clinician"] });
    expect(calls[0]!.headers.get("Authorization")).toBe("Bearer candidate");
  });

  it("sends repeated status filters and a limit", async () => {
    const { api, calls } = client({ "GET /v1/cases": { items: [] } });
    await api.listCases(["PENDING_REVIEW", "NEEDS_ATTENTION"], 20);
    expect(calls[0]!.url.searchParams.getAll("status")).toEqual(["PENDING_REVIEW", "NEEDS_ATTENTION"]);
    expect(calls[0]!.url.searchParams.get("limit")).toBe("20");
  });

  it("returns the 202 body of an async submit with the Idempotency-Key", async () => {
    const accepted = { case_id: "c1", status: "INPUT_RAIL_PASSED", status_url: "/v1/cases/c1" };
    const { api, calls } = client({ "POST /v1/cases": () => ({ status: 202, json: accepted }) });
    await expect(api.submitCase(new FormData(), "key-1")).resolves.toEqual(accepted);
    expect(calls[0]!.headers.get("Idempotency-Key")).toBe("key-1");
  });
});
