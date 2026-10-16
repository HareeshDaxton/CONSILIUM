// Offline fake of the CONSILIUM API for component tests: routes keyed by
// "METHOD /path" (query string ignored for matching; available to handlers).
// No network — every unmatched request is a recorded 404 problem.

export interface MockRequest {
  method: string;
  url: URL;
  headers: Headers;
  body: unknown;
}

export type MockReply = Response | { status?: number; json?: unknown; blob?: Blob };
export type Handler = (req: MockRequest) => MockReply;

export function problem(status: number, reason: string, type = "conflict-status"): Response {
  return new Response(
    JSON.stringify({ type: `urn:consilium:problem:${type}`, title: "Error", status, reason }),
    { status, headers: { "content-type": "application/problem+json" } },
  );
}

function toResponse(reply: MockReply): Response {
  if (reply instanceof Response) return reply;
  if (reply.blob) return new Response(reply.blob, { status: reply.status ?? 200 });
  return new Response(JSON.stringify(reply.json ?? null), {
    status: reply.status ?? 200,
    headers: { "content-type": "application/json" },
  });
}

export function createMockApi(routes: Record<string, Handler | unknown>) {
  const calls: MockRequest[] = [];
  const fetchImpl = async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://ui.test");
    const method = (init.method ?? "GET").toUpperCase();
    let body: unknown = init.body ?? null;
    if (typeof body === "string") {
      try {
        body = JSON.parse(body);
      } catch {
        // keep raw string
      }
    }
    const req: MockRequest = { method, url, headers: new Headers(init.headers), body };
    calls.push(req);
    const route = routes[`${method} ${url.pathname}`];
    if (route === undefined) return problem(404, "NOT_MOCKED", "not-found");
    return toResponse(typeof route === "function" ? (route as Handler)(req) : { json: route });
  };
  return { fetchImpl: fetchImpl as typeof fetch, calls };
}
