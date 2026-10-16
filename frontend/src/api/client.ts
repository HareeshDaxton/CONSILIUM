// Typed fetch wrapper for the CONSILIUM API (ARCHITECTURE.md §5.1).
//
// - Bearer token comes from the in-memory session (never localStorage).
// - Errors are RFC 7807 problem+json; they surface as ApiError carrying the
//   server's short `reason` code. The UI shows the reason; it never decides
//   on its own whether an action is allowed — the server does.
// - 401 calls `onUnauthorized` so the session can end (expired/invalid token).
//   V1 tokens are minted by the IdP (or scripts/dev_token.py in dev); there is
//   no refresh endpoint, so an expired token sends the user back to /login.

import type {
  CaseListView,
  CaseStatus,
  CaseView,
  DraftView,
  FactView,
  Me,
  Problem,
  ReasonCodeOption,
  ReportView,
  ReviewCommand,
  ReviewResult,
  SubmitResponse,
  TimelineEvent,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(problem: Problem) {
    super(problem.reason ?? problem.title);
    this.name = "ApiError";
    this.status = problem.status;
    this.problem = problem;
  }

  get reason(): string {
    return this.problem.reason ?? this.problem.title;
  }
}

// Server-rendered overlay layers (GET /v1/cases/{id}/overlay?layer=…); the UI
// only stacks and fades them — no mask geometry in the browser.
export type OverlayLayer = "composite" | "image" | "disc" | "cup";

export interface ApiClientOptions {
  getToken: () => string | null;
  onUnauthorized?: (problem: Problem) => void;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export interface ApiClient {
  me(token?: string): Promise<Me>;
  listCases(statuses?: readonly CaseStatus[], limit?: number): Promise<CaseListView>;
  getCase(caseId: string): Promise<CaseView>;
  getTimeline(caseId: string): Promise<TimelineEvent[]>;
  getFacts(caseId: string): Promise<FactView[]>;
  getDraft(caseId: string): Promise<DraftView>;
  getOverlay(caseId: string, layer?: OverlayLayer): Promise<Blob>;
  reasonCodes(): Promise<ReasonCodeOption[]>;
  review(caseId: string, cmd: ReviewCommand): Promise<ReviewResult>;
  getReport(caseId: string): Promise<ReportView>;
  submitCase(form: FormData, idempotencyKey?: string): Promise<SubmitResponse>;
}

const PROBLEM_PREFIX = "urn:consilium:problem:";

async function toProblem(res: Response): Promise<Problem> {
  const type = res.headers.get("content-type") ?? "";
  if (type.includes("json")) {
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "status" in body && "type" in body) {
        return body as Problem;
      }
    } catch {
      // fall through to a synthetic problem
    }
  }
  return { type: `${PROBLEM_PREFIX}http`, title: res.statusText || "HTTP error", status: res.status };
}

export function createApiClient(opts: ApiClientOptions): ApiClient {
  const base = opts.baseUrl ?? "";
  const doFetch = opts.fetchImpl ?? ((input, init) => fetch(input, init));

  async function send(path: string, init: RequestInit = {}, token?: string): Promise<Response> {
    const headers = new Headers(init.headers);
    const bearer = token ?? opts.getToken();
    if (bearer) headers.set("Authorization", `Bearer ${bearer}`);
    let res: Response;
    try {
      res = await doFetch(`${base}${path}`, { ...init, headers });
    } catch {
      throw new ApiError({
        type: `${PROBLEM_PREFIX}network`,
        title: "Network error",
        status: 0,
        reason: "NETWORK_ERROR",
      });
    }
    if (!res.ok) {
      const problem = await toProblem(res);
      if (res.status === 401 && token === undefined) opts.onUnauthorized?.(problem);
      throw new ApiError(problem);
    }
    return res;
  }

  async function json<T>(path: string, init?: RequestInit, token?: string): Promise<T> {
    const res = await send(path, init, token);
    return (await res.json()) as T;
  }

  const enc = encodeURIComponent;

  return {
    me: (token) => json<Me>("/v1/me", undefined, token),
    listCases: (statuses = [], limit = 50) => {
      const q = new URLSearchParams();
      for (const s of statuses) q.append("status", s);
      q.set("limit", String(limit));
      return json<CaseListView>(`/v1/cases?${q.toString()}`);
    },
    getCase: (id) => json<CaseView>(`/v1/cases/${enc(id)}`),
    getTimeline: (id) => json<TimelineEvent[]>(`/v1/cases/${enc(id)}/timeline`),
    getFacts: (id) => json<FactView[]>(`/v1/cases/${enc(id)}/facts`),
    getDraft: (id) => json<DraftView>(`/v1/cases/${enc(id)}/draft`),
    getOverlay: async (id, layer = "composite") => {
      const q = layer === "composite" ? "" : `?layer=${layer}`;
      return (await send(`/v1/cases/${enc(id)}/overlay${q}`)).blob();
    },
    reasonCodes: () => json<ReasonCodeOption[]>("/v1/meta/reason-codes"),
    review: (id, cmd) =>
      json<ReviewResult>(`/v1/cases/${enc(id)}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cmd),
      }),
    getReport: (id) => json<ReportView>(`/v1/cases/${enc(id)}/report`),
    submitCase: async (form, idempotencyKey) => {
      const headers: Record<string, string> = {};
      if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
      // 202 Accepted: the pipeline runs asynchronously; poll status_url.
      const res = await send("/v1/cases", { method: "POST", body: form, headers });
      return (await res.json()) as SubmitResponse;
    },
  };
}
