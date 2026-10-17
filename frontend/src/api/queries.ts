// TanStack Query hooks. Polling is a fixed interval (ARCHITECTURE.md §5.1:
// 10 s while processing) applied unconditionally: the UI does not inspect a
// status to decide whether a case is "done" — it just re-reads.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { useApi } from "../auth/session";
import type { OverlayLayer } from "./client";
import type { CaseStatus, ReviewCommand } from "./types";

export const POLL_MS = 10_000;

export const keys = {
  cases: (statuses: readonly CaseStatus[]) => ["cases", [...statuses].sort()] as const,
  case: (id: string) => ["case", id] as const,
  timeline: (id: string) => ["case", id, "timeline"] as const,
  facts: (id: string) => ["case", id, "facts"] as const,
  draft: (id: string) => ["case", id, "draft"] as const,
  report: (id: string) => ["case", id, "report"] as const,
  overlay: (id: string, layer: OverlayLayer) => ["case", id, "overlay", layer] as const,
  reasonCodes: ["meta", "reason-codes"] as const,
};

export function useCaseList(statuses: readonly CaseStatus[]) {
  const api = useApi();
  return useQuery({
    queryKey: keys.cases(statuses),
    queryFn: () => api.listCases(statuses),
    refetchInterval: POLL_MS,
  });
}

/** Number of rows the server returns for a queue filter (capped at `limit`). */
export function useQueueCount(statuses: readonly CaseStatus[], limit: number) {
  const api = useApi();
  return useQuery({
    queryKey: ["cases", "count", [...statuses].sort(), limit] as const,
    queryFn: async () => (await api.listCases(statuses, limit)).items.length,
    refetchInterval: 30_000,
  });
}

export function useCase(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: keys.case(id),
    queryFn: () => api.getCase(id),
    refetchInterval: POLL_MS,
  });
}

export function useTimeline(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: keys.timeline(id),
    queryFn: () => api.getTimeline(id),
    refetchInterval: POLL_MS,
  });
}

export function useFacts(id: string, enabled: boolean) {
  const api = useApi();
  return useQuery({ queryKey: keys.facts(id), queryFn: () => api.getFacts(id), enabled });
}

// The draft is NOT polled: a clinician may be editing it (12.3) and a silent
// refresh would swap the base revision under them.
export function useDraft(id: string) {
  const api = useApi();
  return useQuery({ queryKey: keys.draft(id), queryFn: () => api.getDraft(id) });
}

export function useReport(id: string) {
  const api = useApi();
  return useQuery({ queryKey: keys.report(id), queryFn: () => api.getReport(id) });
}

export function useReasonCodes() {
  const api = useApi();
  return useQuery({ queryKey: keys.reasonCodes, queryFn: () => api.reasonCodes(), staleTime: Infinity });
}

/** One server-rendered overlay layer as an object URL (revoked on change/unmount). */
export function useOverlayLayer(id: string, layer: OverlayLayer) {
  const api = useApi();
  const q = useQuery({
    queryKey: keys.overlay(id, layer),
    queryFn: () => api.getOverlay(id, layer),
    staleTime: Infinity,
  });
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!q.data) return undefined;
    const objectUrl = URL.createObjectURL(q.data);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [q.data]);
  return { url, error: q.error };
}

/** POST /v1/cases/{id}/review. On success every view of the case is re-read
 *  from the server (the UI never patches a status locally). */
export function useReviewMutation(id: string) {
  const api = useApi();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cmd: ReviewCommand) => api.review(id, cmd),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ["case", id] });
      await qc.invalidateQueries({ queryKey: ["cases"] });
    },
  });
}

/** Polls faster while the server says the pipeline is still running. */
export function useCaseProgress(id: string) {
  const api = useApi();
  return useQuery({
    queryKey: keys.case(id),
    queryFn: () => api.getCase(id),
    refetchInterval: (q) => (q.state.data?.in_progress ? 2_500 : POLL_MS),
  });
}

export function useLiveTimeline(id: string, fast: boolean) {
  const api = useApi();
  return useQuery({
    queryKey: keys.timeline(id),
    queryFn: () => api.getTimeline(id),
    refetchInterval: fast ? 2_500 : false,
  });
}

export function useReportDocument(id: string, enabled: boolean) {
  const api = useApi();
  return useQuery({
    queryKey: ["case", id, "report-document"] as const,
    queryFn: () => api.getReportDocument(id),
    enabled,
  });
}

export function useSystemStatus() {
  const api = useApi();
  return useQuery({ queryKey: ["meta", "status"] as const, queryFn: () => api.systemStatus(), refetchInterval: 15_000 });
}

export function useRecentCases(limit = 6) {
  const api = useApi();
  return useQuery({
    queryKey: ["cases", "recent", limit] as const,
    queryFn: () => api.listCases([], limit),
    refetchInterval: POLL_MS,
  });
}
