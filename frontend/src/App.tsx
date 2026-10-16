import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { ApiError } from "./api/client";
import { RequireAuth } from "./auth/RequireAuth";
import { SessionProvider } from "./auth/session";
import { Layout } from "./components/Layout";
import { CaseDetailPage } from "./pages/CaseDetail/CaseDetailPage";
import { CaseListPage } from "./pages/CaseList/CaseListPage";
import { LoginPage } from "./pages/Login/LoginPage";
import { ReportPage } from "./pages/Report/ReportPage";
import { ReviewScreen } from "./pages/ReviewScreen/ReviewScreen";

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // 4xx are answers (403/404/409), not transient failures.
        retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
        refetchOnWindowFocus: false,
      },
    },
  });
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/cases"
          element={
            <RequireAuth>
              <CaseListPage />
            </RequireAuth>
          }
        />
        <Route
          path="/cases/:id"
          element={
            <RequireAuth>
              <CaseDetailPage />
            </RequireAuth>
          }
        />
        <Route
          path="/cases/:id/review"
          element={
            <RequireAuth>
              <ReviewScreen />
            </RequireAuth>
          }
        />
        <Route
          path="/cases/:id/report"
          element={
            <RequireAuth>
              <ReportPage />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/cases" replace />} />
      </Route>
    </Routes>
  );
}

export function App({ fetchImpl }: { fetchImpl?: typeof fetch }) {
  const [queryClient] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider {...(fetchImpl ? { fetchImpl } : {})}>
        <AppRoutes />
      </SessionProvider>
    </QueryClientProvider>
  );
}
