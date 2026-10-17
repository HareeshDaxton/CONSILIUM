import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { ApiError } from "./api/client";
import { RequireAuth } from "./auth/RequireAuth";
import { SessionProvider } from "./auth/session";
import { AppShell } from "./components/Layout";
import { CaseListPage } from "./pages/CaseList/CaseListPage";
import { LandingPage } from "./pages/Landing/LandingPage";
import { LoginPage } from "./pages/Login/LoginPage";
import { NewScreeningPage } from "./pages/NewScreening/NewScreeningPage";
import { ReportPage } from "./pages/Report/ReportPage";
import { ResultPage } from "./pages/Result/ResultPage";
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
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route path="/app" element={<NewScreeningPage />} />
        <Route path="/cases" element={<CaseListPage />} />
        <Route path="/cases/:id" element={<ResultPage />} />
        <Route path="/cases/:id/review" element={<ReviewScreen />} />
        <Route path="/cases/:id/report" element={<ReportPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
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
