import { MotionConfig } from "framer-motion";
import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Footer } from "./components/layout/Footer";
import { Header } from "./components/layout/Header";
import { ErrorBoundary } from "./components/ui/ErrorBoundary";
import { Toaster } from "./components/ui/Toast";
import { ExamPage } from "./pages/ExamPage";
import { HomePage } from "./pages/HomePage";
import { PatientSetupPage } from "./pages/PatientSetupPage";
import { ReportPage } from "./pages/ReportPage";
import { useSettingsStore } from "./state/settingsStore";

export function App() {
  const theme = useSettingsStore((state) => state.theme);
  const location = useLocation();
  const onExam = location.pathname.endsWith("/live");

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    // Reports always print on a light page.
    const beforePrint = () => root.classList.remove("dark");
    const afterPrint = () => root.classList.toggle("dark", theme === "dark");
    window.addEventListener("beforeprint", beforePrint);
    window.addEventListener("afterprint", afterPrint);
    return () => {
      window.removeEventListener("beforeprint", beforePrint);
      window.removeEventListener("afterprint", afterPrint);
    };
  }, [theme]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <MotionConfig reducedMotion="user">
      <div className="flex min-h-screen flex-col">
        <Header />
        <main className={`mx-auto w-full flex-1 px-4 sm:px-6 ${onExam ? "max-w-[1600px] py-4" : "max-w-7xl py-8"}`}>
          <ErrorBoundary key={location.pathname}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/exam/:examId" element={<PatientSetupPage />} />
              <Route path="/exam/:examId/live" element={<ExamPage />} />
              <Route path="/report/:sessionId" element={<ReportPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
        {!onExam && <Footer />}
      </div>
      <Toaster />
    </MotionConfig>
  );
}
