import type { Report } from "../api/types";

export function downloadReport(report: Report): void {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `neural-forge-${report.exam_id}-${report.report_id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
