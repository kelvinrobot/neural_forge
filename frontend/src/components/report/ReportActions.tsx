import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import type { Report } from "../../api/types";
import { useExamStore } from "../../state/examStore";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { toast } from "../ui/Toast";

export function ReportActions({ report }: { report: Report }) {
  const navigate = useNavigate();
  const clearAll = useExamStore((state) => state.clearAll);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `neural-forge-${report.exam_id}-${report.report_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const deleteSession = async () => {
    setDeleting(true);
    try {
      await api.deleteSession(report.session_id);
      clearAll();
      toast("Session data deleted.", "success");
      navigate("/");
    } catch {
      toast("The session data could not be deleted. Please try again.", "error");
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div className="flex flex-wrap gap-2 print:hidden">
      <Button icon="print" onClick={() => window.print()}>
        Print or save as PDF
      </Button>
      <Button variant="secondary" icon="download" onClick={downloadJson}>
        Download JSON
      </Button>
      <Button
        variant="secondary"
        icon="plus"
        onClick={() => {
          clearAll();
          navigate("/");
        }}
      >
        Start another examination
      </Button>
      <Button variant="ghost" icon="trash" className="text-red-600 dark:text-red-400" onClick={() => setConfirmDelete(true)}>
        Delete session data
      </Button>
      <Modal
        open={confirmDelete}
        title="Delete session data?"
        onClose={() => setConfirmDelete(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={deleteSession} disabled={deleting}>
              Delete permanently
            </Button>
          </>
        }
      >
        The report, findings and any follow-up questions are removed from the server. Download or print the report first if you need a copy.
      </Modal>
    </div>
  );
}
