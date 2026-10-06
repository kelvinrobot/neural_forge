import { useState, type FormEvent } from "react";
import { api, ApiError } from "../../api/client";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { Spinner } from "../ui/Spinner";

interface ChatEntry {
  role: "question" | "answer" | "error";
  text: string;
}

interface QuestionChatProps {
  sessionId: string;
  onSpeak: (text: string) => void;
}

export function QuestionChat({ sessionId, onSpeak }: QuestionChatProps) {
  const [question, setQuestion] = useState("");
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [speakAnswers, setSpeakAnswers] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    setEntries((previous) => [...previous, { role: "question", text }]);
    setQuestion("");
    setBusy(true);
    try {
      const response = await api.question(sessionId, text);
      setEntries((previous) => [...previous, { role: "answer", text: response.answer }]);
      if (speakAnswers) onSpeak(response.speech_text || response.answer);
    } catch (error) {
      const message = error instanceof ApiError ? error.detail : "The question could not be answered.";
      setEntries((previous) => [...previous, { role: "error", text: message }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="Ask a follow-up question"
      className="print:hidden"
      action={
        <label className="flex items-center gap-2 text-xs text-ink-600 dark:text-ink-300">
          <input type="checkbox" checked={speakAnswers} onChange={(event) => setSpeakAnswers(event.target.checked)} className="h-4 w-4 accent-brand-600" />
          Speak answers
        </label>
      }
    >
      <div className="max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
        {entries.length === 0 && <p className="text-sm text-ink-500 dark:text-ink-400">Ask about any finding or recommendation in this report.</p>}
        {entries.map((entry, index) => (
          <div
            key={index}
            className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-sm ${
              entry.role === "question"
                ? "ml-auto bg-brand-600 text-white"
                : entry.role === "error"
                  ? "bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-200"
                  : "bg-ink-100 text-ink-800 dark:bg-ink-800 dark:text-ink-100"
            }`}
          >
            {entry.text}
          </div>
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-sm text-ink-500">
            <Spinner size={16} label="Waiting for answer" /> Thinking…
          </div>
        )}
      </div>
      <form onSubmit={submit} className="mt-3 flex gap-2">
        <label htmlFor="follow-up" className="sr-only">
          Your question
        </label>
        <input
          id="follow-up"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          maxLength={1000}
          placeholder="Type your question"
          className="h-11 flex-1 rounded-xl border border-ink-200 bg-white px-3 text-sm text-ink-900 outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:border-ink-700 dark:bg-ink-800 dark:text-ink-50"
        />
        <Button type="submit" icon="send" disabled={busy || !question.trim()}>
          Ask
        </Button>
      </form>
    </Card>
  );
}
