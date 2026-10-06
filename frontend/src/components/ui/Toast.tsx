import { AnimatePresence, motion } from "framer-motion";
import { create } from "zustand";
import { Icon } from "./Icon";

type ToastTone = "info" | "error" | "success";

interface ToastItem {
  id: number;
  tone: ToastTone;
  message: string;
}

interface ToastState {
  toasts: ToastItem[];
  push: (message: string, tone?: ToastTone) => void;
  dismiss: (id: number) => void;
}

let nextId = 1;

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  push: (message, tone = "info") => {
    const id = nextId++;
    set({ toasts: [...get().toasts.slice(-3), { id, tone, message }] });
    window.setTimeout(() => get().dismiss(id), 6000);
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((toast) => toast.id !== id) }),
}));

export function toast(message: string, tone: ToastTone = "info"): void {
  useToastStore.getState().push(message, tone);
}

const toneClasses: Record<ToastTone, string> = {
  info: "bg-ink-900 text-white dark:bg-ink-100 dark:text-ink-900",
  error: "bg-red-600 text-white",
  success: "bg-emerald-600 text-white",
};

export function Toaster() {
  const toasts = useToastStore((state) => state.toasts);
  const dismiss = useToastStore((state) => state.dismiss);
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex flex-col items-center gap-2 px-4 print:hidden" aria-live="assertive">
      <AnimatePresence>
        {toasts.map((item) => (
          <motion.div
            key={item.id}
            layout
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            className={`pointer-events-auto flex max-w-md items-center gap-3 rounded-xl px-4 py-3 text-sm shadow-lg ${toneClasses[item.tone]}`}
            role="alert"
          >
            <Icon name={item.tone === "error" ? "alert" : "check"} />
            <span className="flex-1">{item.message}</span>
            <button type="button" aria-label="Dismiss" onClick={() => dismiss(item.id)} className="rounded p-1 hover:bg-white/15">
              <Icon name="x" size={16} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
