import { Link } from "react-router-dom";

export function Footer() {
  return (
    <footer className="border-t border-ink-100 py-6 text-center text-xs text-ink-500 dark:border-ink-800 dark:text-ink-400 print:hidden">
      <p>Neural Forge · AI Visual Diagnostics · Video is processed in real time and is not stored.</p>
      <p className="mt-1 text-[11px] text-ink-400 dark:text-ink-500">
        Built by Kelvin ·{" "}
        <Link to="/portal" className="hover:underline">
          Doctor portal
        </Link>
      </p>
    </footer>
  );
}
