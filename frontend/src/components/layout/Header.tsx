import { Link } from "react-router-dom";
import { ThemeToggle } from "./ThemeToggle";

export function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="16" className="fill-brand-600" />
      <circle cx="32" cy="32" r="14" fill="none" stroke="#ffffff" strokeWidth="4" />
      <circle cx="32" cy="32" r="5" fill="#ffffff" />
      <path d="M12 32h6M46 32h6M32 12v6M32 46v6" stroke="#a0efe7" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-ink-100 bg-white/85 backdrop-blur dark:border-ink-800 dark:bg-ink-950/85 print:hidden">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-3 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500">
          <BrandMark />
          <span className="leading-tight">
            <span className="block text-base font-bold text-ink-900 dark:text-ink-50">Neural Forge</span>
            <span className="block text-xs text-ink-500 dark:text-ink-400">AI Visual Diagnostics</span>
          </span>
        </Link>
        <div className="flex items-center gap-2">
          <Link
            to="/doctors/join"
            className="rounded-lg px-2 py-1.5 text-sm font-medium text-ink-700 hover:bg-ink-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500 dark:text-ink-200 dark:hover:bg-ink-800"
          >
            Join as a doctor
          </Link>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
