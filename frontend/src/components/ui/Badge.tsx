import type { ReactNode } from "react";

interface BadgeProps {
  className?: string;
  children: ReactNode;
}

export function Badge({ className = "bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200", children }: BadgeProps) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ${className}`}>{children}</span>
  );
}
