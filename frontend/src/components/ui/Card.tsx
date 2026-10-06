import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}

export function Card({ title, action, children, className = "", ...rest }: CardProps) {
  return (
    <section
      className={`rounded-2xl border border-ink-100 bg-white p-5 shadow-soft dark:border-ink-800 dark:bg-ink-900 print:break-inside-avoid print:shadow-none ${className}`}
      {...rest}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          {title && <h2 className="text-base font-semibold text-ink-900 dark:text-ink-50">{title}</h2>}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}
