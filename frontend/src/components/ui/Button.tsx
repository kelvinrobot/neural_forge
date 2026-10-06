import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: IconName;
  children: ReactNode;
}

const variantClasses: Record<Variant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-700 shadow-soft disabled:bg-brand-600/50",
  secondary:
    "bg-white text-ink-800 border border-ink-200 hover:bg-ink-50 dark:bg-ink-800 dark:text-ink-100 dark:border-ink-700 dark:hover:bg-ink-700",
  ghost: "text-ink-700 hover:bg-ink-100 dark:text-ink-200 dark:hover:bg-ink-800",
  danger: "bg-red-600 text-white hover:bg-red-700 shadow-soft disabled:bg-red-600/50",
};

const sizeClasses: Record<Size, string> = {
  sm: "h-9 px-3 text-sm gap-1.5",
  md: "h-11 px-4 text-sm gap-2",
  lg: "h-14 px-6 text-base gap-2.5",
};

export function Button({ variant = "primary", size = "md", icon, children, className = "", type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={`inline-flex items-center justify-center rounded-xl font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500 disabled:cursor-not-allowed disabled:opacity-60 ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...rest}
    >
      {icon && <Icon name={icon} size={size === "lg" ? 22 : 18} />}
      {children}
    </button>
  );
}
