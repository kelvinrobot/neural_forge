import type { FindingStatus, OverallStatus, ReferralUrgency } from "../api/types";

export interface Tone {
  label: string;
  banner: string;
  badge: string;
  bar: string;
}

export const overallStatusTone: Record<OverallStatus, Tone> = {
  urgent_attention: {
    label: "Urgent attention",
    banner: "bg-red-600 text-white",
    badge: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200",
    bar: "bg-red-600",
  },
  findings_detected: {
    label: "Findings detected",
    banner: "bg-orange-500 text-white",
    badge: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-200",
    bar: "bg-orange-500",
  },
  borderline_findings: {
    label: "Borderline findings",
    banner: "bg-amber-400 text-amber-950",
    badge: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200",
    bar: "bg-amber-400",
  },
  no_significant_findings: {
    label: "No significant findings",
    banner: "bg-emerald-600 text-white",
    badge: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200",
    bar: "bg-emerald-600",
  },
  inconclusive: {
    label: "Inconclusive",
    banner: "bg-ink-500 text-white",
    badge: "bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200",
    bar: "bg-ink-500",
  },
};

export const findingStatusTone: Record<FindingStatus, Tone> = {
  abnormal: overallStatusTone.findings_detected,
  borderline: overallStatusTone.borderline_findings,
  normal: overallStatusTone.no_significant_findings,
  inconclusive: overallStatusTone.inconclusive,
};

export const findingStatusLabel: Record<FindingStatus, string> = {
  abnormal: "Abnormal",
  borderline: "Borderline",
  normal: "Normal",
  inconclusive: "Inconclusive",
};

export const findingStatusOrder: FindingStatus[] = ["abnormal", "borderline", "normal", "inconclusive"];

export const urgencyTone: Record<ReferralUrgency, { label: string; badge: string }> = {
  none: { label: "No referral needed", badge: "bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200" },
  routine: { label: "Routine referral", badge: "bg-brand-100 text-brand-800 dark:bg-brand-950 dark:text-brand-200" },
  soon: { label: "Refer soon", badge: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200" },
  urgent: { label: "Urgent referral", badge: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200" },
};

export function likelihoodTone(likelihood: string): string {
  const value = likelihood.toLowerCase();
  if (value.includes("high") || (value.includes("likely") && !value.includes("unlikely"))) {
    return "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-200";
  }
  if (value.includes("moderate") || value.includes("possible") || value.includes("medium")) {
    return "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200";
  }
  return "bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200";
}
