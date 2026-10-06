const ONE_DECIMAL_UNITS = new Set(["mm", "cm", "deg", "degree", "degrees", "°", "º"]);
const TWO_DECIMAL_UNITS = new Set(["hz", "s", "sec", "second", "seconds"]);
const INTEGER_UNITS = new Set(["%", "count", "counts", "steps", "taps", "reps", "repetitions", "blinks", "frames", "bpm", "blinks/min", "steps/min"]);

function normaliseUnit(unit: string | null | undefined): string {
  return (unit ?? "").trim().toLowerCase();
}

export function decimalsForUnit(unit: string | null | undefined): number | null {
  const normalised = normaliseUnit(unit);
  if (ONE_DECIMAL_UNITS.has(normalised)) return 1;
  if (TWO_DECIMAL_UNITS.has(normalised)) return 2;
  if (INTEGER_UNITS.has(normalised)) return 0;
  return null;
}

export function formatNumber(value: number, unit?: string | null): string {
  if (!Number.isFinite(value)) return "–";
  const decimals = decimalsForUnit(unit);
  if (decimals !== null) return value.toFixed(decimals);
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 100 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
}

export function displayUnit(unit: string | null | undefined): string {
  const normalised = normaliseUnit(unit);
  if (normalised === "deg" || normalised === "degree" || normalised === "degrees") return "°";
  if (normalised === "count" || normalised === "counts") return "";
  return unit ?? "";
}

export function formatValue(value: number | string | null | undefined, unit?: string | null): string {
  if (value === null || value === undefined || value === "") return "–";
  const text = typeof value === "number" ? formatNumber(value, unit) : value;
  const suffix = displayUnit(unit);
  if (!suffix) return text;
  return suffix === "°" || suffix === "%" ? `${text}${suffix}` : `${text} ${suffix}`;
}

export function formatPercent(fraction: number): string {
  const percent = fraction <= 1 ? fraction * 100 : fraction;
  return `${Math.round(percent)}%`;
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function formatSeconds(seconds: number): string {
  const safe = Math.max(0, Math.ceil(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return minutes > 0 ? `${minutes}:${String(rest).padStart(2, "0")}` : `${rest}s`;
}

export function titleCase(text: string): string {
  return text.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function evidenceSource(image: string): string {
  return image.startsWith("data:") ? image : `data:image/jpeg;base64,${image}`;
}
