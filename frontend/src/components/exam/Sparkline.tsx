import type { LivePoint } from "../../state/examStore";

const COLORS = ["#5eead4", "#fcd34d"];

export function Sparkline({ points, width = 140, height = 36 }: { points: LivePoint[]; width?: number; height?: number }) {
  if (points.length < 2) return null;
  const series = points[0].values.length;
  const all = points.flatMap((item) => item.values);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const start = points[0].t;
  const duration = points[points.length - 1].t - start || 1;
  const lines = Array.from({ length: series }, (_, index) =>
    points
      .map((item) => {
        const x = ((item.t - start) / duration) * (width - 4) + 2;
        const y = height - 2 - ((item.values[index] - min) / span) * (height - 4);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" "),
  );
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" className="rounded-lg bg-black/40">
      {lines.map((line, index) => (
        <polyline key={index} points={line} fill="none" stroke={COLORS[index % COLORS.length]} strokeWidth="1.6" strokeLinejoin="round" />
      ))}
    </svg>
  );
}
