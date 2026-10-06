import type { Measurement } from "../../api/types";
import { formatValue, titleCase } from "../../utils/format";

export function MeasurementTable({ measurements }: { measurements: Measurement[] }) {
  if (measurements.length === 0) return null;
  const showSide = measurements.some((measurement) => measurement.side);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-ink-100 text-xs uppercase tracking-wide text-ink-500 dark:border-ink-800 dark:text-ink-400">
            <th scope="col" className="py-2 pr-3 font-semibold">Measurement</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Value</th>
            <th scope="col" className="py-2 pr-3 font-semibold">Reference</th>
            {showSide && <th scope="col" className="py-2 font-semibold">Side</th>}
          </tr>
        </thead>
        <tbody>
          {measurements.map((measurement, index) => (
            <tr key={`${measurement.name}-${measurement.side ?? ""}-${index}`} className="border-b border-ink-50 last:border-0 dark:border-ink-800">
              <td className="py-2 pr-3 text-ink-800 dark:text-ink-100">{measurement.name}</td>
              <td className="py-2 pr-3 font-semibold tabular-nums text-ink-900 dark:text-ink-50">{formatValue(measurement.value, measurement.unit)}</td>
              <td className="py-2 pr-3 text-ink-500 dark:text-ink-400">{measurement.reference ?? "–"}</td>
              {showSide && <td className="py-2 text-ink-600 dark:text-ink-300">{measurement.side ? titleCase(measurement.side) : "–"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
