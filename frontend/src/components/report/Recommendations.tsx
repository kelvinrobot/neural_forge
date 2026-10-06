import { Card } from "../ui/Card";
import { Icon } from "../ui/Icon";

export function Recommendations({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <Card title="Recommendations">
      <ol className="space-y-2 text-sm text-ink-700 dark:text-ink-200">
        {items.map((item, index) => (
          <li key={`${index}-${item}`} className="flex gap-2.5">
            <span className="mt-0.5 text-brand-600 dark:text-brand-300">
              <Icon name="chevronRight" size={16} />
            </span>
            {item}
          </li>
        ))}
      </ol>
    </Card>
  );
}
