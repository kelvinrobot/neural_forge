import { useSettingsStore } from "../../state/settingsStore";
import { Icon } from "../ui/Icon";

export function ThemeToggle() {
  const theme = useSettingsStore((state) => state.theme);
  const toggleTheme = useSettingsStore((state) => state.toggleTheme);
  const dark = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="rounded-xl p-2.5 text-ink-600 hover:bg-ink-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-500 dark:text-ink-300 dark:hover:bg-ink-800"
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
    >
      <Icon name={dark ? "sun" : "moon"} />
    </button>
  );
}
