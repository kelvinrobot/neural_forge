import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      colors: {
        brand: {
          50: "#ecfdfb",
          100: "#cff8f3",
          200: "#a0efe7",
          300: "#67dfd6",
          400: "#2fc4be",
          500: "#14a8a4",
          600: "#0d8786",
          700: "#0f6b6c",
          800: "#115557",
          900: "#123f43",
          950: "#04272a",
        },
        ink: {
          50: "#f6f8fa",
          100: "#eceff3",
          200: "#d8dee6",
          300: "#b5c0cc",
          400: "#8a99aa",
          500: "#66778a",
          600: "#4d5d70",
          700: "#3c4a5a",
          800: "#28323e",
          900: "#18202a",
          950: "#0e141b",
        },
      },
      boxShadow: {
        soft: "0 1px 2px rgba(16,24,40,0.04), 0 6px 20px rgba(16,24,40,0.06)",
      },
    },
  },
  plugins: [],
} satisfies Config;
