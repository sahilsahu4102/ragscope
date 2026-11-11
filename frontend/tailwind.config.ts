import type { Config } from "tailwindcss";

// RAGScope "Experiment Notebook": navy cloth rail, bone paper, blue-black ink, one margin red.
// Components are styled in app/globals.css; Tailwind is used for layout utilities.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#f4f1e8",
        sheet: "#fbf9f3",
        sunk: "#ece8db",
        ink: "#14182b",
        "ink-2": "#3a425a",
        "ink-3": "#566075",
        rule: "#d4cebc",
        cloth: "#1f2a44",
        red: "#a8302a",
        blue: "#2b5fb0",
      },
      fontFamily: {
        sans: ["var(--font-ui)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
