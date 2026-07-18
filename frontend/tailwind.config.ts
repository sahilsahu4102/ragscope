import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // RAGScope "Lumina Nexus" design system (teal / cyan on deep navy).
        background: "#0a0f1c",
        surface: "#0d1512",
        "surface-container": "#141b26",
        "surface-container-high": "#1b2431",
        border: "#26313f",
        primary: "#00d4aa",
        "primary-dim": "#00a888",
        secondary: "#00b4d8",
        accent: "#4cd6fb",
        "on-surface": "#dce4df",
        "on-surface-muted": "#8fa0b0",
        warning: "#ffb77a",
        danger: "#ffb4ab",
        // span-kind colors
        "kind-chain": "#8fa0b0",
        "kind-retriever": "#00b4d8",
        "kind-reranker": "#a78bfa",
        "kind-llm": "#00d4aa",
        "kind-embedding": "#4cd6fb",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
