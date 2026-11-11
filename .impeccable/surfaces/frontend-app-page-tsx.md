---
version: 1
slug: "frontend-app-page-tsx"
primary_target: "frontend/app/page.tsx"
related_targets: ["frontend/app/layout.tsx","frontend/components/Sidebar.tsx","frontend/app/documents/page.tsx","frontend/app/playground/page.tsx","frontend/app/inspector/page.tsx","frontend/app/traces/page.tsx","frontend/app/experiments/page.tsx","frontend/app/analytics/page.tsx"]
---

# RAGScope console: overview, documents, playground, inspector, traces, experiments, analytics

Mode: Operate. Visitor: a recruiter or engineer who sees the console as screenshots and a demo video now (and a live demo after it is deployed) and decides in seconds whether the evaluation work is real; behind them, the person running the stack. Task: see the stack's state and the latest activity, upload documents and watch ingestion, ask grounded questions with citations, inspect retrieval stage by stage, read a trace's span timeline, run and read A/B experiments with significance, and read latency, cost and cache analytics. Constraints: every figure and caption is read from the API (no hard-coded stats, versions, phases, sample queries or benchmarks); a starting, unreachable or empty backend is a stated state; state and span kind are never carried by colour alone; existing behaviour (upload, delete, streaming query, citations, feedback, filters, trace waterfall, experiment creation) is unchanged.

## Direction contract

THESIS: The console is a laboratory notebook, not a dark card dashboard. Every trace, experiment and document is an entry stamped with its real date and id, and every result is a figure whose caption is written from the data (n, the comparison, the interval). It refuses the navy-and-teal card grid, the gradient hero, the icon-on-card features and the stale version and phase labels.

OWN-WORLD: Navy cloth rail (#1F2A44) with a cream label plate; bone paper (#F4F1E8); blue-black ink (#14182B); pencil grey for secondary text; one margin red (#A8302A) reserved for a failure, a regression or a refusal; chart grounds in a pale graph blue, only inside charts. One grotesk for the interface and headings, a mono for ids, timestamps and figures. Square or barely rounded corners, hairline rules, no shadow, blur or gradient. State and span kind are a drawn mark plus a word: a solid, hollow, dashed or struck square for state; a distinct shape per span kind.

STORY: A viewer opens the Overview and sees the stack's real state, what has been ingested, and the latest traces and experiments as dated, identified entries. They open an experiment and read a figure with a caption they could cite: the two configurations on the same axes, n, the metric, the interval and the significance. They trust it because nothing on screen is typed in by hand and every entry carries the id that proves where it came from.

FIRST VIEWPORT: A navy cloth rail on the left (cream label plate with the wordmark, the seven tools as tabs, a small stack-state line), collapsing to a top bar on a phone. Beside it a bone page headed Overview: a Stack entry (API, database, cache, each with a drawn state mark and word, and a retry when something is down), an Ingested entry (documents, chunks, by status), then Latest traces and Latest experiments as dated, identified entries, newest first, each linking to its page. Empty and old data is stated plainly, with the age of the newest record.

FORM: Experiment Notebook (assigned, index 7 of the grounded list), seed key 301fa45f.

SIGNATURE: The entry: a header strip with the record kind, its date and short id, and a witness footer naming the full id and where it came from. Figures carry a caption written from the record's own fields. Nothing animates on entry; reduced motion is respected everywhere.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
