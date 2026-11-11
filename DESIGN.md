---
name: RAGScope
description: An evaluation and observability console for RAG, set like a lab notebook: navy cloth rail, bone paper, blue-black ink, and one entry per recorded fact with its real date and id.
colors:
  cloth: "#1f2a44"
  cloth-hover: "#2c3a5c"
  cloth-ink: "#dde2ee"
  cloth-rule: "#4b5a7c"
  paper: "#f4f1e8"
  sheet: "#fbf9f3"
  sunk: "#ece8db"
  ink: "#14182b"
  ink-2: "#3a425a"
  ink-3: "#566075"
  rule: "#d4cebc"
  hover-row: "#efebdd"
  red: "#a8302a"
  red-wash: "#f4e0dc"
  graph: "#cfd8ea"
  blue: "#2b5fb0"
  green: "#2f6b4f"
  plum: "#6b3f73"
  ochre: "#8a5a10"
typography:
  display:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "clamp(1.75rem, 3.4vw, 2.5rem)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.15
  title:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 700
    lineHeight: 1.15
  body:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 650
    lineHeight: 1.5
  caption:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
  figure:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  none: "0px"
spacing:
  rail: "248px"
  page-gutter: "40px"
  entry-gap: "40px"
components:
  button:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "40px"
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "40px"
  button-danger:
    backgroundColor: "transparent"
    textColor: "{colors.red}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "40px"
  field:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 10px"
    height: "40px"
  toggle:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 12px"
    height: "36px"
  toggle-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  nav-link:
    backgroundColor: "transparent"
    textColor: "{colors.cloth-ink}"
    rounded: "{rounded.none}"
    padding: "0 12px"
    height: "44px"
  nav-link-active:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  note:
    backgroundColor: "{colors.sunk}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "12px 14px"
---

# Design System: RAGScope

## Overview

**Creative North Star: "The Experiment Notebook"**

A good experiment notebook records each result as an entry: what kind of thing it is, the real date, an id you can look it up by, and a witness line at the foot saying who recorded it. RAGScope sets its whole console that way. Every trace, retrieval, experiment, ingest and answer is an **Entry**, and every chart is a **Figure** with a caption written from its own data. The world was chosen by the project owner from a rolled hand and replaces the earlier dark glass dashboard completely. It lends type, palette, density and one signature move; navigation, controls and page structure stay the standard ones anyone who has used an admin tool already knows.

The system is flat and legible. A navy cloth rail holds the plate, the seven pages and the live API state. Everything you read is blue-black ink on bone paper. Margin red appears only for failure and regression. Nothing glows, blurs, floats or casts a shadow; structure is a heavy rule above an entry, a hairline under its header, and a witness line at its foot. State is a drawn mark plus a word, so the interface survives greyscale.

Nothing on screen is typed in by hand. Every number, date, id and count comes from the API at load time, a missing value is written "n/a", and there is no invented numbering: an entry carries the id and date the backend recorded, never a counter the interface made up.

**Key Characteristics:**
- A navy cloth rail carries the paper-coloured plate, the nav, and the live API state; the page beside it is paper.
- The **Entry**: a heavy top rule, a header strip (kind, real date, short id or the source endpoint), the body, and a witness footer printing the full id.
- The **Figure**: a chart on sheet paper with a caption generated from the data (counts, window, how many differences are significant), never a hand-written claim.
- Hanken Grotesk for reading, JetBrains Mono for ids, durations, scores and endpoints; tabular numerals everywhere a number is read.
- One vocabulary of drawn state marks: solid square (ok, ready), ring (off), dashed ring (waiting, running), struck square (down, failed). Every mark is paired with a word.
- Span kinds are told apart by shape as well as colour: square chain, circle retriever, diamond reranker, triangle LLM, inverted triangle embedding.
- Square corners, no shadow, no gradient, no blur.
- Every wait is stated honestly: a sleeping host is announced as waking, a down API is named as unreachable, and nothing spins forever.

## Colors

Navy cloth, bone paper, blue-black ink, and one margin red. Blue, green, plum and ochre exist only to tell span kinds apart inside charts and marks.

### Primary
- **Cloth** (#1f2a44): the side rail and the hover fill of a pressed primary button. Text on it is Cloth Ink or Paper (12.6:1 for paper).

### Secondary
- **Margin Red** (#a8302a): failure, a down API, a destructive action, a regression. 6.0:1 on paper, 5.3:1 on its wash. Red text sits on paper, sheet or red-wash only.

### Neutral
- **Paper** (#f4f1e8): the page ground and the active nav tab.
- **Sheet** (#fbf9f3): typing fields and chart plates, a touch lighter than the paper.
- **Sunk** (#ece8db): notes, disabled controls, a selected row.
- **Ink** (#14182b): text, rules, the primary button, solid state marks (15.6:1 on paper).
- **Ink 2** (#3a425a): supporting text and labels (8.8:1).
- **Ink 3** (#566075): ids, hints, placeholders, axes (5.6:1 on paper, 5.2:1 on sunk).
- **Rule** (#d4cebc): hairlines under headers and between rows, never an outer edge and never the only carrier of meaning.
- **Hover Row** (#efebdd): a hovered table row.
- **Cloth Ink** (#dde2ee): nav text on the cloth (11.0:1). **Cloth Hover** (#2c3a5c) and **Cloth Rule** (#4b5a7c) are the rail's hover fill and divider.

### Chart colours
- **Graph Blue** (#cfd8ea): chart grid lines only.
- **Span kinds:** Ink 3 for chain, Blue (#2b5fb0) for retriever, Plum (#6b3f73) for reranker, Green (#2f6b4f) for LLM, Ochre (#8a5a10) for embedding. All are at least 5.2:1 on paper, and none is ever the only signal: each has its own shape.

### Named Rules
**The Margin-Red Rule.** Red marks failure and regression only. A destructive button is red; a neutral one never is, and "completed" is never green.

**The Colour-Is-Never-Alone Rule.** Every colour that means something also has a shape, a word or a position. Span kinds have shapes, states have marks plus words, and A against B is solid against hatched.

## Typography

**Reading Font:** Hanken Grotesk (400 to 800, loaded with `next/font/google`).
**Record Font:** JetBrains Mono (400 to 600), for everything that is looked up rather than read.

**Character:** a plain humanist grotesk for sentences and a monospace for the facts, so a reader can tell at a glance what was written and what was recorded.

### Hierarchy
- **Display** (700, `clamp(1.75rem, 3.4vw, 2.5rem)`, 1.15): page titles only.
- **Headline** (700, 1.25rem): reserved for the few second-level headings.
- **Title** (700, 1.0625rem): entry titles such as a question or an experiment name.
- **Body** (400, 0.9375rem, 1.5): sentences. Lede paragraphs are capped at 70ch and captions at 76ch.
- **Label** (650, 0.875rem): field labels, table heads, the entry kind.
- **Caption** (400, 0.8125rem): entry header text and figure captions.
- **Figure** (mono 400, 0.8125rem, tabular): ids, durations, scores, endpoints, counts in tables.

### Named Rules
**The Mono-Means-Recorded Rule.** Monospace is for values the system recorded: ids, durations, scores, endpoints. Never for decoration or headings.

**The Tabular Rule.** Every number a person compares sits in tabular numerals, and numeric columns are right-aligned.

## Layout

A 248px cloth rail on the left and a paper page with a 40px gutter. On a page, entries stack with 40px between them, or sit two to a row (`.cols`) at 1100px and wider. Below 1100px columns stack; below 860px the rail becomes a top bar with a scrolling nav; below 640px secondary table columns (marked `hide-sm`) are dropped and their meaning folded into the first cell (`only-sm`), so a table never needs sideways scrolling to be understood.

Rhythm comes from the entry: heavy rule, header, body, witness. Spacing follows a 4px base with a few deliberate 10px and 6px steps tuned to the 15px text.

## Elevation & Depth

There is none. No shadows, no blur, no gradients, no layered glass. Depth is expressed by three things only: the heavy 3px rule that opens an entry, the 1px hairlines inside it, and the sheet colour that lifts a typing field or chart plate a step off the paper.

### Named Rules
**The Flat Rule.** If something needs to stand out, change its rule weight, its ground or its mark. Never add a shadow.

## Shapes

Square corners everywhere (`border-radius: 0`). Borders are 1.5px ink on controls, 2px on the focus ring's offset, 3px ink for the focus ring and for an entry's top rule. A dashed border always means waiting or unavailable (dashed note, dashed drop zone, dashed disabled control).

## Components

- **Button.** Outline by default, ink-filled for the single primary action, red outline for destructive. Hover fills ink (or red); `:active` darkens and presses down 1px; disabled becomes a dashed sunk box.
- **Field.** Sheet ground, 1.5px ink border, label above, optional hint below. Disabled is dashed.
- **Toggle.** A pressed ink block with a drawn mark and a word; `aria-pressed` carries the state.
- **Entry / Figure.** The signature pair described above. The witness footer reads "Recorded by the RAGScope API. {kind} id {full id}".
- **Nav link.** Cloth Ink on the cloth; the current page is a paper block with ink text and a transparent-to-paper border.
- **Note.** A sunk box with a mark: dashed border for waiting, red border for an error with the recovery action inside it.
- **Tables.** Heavy rule under the head, hairlines between rows, numeric columns right-aligned in mono, ids as a second line in Ink 3.
- **Span waterfall.** Rows of kind mark, name and a bar on a shared 0 to total axis; selecting a span dims all the others and opens its attributes.
- **Source and citation.** A numbered square in the answer opens the matching source row; each source shows file, page, score and a clamped snippet.

## Do's and Don'ts

- **Do** show only values the API returned, and write a missing value as "n/a".
- **Do** pair every state colour with a drawn mark and a word.
- **Do** write each figure's caption from its own data: counts, window, how many differences are significant.
- **Do** announce a waiting or unreachable API in words, with a way to retry, instead of a spinner.
- **Do** keep entries to the same anatomy: heavy rule, header strip, body, witness footer.
- **Do** put A and B on identical axes and tell them apart by fill (solid and hatched), not hue alone.
- **Don't** invent numbering, sample data, trend arrows or stats for the sake of a fuller screen.
- **Don't** use shadows, blur, gradients, glow or rounded corners.
- **Don't** use red for anything except failure, a regression, or a destructive action.
- **Don't** use monospace for headings or decoration.
- **Don't** let a table need sideways scrolling to be understood on a phone; fold secondary columns into the first cell.
- **Don't** claim a benchmark gain the data does not support; the interface shows the paired-bootstrap verdict, including "not significant".
