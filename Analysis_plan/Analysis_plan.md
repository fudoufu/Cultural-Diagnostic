# Survey Analysis Instructions

## The Analysis Goal
Explore relationships between two sets of variables from survey data: **behaviours** and **outcomes**. The core question is which ways of working most strongly predict which outcomes (e.g. does collectivist behaviour correlate with people feeling valued, or planning to stay?).

---

## Analysis Visualisations

The main analysis visualisations will be:

- Descriptive analysis showing the average and spread of the rating of each ways of working (Individual & Place), and how this varies by organisational attributes and EDI attributes
- Descriptive analysis showing the average and spread of the rating of each sentiment outcome, and how this varies by organisational attributes and EDI attributes
- **[MOST IMPORTANT]** Correlational heat maps between ways of working (Place & Individual) against sentiment outcomes, filtered by organisational attributes and EDI attributes — shows which ways of working are most strongly predictive of better outcomes
- Correlational heat maps of ways of working against ways of working (for both People & Place) — to identify whether any behaviours are so highly correlated that they go hand in hand
- Correlational heat maps of outcomes against outcomes — to understand how the outcome variables relate to one another

---

## Visual Design Specification

- **Mode:** Light mode
- **Layout:** Streamlit wide mode (`st.set_page_config(layout="wide")`)
- **Font:** Inter (via Google Fonts CSS injection)
- **Colour palette:**
  - Primary: `#0F4C6B` (deep teal-navy)
  - Accent: `#00A8A8` (teal)
  - Background: `#F7F9FC`
  - Card/surface: `#FFFFFF`
  - Text: `#1A2B3C`
- **Heatmap colour scale:** Diverging — `#0F4C6B` (strong negative) → `#FFFFFF` (zero) → `#C0392B` (strong positive)
- **Descriptive tables:** Alternating row shading, bold row headers, clean borders — styled to match the CultureScope reference table aesthetic
- **RAG colours:**
  - Red: `#C0392B`
  - Amber: `#F39C12`
  - Green: `#27AE60`

### Summary Insight Cards
Each tab should display a row of summary KPI-style cards at the top containing:
- **n=** number of respondents included in the current filtered view
- **Highest and lowest scoring item** for the current tab (e.g. highest/lowest outcome, strongest correlation pair, biggest I–P gap)

---

## Technical Specification

- **Platform:** Streamlit (Python)
- **Correlation method:** Spearman rank correlation throughout — for all heatmaps (Ways of Working × Outcomes, Ways of Working × Ways of Working, Outcomes × Outcomes)
- **Missing data:** Pairwise exclusion — for any given correlation between two variables, exclude only respondents who are missing a value for one or both of that specific pair. Do not drop respondents wholesale from the dataset.
- **Expected sample size:** 2,000–4,000 rows (no validation warning required).

---

## Survey Design

**Step 1 — Column mapping:** Infer the question mapping from the raw data file by taking the first 77 columns in order and labelling them Q1–Q77. Ignore all columns after the 77th (i.e. from column BZ / 'Response ID' onwards). All subsequent references to question numbers refer to this mapped sequence.

**Step 2 — Directorate and organisational level values:** Infer directorate names from the distinct answer values in Q1. Infer Q9 organisational level categories from the distinct answer values in Q9. Do not hardcode these — read them dynamically from the data so the app remains robust to any label changes.

### Section Overview

| Section | Questions | Description |
|---|---|---|
| 1 | 1–10 | Organisational attributes |
| 2 | 11–32 | Ways of working — Place |
| 3 | 33–54 | Ways of working — Individual |
| 4 | 55–69 | Sentiment outcomes |
| 5 | 70–71 | Free text — **ignore; no analysis required** |
| 6 | 72–77 + Q8 | EDI attributes |

---

### Section 1 — Organisational Attributes (Qs 1–10)

- **Q1:** Directorate — used as the primary breakdown in Section B; filter all analyses by this
- **Q2–7:** Service area (each maps to one directorate from Q1; respondent jumps to the relevant sub-question) — filter all analyses by this
- **Q8:** Length of time at the council — **treated as an EDI attribute** (see Section 6)
- **Q9:** Part of the council that most shapes day-to-day work — used as the basis for Tab A6 (descriptive breakdown by organisational level); also available as a filter
- **Q10:** Free text follow-up for 'Other' answer in Q9 — **ignore; no analysis required**

---

### Section 2 & 3 — Ways of Working (Qs 11–54)

Sections 2 (Place) and 3 (Individual) ask about the same 11 pairs of ways of working, phrased differently. The 22 themes map onto questions in order:

| # | Theme |
|---|---|
| 1 | Collective Responsibility |
| 2 | Individual Accountability |
| 3 | Hierarchical Control |
| 4 | Empowered Decision-Making |
| 5 | Prioritise People |
| 6 | Protect Performance |
| 7 | Challenge Decisions |
| 8 | Preserve Cohesion |
| 9 | Follow Procedures |
| 10 | Adapt to Situation |
| 11 | Consolidate Existing Processes |
| 12 | Experiment & Innovate |
| 13 | Deliver Immediate Results |
| 14 | Invest in Long-Term Changes |
| 15 | Proactive Learning |
| 16 | Reactive Learning |
| 17 | Target-Driven Interactions |
| 18 | Relationship-Led Working |
| 19 | Plan-Based Working |
| 20 | Agile Working |
| 21 | Value Individual Performance |
| 22 | Value Individual Status |

---

### Section 4 — Sentiment Outcomes (Qs 55–69)

| Q | Label | Scoring |
|---|---|---|
| 55 | Intent to stay | Standard Likert 1–5 |
| 56 | Good place to work | Standard Likert 1–5 |
| 57 | Feeling valued | Standard Likert 1–5 |
| 58 | Pride | Standard Likert 1–5 |
| 59 | Sense of impact | Standard Likert 1–5 |
| 60 | Empowerment | Standard Likert 1–5 |
| 61 | Workload manageability | Standard Likert 1–5 |
| 62 | Role clarity | Standard Likert 1–5 |
| 63 | Voice heard | Standard Likert 1–5 |
| 64 | Psychological safety | Standard Likert 1–5 |
| 65 | Breaking silos | Standard Likert 1–5 |
| 66 | Opportunity for contribution | Standard Likert 1–5 |
| 67 | LM time | Standard Likert 1–5 |
| 68 | LM effectiveness | Standard Likert 1–5 |
| 69 | Employer rating | Rated 1–10; divide by 2 to standardise to 1–5 |

---

### Scoring

- **Sections 2–4, Qs 55–68:** Convert Likert responses to numeric:
  - Strongly disagree = 1
  - Disagree = 2
  - Neither agree nor disagree = 3
  - Agree = 4
  - Strongly agree = 5
- **Q69:** Rated out of 10 — divide by 2 to standardise to a 1–5 scale before including in any analysis

---

### Section 5 — Free Text (Qs 70–71)

**Ignore — no analysis required.**

---

### Section 6 — EDI Attributes (Qs 72–77 + Q8)

- **Q72–77:** EDI attributes (e.g. age, gender) — filter all analyses by these
- **Q8:** Length of time at the council — treated as an EDI attribute; filter all analyses by this

---

## Web App Layout

The app has two top-level sections:

- **Section A:** Whole council / cross-directorate view
- **Section B:** Broken down by directorate, then by service area within each directorate

---

## Section A — Tabs

### Tab A1 — Heatmap: Ways of Working × Outcomes

- Correlation heatmap of all 22 Ways of Working (rows) against all 15 Sentiment Outcomes (columns, Q55–69)
- Colour intensity encodes correlation strength
- Filterable by EDI attributes (Section 6)
- Shows whole-council data

---

### Tab A2 — Heatmap: Ways of Working × Ways of Working

- Correlation heatmap of all 22 Ways of Working against each other
- Identifies behaviours that co-occur so tightly they may be redundant to treat separately
- Filterable by EDI attributes
- Shows whole-council data

---

### Tab A3 — Heatmap: Outcomes × Outcomes

- Correlation heatmap of all 15 Sentiment Outcomes (Q55–69) against each other
- Shows how outcome variables cluster and relate
- Filterable by EDI attributes
- Shows whole-council data

---

### Tab A4 — Descriptive Table: Ways of Working (by Directorate)

Styled after the CultureScope reference table. Rows = 22 Ways of Working themes.

- One set of columns per directorate (7 total: 6 directorates + Overall), each with:
  - **I** = average Individual score (Q33–54)
  - **P** = average Place score (Q11–32)
  - **Δ** = difference (I minus P)

**RAG colour coding for I and P scores:**
- 🔴 Red: < 2.5
- 🟡 Amber: 2.5 – 3.5
- 🟢 Green: > 3.5

**Colour coding for Δ:**
- |Δ| > 1.0 → red
- |Δ| > 0.5 → amber
- Otherwise → uncoloured
- Both directions (I > P and P > I) are treated as equally noteworthy

**Filterable by EDI attributes (Section 6)**

---

### Tab A5 — Descriptive Table: Sentiment Outcomes (by Directorate)

Same structure as Tab A4, but rows = 15 Sentiment Outcome labels (Q55–69).

- One column per directorate (6 directorates + Overall)
- Each column shows average score only (single score per outcome — no I/P split)
- Same RAG colour coding: red < 2.5, amber 2.5–3.5, green > 3.5
- Filterable by EDI attributes

---

### Tab A6 — Descriptive Tables: Ways of Working & Outcomes (by Q9 Organisational Level)

Parallel to Tabs A4 and A5, but columns represent Q9 answer groups (organisational levels) rather than directorates.

- **Table 1:** Ways of Working — rows = 22 themes; columns = each Q9 answer group + Overall; shows I, P, and Δ per group
- **Table 2:** Sentiment Outcomes — rows = 15 outcomes; columns = each Q9 answer group + Overall; shows average score per group
- Same RAG colour coding and Δ flagging logic as Tabs A4/A5
- Filterable by EDI attributes

---

## Section B — Tabs

Same tab structure as Section A (B1–B5), but data is scoped to the selected directorate and broken down by service area rather than directorate. The directorate is selected via a row of clickable tabs/buttons at the top of Section B.

- **Tab B1:** Heatmap: Ways of Working × Outcomes (directorate-scoped)
- **Tab B2:** Heatmap: Ways of Working × Ways of Working (directorate-scoped)
- **Tab B3:** Heatmap: Outcomes × Outcomes (directorate-scoped)
- **Tab B4:** Descriptive Table: Ways of Working — columns = service areas within selected directorate + directorate total; shows I, P, and Δ per service area
- **Tab B5:** Descriptive Table: Sentiment Outcomes — columns = service areas within selected directorate + directorate total; shows average score per service area

All tabs filterable by EDI attributes. No Q9 breakdown tab in Section B.
