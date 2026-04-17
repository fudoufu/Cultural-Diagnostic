import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cultural Diagnostic", layout="wide", initial_sidebar_state="expanded")

# ── CSS / font injection ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Inter', sans-serif !important; }
.stApp, [data-testid="stAppViewContainer"], .main, .block-container { background-color: #F7F9FC; }
.block-container { padding-top: 4.25rem; padding-bottom: 2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF;
    border: 1px solid #D6E0EA;
    border-radius: 6px;
    padding: 6px 18px;
    color: #1A2B3C;
    font-weight: 500;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: #0F4C6B !important;
    color: #FFFFFF !important;
    border-color: #0F4C6B !important;
}
/* Level 2 tabs (Correlation / Descriptive) */
.stTabs .stTabs [aria-selected="true"] {
    background-color: #2E7096 !important;
    border-color: #2E7096 !important;
    color: #FFFFFF !important;
}
/* Level 3 tabs (A1/A2/A3 etc.) */
.stTabs .stTabs .stTabs [aria-selected="true"] {
    background-color: #5A9BB5 !important;
    border-color: #5A9BB5 !important;
    color: #FFFFFF !important;
}
/* Level 4 tabs (Place / Individual) */
.stTabs .stTabs .stTabs .stTabs [aria-selected="true"] {
    background-color: #8DC0D4 !important;
    border-color: #8DC0D4 !important;
    color: #1A2B3C !important;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D6E0EA;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.metric-card .card-label { color: #5A7080; font-size: 12px; font-weight: 500; margin: 0; }
.metric-card .card-value { color: #0F4C6B; font-size: 20px; font-weight: 700; margin: 4px 0 0; }
.metric-card .card-sub   { color: #1A2B3C; font-size: 13px; font-weight: 500; margin: 2px 0 0; }
.section-pill {
    display: inline-block;
    background: #0F4C6B;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
}
.section-pill-alt {
    display: inline-block;
    background: #00A8A8;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
    margin-top: 18px;
}
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p {
    color: #1A2B3C !important;
    font-weight: 500;
}
h4 {
    color: #0F4C6B !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    margin: 4px 0 12px 0 !important;
}
.landing-hero {
    min-height: 62vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #1A2B3C;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────────
WOW_THEMES = [
    "Collective Responsibility", "Individual Accountability",
    "Status Orientation", "Autonomy over Decisions",
    "Prioritise People's Well-Being", "Prioritise Results",
    "Challenge Decisions", "Preserve Cohesion",
    "Follow Procedures", "Adapt to Situation",
    "Stick to Current Ways", "Experiment & Innovate",
    "Prioritise Immediate Results", "Consider the Long Term",
    "Proactive Learning", "Reactive Learning",
    "Target-Driven Interactions", "Relationship-Led Working",
    "Plan-Based Working", "Agile Working",
    "Recognise Contributions", "Value Individual Status",
]

OUTCOME_LABELS = [
    "Intent to stay", "Good place to work", "Feeling valued", "Pride",
    "Sense of impact", "Empowerment", "Workload manageability", "Role clarity",
    "Voice heard", "Psychological safety", "Breaking silos",
    "Opportunity for contribution", "LM time", "LM effectiveness", "Employer rating",
]

LIKERT_MAP = {
    "Strongly agree": 5, "Agree": 4, "Neither agree nor disagree": 3,
    "Disagree": 2, "Strongly disagree": 1,
}

EDI_FILTERS = {
    "Q8":  "Length of service",
    "Q72": "Age",
    "Q73": "Gender",
    "Q74": "Ethnic group",
    "Q75": "Sexual orientation",
    "Q76": "Disabled",
    "Q77": "Carer",
}

# Column ranges
WOW_PLACE_COLS = [f"Q{i}" for i in range(11, 33)]   # Q11–Q32
WOW_IND_COLS   = [f"Q{i}" for i in range(33, 55)]   # Q33–Q54
WOW_ALL_COLS   = WOW_PLACE_COLS + WOW_IND_COLS
OUTCOME_COLS   = [f"Q{i}" for i in range(55, 70)]   # Q55–Q69

WOW_ALL_LABELS = [f"P · {t}" for t in WOW_THEMES] + [f"I · {t}" for t in WOW_THEMES]

Q9_ORDER = ["My directorate", "My service", "My immediate team", "Other"]

# Colours
PRIMARY  = "#0F4C6B"
RED      = "#C0392B"
AMBER    = "#F39C12"
GREEN    = "#27AE60"

HEATMAP_COLORSCALE = [[0.0, "#0F4C6B"], [0.5, "#FFFFFF"], [1.0, "#C0392B"]]

# ── Data loading ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading survey data…")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), header=0)
    df = df.iloc[:, :77].copy()
    df.columns = [f"Q{i+1}" for i in range(77)]

    # Coalesce Q2–Q7 into a single service_area column
    sa_cols = [f"Q{i}" for i in range(2, 8)]
    df["service_area"] = df[sa_cols].apply(
        lambda row: next(
            (str(v).strip() for v in row
             if pd.notna(v) and str(v).strip() not in ("", "Not Answered")),
            "Not Answered",
        ),
        axis=1,
    )

    # Convert Likert columns Q11–Q68 to numeric
    for col in [f"Q{i}" for i in range(11, 69)]:
        df[col] = df[col].map(LIKERT_MAP)

    # Convert Q69 (1–10) → 1–5
    def parse_q69(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).split()[0]) / 2
        except (ValueError, IndexError):
            return np.nan

    df["Q69"] = df["Q69"].apply(parse_q69)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, vals in selections.items():
        if vals:
            mask &= df[col].isin(vals)
    return df[mask]


def spearman_matrix(df: pd.DataFrame, x_cols: list, y_cols: list) -> pd.DataFrame:
    """Rectangular Spearman correlation matrix (pairwise complete obs)."""
    all_cols = list(dict.fromkeys(x_cols + y_cols))
    corr = df[all_cols].astype(float).corr(method="spearman")
    return corr.loc[x_cols, y_cols]


def make_writable_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(mat.to_numpy(copy=True), index=mat.index, columns=mat.columns)


def fill_diagonal_with_nan(mat: pd.DataFrame) -> pd.DataFrame:
    data = mat.to_numpy(copy=True)
    np.fill_diagonal(data, np.nan)
    mat.iloc[:, :] = data
    return mat


def get_filter_options(df: pd.DataFrame, col: str) -> list:
    return sorted(
        v for v in df[col].dropna().unique()
        if str(v).strip() not in ("Not Answered", "")
    )


# ── Visualisations ────────────────────────────────────────────────────────────────
def make_heatmap(matrix: pd.DataFrame, y_labels: list, x_labels: list) -> go.Figure:
    z = matrix.values.astype(float)
    text = np.where(np.isnan(z), "", np.round(z, 2).astype(str))
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=HEATMAP_COLORSCALE,
        zmid=0, zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1], len=0.7),
    ))
    fig.update_layout(
        font=dict(family="Inter", color="#1A2B3C"),
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#F7F9FC",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-40, tickfont=dict(size=11, color="#1A2B3C"), side="bottom"),
        yaxis=dict(tickfont=dict(size=11, color="#1A2B3C"), autorange="reversed"),
        height=max(420, len(y_labels) * 26 + 100),
    )
    return fig


def make_wow_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Place, Individual, Δ (P−I) per WoW theme."""
    p_vals = [df[col].mean() for col in WOW_PLACE_COLS]
    i_vals = [df[col].mean() for col in WOW_IND_COLS]
    delta_vals = [
        p - i if not (np.isnan(p) or np.isnan(i)) else np.nan
        for p, i in zip(p_vals, i_vals)
    ]
    pos_delta = [d if not np.isnan(d) and d >= 0 else np.nan for d in delta_vals]
    neg_delta = [d if not np.isnan(d) and d <  0 else np.nan for d in delta_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Place (P)", x=WOW_THEMES, y=p_vals,
        marker_color=PRIMARY,
        hovertemplate="%{x}<br>Place: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Individual (I)", x=WOW_THEMES, y=i_vals,
        marker_color="#5A9BB5",
        hovertemplate="%{x}<br>Individual: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Δ — Place higher", x=WOW_THEMES, y=pos_delta,
        marker_color=GREEN,
        hovertemplate="%{x}<br>Δ (P−I): %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Δ — Individual higher", x=WOW_THEMES, y=neg_delta,
        marker_color=RED,
        hovertemplate="%{x}<br>Δ (P−I): %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group",
        font=dict(family="Inter", color="#1A2B3C"),
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#F7F9FC",
        margin=dict(l=10, r=10, t=10, b=180),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#1A2B3C")),
        yaxis=dict(
            title=dict(text="Average Score", font=dict(color="#1A2B3C", size=12)),
            tickfont=dict(size=11, color="#1A2B3C"),
            zeroline=True, zerolinecolor="#9EB5C2", zerolinewidth=1.5,
            gridcolor="#E8EEF2",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#1A2B3C", size=12),
        ),
        height=560,
    )
    return fig


def make_outcome_bar_chart(df: pd.DataFrame, overall_df: pd.DataFrame = None) -> go.Figure:
    """Bar chart: average score for each employee experience outcome.
    If overall_df is provided, adds a lighter overall bar for comparison."""
    vals = [df[col].mean() for col in OUTCOME_COLS]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Selected group",
        x=OUTCOME_LABELS, y=vals,
        marker_color=PRIMARY,
        hovertemplate="%{x}<br>Score: %{y:.2f}<extra></extra>",
    ))
    if overall_df is not None:
        fig.add_trace(go.Bar(
            name="Overall",
            x=OUTCOME_LABELS,
            y=[overall_df[col].mean() for col in OUTCOME_COLS],
            marker_color="#6BA3BA",
            hovertemplate="%{x}<br>Overall: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        font=dict(family="Inter", color="#1A2B3C"),
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#F7F9FC",
        margin=dict(l=10, r=10, t=10, b=160),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#1A2B3C")),
        yaxis=dict(
            title=dict(text="Average Score", font=dict(color="#1A2B3C", size=12)),
            range=[0, 5.5],
            tickfont=dict(size=11, color="#1A2B3C"),
            gridcolor="#E8EEF2",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#1A2B3C", size=12),
        ),
        height=480,
    )
    return fig


def render_heatmap_cards(n: int, matrix: pd.DataFrame):
    flat = matrix.stack().dropna()
    # Deduplicate symmetric pairs: (A, B) and (B, A) are the same correlation
    seen: set = set()
    unique: dict = {}
    for idx, val in flat.items():
        key = frozenset(idx)
        if key not in seen:
            seen.add(key)
            unique[idx] = val
    flat = pd.Series(unique)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                    f'<p class="card-value">n = {n:,}</p></div>', unsafe_allow_html=True)
    if flat.empty:
        return
    top3_pos = flat.nlargest(3)
    top3_neg = flat.nsmallest(3)
    pos_html = "".join(
        f'<p class="card-sub">{idx[0]} × {idx[1]}</p><p class="card-value">r = {val:.3f}</p>'
        for idx, val in top3_pos.items()
    )
    neg_html = "".join(
        f'<p class="card-sub">{idx[0]} × {idx[1]}</p><p class="card-value">r = {val:.3f}</p>'
        for idx, val in top3_neg.items()
    )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Strongest positive correlations</p>'
            f'{pos_html}</div>',
            unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Strongest negative correlations</p>'
            f'{neg_html}</div>',
            unsafe_allow_html=True)


def render_summary_cards(n: int, top3_high: list, top3_low: list):
    """top3_high / top3_low are lists of (label, value_str) tuples."""
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                    f'<p class="card-value">n = {n:,}</p></div>', unsafe_allow_html=True)
    high_html = "".join(
        f'<p class="card-sub">{lbl}</p><p class="card-value">{val}</p>'
        for lbl, val in top3_high
    )
    low_html = "".join(
        f'<p class="card-sub">{lbl}</p><p class="card-value">{val}</p>'
        for lbl, val in top3_low
    )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Highest scoring</p>'
            f'{high_html}</div>',
            unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Lowest scoring</p>'
            f'{low_html}</div>',
            unsafe_allow_html=True)


# ── Table builders ────────────────────────────────────────────────────────────────
def _rag(val, is_delta=False):
    if pd.isna(val):
        return ""
    if is_delta:
        if abs(val) > 1.0:
            return f"background-color: {RED}; color: white"
        if abs(val) > 0.5:
            return f"background-color: {AMBER}; color: #1A2B3C"
        return ""
    if val < 2.5:
        return f"background-color: {RED}; color: white"
    if val <= 3.5:
        return f"background-color: {AMBER}; color: #1A2B3C"
    return f"background-color: {GREEN}; color: #1A2B3C"


def build_wow_table(df: pd.DataFrame, breakdown_col: str,
                    groups: list) -> tuple[pd.DataFrame, object]:
    """Returns (raw DataFrame, Styler) for the WoW I/P/Δ table."""
    rows = []
    for idx, theme in enumerate(WOW_THEMES):
        p_col = WOW_PLACE_COLS[idx]
        i_col = WOW_IND_COLS[idx]
        row = {"Ways of Working": theme}

        p_ov = df[p_col].mean()
        i_ov = df[i_col].mean()
        row["Overall — P"] = round(p_ov, 2) if not np.isnan(p_ov) else None
        row["Overall — I"] = round(i_ov, 2) if not np.isnan(i_ov) else None
        row["Overall — Δ"] = (round(i_ov - p_ov, 2)
                              if not (np.isnan(p_ov) or np.isnan(i_ov)) else None)

        for g in groups:
            sub = df[df[breakdown_col] == g]
            pm = sub[p_col].mean() if len(sub) else np.nan
            im = sub[i_col].mean() if len(sub) else np.nan
            row[f"{g} — P"] = round(pm, 2) if not np.isnan(pm) else None
            row[f"{g} — I"] = round(im, 2) if not np.isnan(im) else None
            row[f"{g} — Δ"] = (round(im - pm, 2)
                                if not (np.isnan(pm) or np.isnan(im)) else None)
        rows.append(row)

    tdf = pd.DataFrame(rows).set_index("Ways of Working")

    def style_col(col):
        is_delta = "— Δ" in col.name
        return [_rag(v, is_delta) for v in col]

    styler = (tdf.style
              .apply(style_col, axis=0)
              .format("{:.2f}", na_rep="—")
              .set_table_styles([
                  {"selector": "th", "props": [("font-weight", "600"),
                                               ("background-color", "#F0F4F8"),
                                               ("color", "#1A2B3C")]},
                  {"selector": "tr:nth-child(even) td",
                   "props": [("background-color", "#FAFBFC")]},
              ]))
    return tdf, styler


def build_outcome_table(df: pd.DataFrame, breakdown_col: str,
                        groups: list) -> tuple[pd.DataFrame, object]:
    rows = []
    for label, col in zip(OUTCOME_LABELS, OUTCOME_COLS):
        row = {"Outcome": label}
        ov = df[col].mean()
        row["Overall"] = round(ov, 2) if not np.isnan(ov) else None
        for g in groups:
            sub = df[df[breakdown_col] == g]
            m = sub[col].mean() if len(sub) else np.nan
            row[g] = round(m, 2) if not np.isnan(m) else None
        rows.append(row)

    tdf = pd.DataFrame(rows).set_index("Outcome")

    def style_col(col):
        return [_rag(v, is_delta=False) for v in col]

    styler = (tdf.style
              .apply(style_col, axis=0)
              .format("{:.2f}", na_rep="—")
              .set_table_styles([
                  {"selector": "th", "props": [("font-weight", "600"),
                                               ("background-color", "#F0F4F8"),
                                               ("color", "#1A2B3C")]},
                  {"selector": "tr:nth-child(even) td",
                   "props": [("background-color", "#FAFBFC")]},
              ]))
    return tdf, styler


def wow_table_cards(n, tdf):
    p = pd.to_numeric(tdf["Overall — P"], errors="coerce")
    i = pd.to_numeric(tdf["Overall — I"], errors="coerce")
    combined = pd.concat([
        p.rename(index=lambda x: f"P · {x}"),
        i.rename(index=lambda x: f"I · {x}"),
    ]).dropna()
    if combined.empty:
        render_summary_cards(n, [], [])
        return
    top3_high = [(lbl, f"{val:.2f}") for lbl, val in combined.nlargest(3).items()]
    top3_low  = [(lbl, f"{val:.2f}") for lbl, val in combined.nsmallest(3).items()]
    render_summary_cards(n, top3_high, top3_low)


def outcome_table_cards(n, tdf):
    ov = pd.to_numeric(tdf["Overall"], errors="coerce").dropna()
    if ov.empty:
        render_summary_cards(n, [], [])
        return
    top3_high = [(lbl, f"{val:.2f}") for lbl, val in ov.nlargest(3).items()]
    top3_low  = [(lbl, f"{val:.2f}") for lbl, val in ov.nsmallest(3).items()]
    render_summary_cards(n, top3_high, top3_low)


# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="color:{PRIMARY};font-size:19px;font-weight:700;'
        f'padding-bottom:10px">Cultural Diagnostic</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload survey export (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.markdown("**EDI Filters**")
    filter_selections: dict = {}

# ── Landing screen ────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown(
        f"""
        <div class="landing-hero" style="padding:24px 40px 12px;">
            <div style="font-size:52px;margin-bottom:16px">📊</div>
            <h2 style="color:{PRIMARY};margin-bottom:8px">Our Culture Discovery - survey analysis tool</h2>
            <p style="font-size:15px;color:#5A7080">
                Upload a survey export using the panel on the left to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load & process data ───────────────────────────────────────────────────────────
df = load_data(uploaded.read())

# Build EDI filter controls in sidebar
with st.sidebar:
    for q_col, label in EDI_FILTERS.items():
        opts = get_filter_options(df, q_col)
        if opts:
            sel = st.multiselect(label, opts, default=[], key=f"f_{q_col}")
            if sel:
                filter_selections[q_col] = sel

filtered = apply_filters(df, filter_selections)
n_total = len(filtered)

if n_total == 0:
    st.warning("No respondents match the current filters. Adjust the filters to see results.")
    st.stop()

# Dynamic lookup values (read from data, not hardcoded)
directorates = get_filter_options(df, "Q1")
q9_levels    = get_filter_options(df, "Q9")

# ── Top-level section tabs ────────────────────────────────────────────────────────
sec_a, sec_b = st.tabs([
    "Section A — Council-Wide",
    "Section B — Service Deep Dive",
])

# ═══════════════════════════════════════════════════════════════════════════════════
# SECTION A
# ═══════════════════════════════════════════════════════════════════════════════════
with sec_a:
    corr_group, desc_group = st.tabs(["Correlation Analysis", "Descriptive Analysis"])

    # ── Correlation Analysis group ────────────────────────
    with corr_group:
        a1, a2, a3 = st.tabs([
            "A1 · WoW × Outcomes",
            "A2 · WoW × WoW",
            "A3 · Outcomes × Outcomes",
        ])

        # ── A1: Ways of Working × Outcomes ───────────────────
        with a1:
            a1_place, a1_ind = st.tabs(["Place (P)", "Individual (I)"])
            with a1_place:
                st.markdown("#### Correlational Heatmap: Ways of Working (Place) × Employee Experience")
                mat = spearman_matrix(filtered, WOW_PLACE_COLS, OUTCOME_COLS)
                mat.index   = WOW_THEMES
                mat.columns = OUTCOME_LABELS
                render_heatmap_cards(n_total, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                use_container_width=True, key="a1_place")
            with a1_ind:
                st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Employee Experience")
                mat = spearman_matrix(filtered, WOW_IND_COLS, OUTCOME_COLS)
                mat.index   = WOW_THEMES
                mat.columns = OUTCOME_LABELS
                render_heatmap_cards(n_total, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                use_container_width=True, key="a1_ind")

        # ── A2: Ways of Working × Ways of Working ────────────
        with a2:
            a2_place, a2_ind = st.tabs(["Place (P)", "Individual (I)"])
            with a2_place:
                st.markdown("#### Correlational Heatmap: Ways of Working (Place) × Ways of Working (Place)")
                mat = make_writable_matrix(spearman_matrix(filtered, WOW_PLACE_COLS, WOW_PLACE_COLS))
                mat.index = mat.columns = WOW_THEMES
                fill_diagonal_with_nan(mat)
                render_heatmap_cards(n_total, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                use_container_width=True, key="a2_place")
            with a2_ind:
                st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Ways of Working (Individual)")
                mat = make_writable_matrix(spearman_matrix(filtered, WOW_IND_COLS, WOW_IND_COLS))
                mat.index = mat.columns = WOW_THEMES
                fill_diagonal_with_nan(mat)
                render_heatmap_cards(n_total, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                use_container_width=True, key="a2_ind")

        # ── A3: Outcomes × Outcomes ───────────────────────────
        with a3:
            st.markdown("#### Correlational Heatmap: Employee Experience × Employee Experience")
            mat = make_writable_matrix(spearman_matrix(filtered, OUTCOME_COLS, OUTCOME_COLS))
            mat.index = mat.columns = OUTCOME_LABELS
            fill_diagonal_with_nan(mat)
            render_heatmap_cards(n_total, mat)
            st.plotly_chart(make_heatmap(mat, OUTCOME_LABELS, OUTCOME_LABELS),
                            use_container_width=True, key="a3")

    # ── Descriptive Analysis group ────────────────────────
    with desc_group:
        a4, a5, a6 = st.tabs([
            "A4 · Ways of Working",
            "A5 · Sentiment Outcomes",
            "A6 · By Org Level",
        ])

        # ── A4: Ways of Working descriptive table ─────────────
        with a4:
            st.markdown("#### Ways of Working — Average Scores by Directorate")
            a4_table, a4_chart = st.tabs(["Table", "Bar Chart"])
            with a4_table:
                tdf, styler = build_wow_table(filtered, "Q1", directorates)
                wow_table_cards(n_total, tdf)
                st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                            "**P** = Place average &nbsp;|&nbsp; "
                            "**Δ** = I − P")
                st.dataframe(styler, use_container_width=True, height=720)
            with a4_chart:
                st.markdown(
                    '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                    'Select group</p>', unsafe_allow_html=True)
                dir_options = ["Overall"] + list(directorates)
                chart_dir = st.radio("Select group", dir_options, horizontal=True,
                                     label_visibility="collapsed", key="a4_dir_chart")
                chart_df = filtered if chart_dir == "Overall" else filtered[filtered["Q1"] == chart_dir]
                n_col, _ = st.columns([1, 5])
                with n_col:
                    st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                unsafe_allow_html=True)
                st.plotly_chart(make_wow_bar_chart(chart_df), use_container_width=True, key="a4_bar")

        # ── A5: Sentiment Outcomes descriptive table ──────────
        with a5:
            st.markdown("#### Employee Experience — Average Scores by Directorate")
            a5_table, a5_chart = st.tabs(["Table", "Bar Chart"])
            with a5_table:
                tdf, styler = build_outcome_table(filtered, "Q1", directorates)
                outcome_table_cards(n_total, tdf)
                st.dataframe(styler, use_container_width=True, height=580)
            with a5_chart:
                st.markdown(
                    '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                    'Select group</p>', unsafe_allow_html=True)
                dir_options = ["Overall"] + list(directorates)
                chart_dir = st.radio("Select group", dir_options, horizontal=True,
                                     label_visibility="collapsed", key="a5_dir_chart")
                chart_df = filtered if chart_dir == "Overall" else filtered[filtered["Q1"] == chart_dir]
                overall_df = None if chart_dir == "Overall" else filtered
                n_col, _ = st.columns([1, 5])
                with n_col:
                    st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                unsafe_allow_html=True)
                st.plotly_chart(make_outcome_bar_chart(chart_df, overall_df), use_container_width=True,
                                key=f"a5_bar_{chart_dir}")

        # ── A6: By Q9 Organisational Level ───────────────────
        with a6:
            q9_ordered = sorted(q9_levels,
                                key=lambda x: next(
                                    (i for i, o in enumerate(Q9_ORDER) if o.lower() == x.lower()),
                                    len(Q9_ORDER)))

            st.markdown("#### Ways of Working — Average Scores by Organisational Level")
            a6_wow_table, a6_wow_chart = st.tabs(["Table", "Bar Chart"])
            with a6_wow_table:
                tdf_wow, styler_wow = build_wow_table(filtered, "Q9", q9_levels)
                wow_table_cards(n_total, tdf_wow)
                st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                            "**P** = Place average &nbsp;|&nbsp; "
                            "**Δ** = I − P")
                st.dataframe(styler_wow, use_container_width=True, height=720)
            with a6_wow_chart:
                st.markdown(
                    '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                    'Select org level</p>', unsafe_allow_html=True)
                sel_q9_wow = st.radio("Select org level", q9_ordered, horizontal=True,
                                      label_visibility="collapsed", key="a6_wow_q9")
                chart_df = filtered[filtered["Q9"] == sel_q9_wow]
                n_col, _ = st.columns([1, 5])
                with n_col:
                    st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                unsafe_allow_html=True)
                st.plotly_chart(make_wow_bar_chart(chart_df), use_container_width=True, key="a6_wow_bar")

            st.markdown("---")
            st.markdown("#### Employee Experience — Average Scores by Organisational Level")
            a6_out_table, a6_out_chart = st.tabs(["Table", "Bar Chart"])
            with a6_out_table:
                tdf_out, styler_out = build_outcome_table(filtered, "Q9", q9_levels)
                outcome_table_cards(n_total, tdf_out)
                st.dataframe(styler_out, use_container_width=True, height=580)
            with a6_out_chart:
                st.markdown(
                    '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                    'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                    'Select org level</p>', unsafe_allow_html=True)
                sel_q9_out = st.radio("Select org level", q9_ordered, horizontal=True,
                                      label_visibility="collapsed", key="a6_out_q9")
                chart_df = filtered[filtered["Q9"] == sel_q9_out]
                n_col, _ = st.columns([1, 5])
                with n_col:
                    st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                unsafe_allow_html=True)
                st.plotly_chart(make_outcome_bar_chart(chart_df), use_container_width=True, key="a6_out_bar")


# ═══════════════════════════════════════════════════════════════════════════════════
# SECTION B
# ═══════════════════════════════════════════════════════════════════════════════════
with sec_b:
    if not directorates:
        st.warning("No directorate data found in this file.")
        st.stop()

    st.markdown(
        '<p style="font-size:13px;font-weight:600;color:#5A7080;text-transform:uppercase;'
        'letter-spacing:0.06em;margin-bottom:4px">Select Directorate</p>',
        unsafe_allow_html=True,
    )
    selected_dir = st.radio(
        "Directorate",
        directorates,
        horizontal=True,
        label_visibility="collapsed",
        key="dir_selector",
    )

    dir_df = filtered[filtered["Q1"] == selected_dir]
    n_dir  = len(dir_df)

    service_areas = get_filter_options(dir_df, "service_area")

    st.caption(f"{n_dir:,} respondents in **{selected_dir}**")

    b_corr_group, b_desc_group = st.tabs(["Correlation Analysis", "Descriptive Analysis"])

    # ── Correlation Analysis group ────────────────────────
    with b_corr_group:
        b1, b2, b3 = st.tabs([
            "B1 · WoW × Outcomes",
            "B2 · WoW × WoW",
            "B3 · Outcomes × Outcomes",
        ])

        # ── B1 ────────────────────────────────────────────────
        with b1:
            if n_dir < 3:
                st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
            else:
                b1_place, b1_ind = st.tabs(["Place (P)", "Individual (I)"])
                with b1_place:
                    st.markdown("#### Correlational Heatmap: Ways of Working (Place) × Employee Experience")
                    mat = spearman_matrix(dir_df, WOW_PLACE_COLS, OUTCOME_COLS)
                    mat.index   = WOW_THEMES
                    mat.columns = OUTCOME_LABELS
                    render_heatmap_cards(n_dir, mat)
                    st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                    use_container_width=True, key="b1_place")
                with b1_ind:
                    st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Employee Experience")
                    mat = spearman_matrix(dir_df, WOW_IND_COLS, OUTCOME_COLS)
                    mat.index   = WOW_THEMES
                    mat.columns = OUTCOME_LABELS
                    render_heatmap_cards(n_dir, mat)
                    st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                    use_container_width=True, key="b1_ind")

        # ── B2 ────────────────────────────────────────────────
        with b2:
            if n_dir < 3:
                st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
            else:
                b2_place, b2_ind = st.tabs(["Place (P)", "Individual (I)"])
                with b2_place:
                    st.markdown("#### Correlational Heatmap: Ways of Working (Place) × Ways of Working (Place)")
                    mat = make_writable_matrix(spearman_matrix(dir_df, WOW_PLACE_COLS, WOW_PLACE_COLS))
                    mat.index = mat.columns = WOW_THEMES
                    fill_diagonal_with_nan(mat)
                    render_heatmap_cards(n_dir, mat)
                    st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                    use_container_width=True, key="b2_place")
                with b2_ind:
                    st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Ways of Working (Individual)")
                    mat = make_writable_matrix(spearman_matrix(dir_df, WOW_IND_COLS, WOW_IND_COLS))
                    mat.index = mat.columns = WOW_THEMES
                    fill_diagonal_with_nan(mat)
                    render_heatmap_cards(n_dir, mat)
                    st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                    use_container_width=True, key="b2_ind")

        # ── B3 ────────────────────────────────────────────────
        with b3:
            if n_dir < 3:
                st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
            else:
                st.markdown("#### Correlational Heatmap: Employee Experience × Employee Experience")
                mat = make_writable_matrix(spearman_matrix(dir_df, OUTCOME_COLS, OUTCOME_COLS))
                mat.index = mat.columns = OUTCOME_LABELS
                fill_diagonal_with_nan(mat)
                render_heatmap_cards(n_dir, mat)
                st.plotly_chart(make_heatmap(mat, OUTCOME_LABELS, OUTCOME_LABELS),
                                use_container_width=True, key="b3")

    # ── Descriptive Analysis group ────────────────────────
    with b_desc_group:
        b4, b5 = st.tabs([
            "B4 · Ways of Working",
            "B5 · Sentiment Outcomes",
        ])

        # ── B4 ────────────────────────────────────────────────
        with b4:
            if not service_areas:
                st.info("No service area breakdown available for this directorate.")
            else:
                st.markdown(f"#### Ways of Working — Average Scores by Service Area ({selected_dir})")
                b4_table, b4_chart = st.tabs(["Table", "Bar Chart"])
                with b4_table:
                    tdf, styler = build_wow_table(dir_df, "service_area", service_areas)
                    wow_table_cards(n_dir, tdf)
                    st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                                "**P** = Place average &nbsp;|&nbsp; "
                                "**Δ** = I − P")
                    st.dataframe(styler, use_container_width=True, height=720)
                with b4_chart:
                    st.markdown(
                        '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                        'Select service area</p>', unsafe_allow_html=True)
                    sa_options = ["Overall"] + list(service_areas)
                    chart_sa = st.radio("Select service area", sa_options, horizontal=True,
                                        label_visibility="collapsed", key="b4_sa_chart")
                    chart_df = dir_df if chart_sa == "Overall" else dir_df[dir_df["service_area"] == chart_sa]
                    n_col, _ = st.columns([1, 5])
                    with n_col:
                        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                    f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                    unsafe_allow_html=True)
                    st.plotly_chart(make_wow_bar_chart(chart_df), use_container_width=True, key="b4_bar")

        # ── B5 ────────────────────────────────────────────────
        with b5:
            if not service_areas:
                st.info("No service area breakdown available for this directorate.")
            else:
                st.markdown(f"#### Employee Experience — Average Scores by Service Area ({selected_dir})")
                b5_table, b5_chart = st.tabs(["Table", "Bar Chart"])
                with b5_table:
                    tdf, styler = build_outcome_table(dir_df, "service_area", service_areas)
                    outcome_table_cards(n_dir, tdf)
                    st.dataframe(styler, use_container_width=True, height=580)
                with b5_chart:
                    st.markdown(
                        '<p style="font-size:13px;font-weight:600;color:#5A7080;'
                        'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">'
                        'Select service area</p>', unsafe_allow_html=True)
                    sa_options = ["Overall"] + list(service_areas)
                    chart_sa = st.radio("Select service area", sa_options, horizontal=True,
                                        label_visibility="collapsed", key="b5_sa_chart")
                    chart_df = dir_df if chart_sa == "Overall" else dir_df[dir_df["service_area"] == chart_sa]
                    overall_df = None if chart_sa == "Overall" else dir_df
                    n_col, _ = st.columns([1, 5])
                    with n_col:
                        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                                    f'<p class="card-value">n = {len(chart_df):,}</p></div>',
                                    unsafe_allow_html=True)
                    st.plotly_chart(make_outcome_bar_chart(chart_df, overall_df), use_container_width=True,
                                    key=f"b5_bar_{chart_sa}")
