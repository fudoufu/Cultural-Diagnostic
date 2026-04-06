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
.main, .block-container { background-color: #F7F9FC; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
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
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p {
    color: #1A2B3C !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────────
WOW_THEMES = [
    "Collective Responsibility", "Individual Accountability",
    "Hierarchical Control", "Empowered Decision-Making",
    "Prioritise People", "Protect Performance",
    "Challenge Decisions", "Preserve Cohesion",
    "Follow Procedures", "Adapt to Situation",
    "Consolidate Existing Processes", "Experiment & Innovate",
    "Deliver Immediate Results", "Invest in Long-Term Changes",
    "Proactive Learning", "Reactive Learning",
    "Target-Driven Interactions", "Relationship-Led Working",
    "Plan-Based Working", "Agile Working",
    "Value Individual Performance", "Value Individual Status",
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


def render_heatmap_cards(n: int, matrix: pd.DataFrame):
    flat = matrix.stack().dropna()
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                    f'<p class="card-value">n = {n:,}</p></div>', unsafe_allow_html=True)
    if flat.empty:
        return
    max_idx, min_idx = flat.idxmax(), flat.idxmin()
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Strongest positive correlation</p>'
            f'<p class="card-sub">{max_idx[0]} × {max_idx[1]}</p>'
            f'<p class="card-value">r = {flat.max():.3f}</p></div>',
            unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Strongest negative correlation</p>'
            f'<p class="card-sub">{min_idx[0]} × {min_idx[1]}</p>'
            f'<p class="card-value">r = {flat.min():.3f}</p></div>',
            unsafe_allow_html=True)


def render_summary_cards(n: int, high_label: str, high_val: str,
                         low_label: str, low_val: str):
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><p class="card-label">Respondents</p>'
                    f'<p class="card-value">n = {n:,}</p></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Highest scoring</p>'
            f'<p class="card-sub">{high_label}</p>'
            f'<p class="card-value">{high_val}</p></div>',
            unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><p class="card-label">Lowest scoring</p>'
            f'<p class="card-sub">{low_label}</p>'
            f'<p class="card-value">{low_val}</p></div>',
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
        render_summary_cards(n, "—", "—", "—", "—")
        return
    render_summary_cards(n,
                         combined.idxmax(), f"{combined.max():.2f}",
                         combined.idxmin(), f"{combined.min():.2f}")


def outcome_table_cards(n, tdf):
    ov = pd.to_numeric(tdf["Overall"], errors="coerce").dropna()
    if ov.empty:
        render_summary_cards(n, "—", "—", "—", "—")
        return
    render_summary_cards(n,
                         ov.idxmax(), f"{ov.max():.2f}",
                         ov.idxmin(), f"{ov.min():.2f}")


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
        <div style="text-align:center;padding:90px 40px;color:#1A2B3C">
            <div style="font-size:52px;margin-bottom:16px">📊</div>
            <h2 style="color:{PRIMARY};margin-bottom:8px">Cultural Diagnostic Dashboard</h2>
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
    "Section A — Whole Council",
    "Section B — By Directorate",
])

# ═══════════════════════════════════════════════════════════════════════════════════
# SECTION A
# ═══════════════════════════════════════════════════════════════════════════════════
with sec_a:
    a1, a2, a3, a4, a5, a6 = st.tabs([
        "A1 · WoW × Outcomes",
        "A2 · WoW × WoW",
        "A3 · Outcomes × Outcomes",
        "A4 · Ways of Working",
        "A5 · Sentiment Outcomes",
        "A6 · By Org Level",
    ])

    # ── A1: Ways of Working × Outcomes ───────────────────
    with a1:
        a1_place, a1_ind = st.tabs(["Place (P)", "Individual (I)"])
        with a1_place:
            mat = spearman_matrix(filtered, WOW_PLACE_COLS, OUTCOME_COLS)
            mat.index   = WOW_THEMES
            mat.columns = OUTCOME_LABELS
            render_heatmap_cards(n_total, mat)
            st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                            use_container_width=True)
        with a1_ind:
            mat = spearman_matrix(filtered, WOW_IND_COLS, OUTCOME_COLS)
            mat.index   = WOW_THEMES
            mat.columns = OUTCOME_LABELS
            render_heatmap_cards(n_total, mat)
            st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                            use_container_width=True)

    # ── A2: Ways of Working × Ways of Working ────────────
    with a2:
        a2_place, a2_ind = st.tabs(["Place (P)", "Individual (I)"])
        with a2_place:
            mat = make_writable_matrix(spearman_matrix(filtered, WOW_PLACE_COLS, WOW_PLACE_COLS))
            mat.index = mat.columns = WOW_THEMES
            fill_diagonal_with_nan(mat)
            render_heatmap_cards(n_total, mat)
            st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                            use_container_width=True)
        with a2_ind:
            mat = make_writable_matrix(spearman_matrix(filtered, WOW_IND_COLS, WOW_IND_COLS))
            mat.index = mat.columns = WOW_THEMES
            fill_diagonal_with_nan(mat)
            render_heatmap_cards(n_total, mat)
            st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                            use_container_width=True)

    # ── A3: Outcomes × Outcomes ───────────────────────────
    with a3:
        mat = make_writable_matrix(spearman_matrix(filtered, OUTCOME_COLS, OUTCOME_COLS))
        mat.index = mat.columns = OUTCOME_LABELS
        fill_diagonal_with_nan(mat)
        render_heatmap_cards(n_total, mat)
        st.plotly_chart(make_heatmap(mat, OUTCOME_LABELS, OUTCOME_LABELS),
                        use_container_width=True)

    # ── A4: Ways of Working descriptive table ─────────────
    with a4:
        tdf, styler = build_wow_table(filtered, "Q1", directorates)
        wow_table_cards(n_total, tdf)
        st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                    "**P** = Place average &nbsp;|&nbsp; "
                    "**Δ** = I − P")
        st.dataframe(styler, use_container_width=True, height=720)

    # ── A5: Sentiment Outcomes descriptive table ──────────
    with a5:
        tdf, styler = build_outcome_table(filtered, "Q1", directorates)
        outcome_table_cards(n_total, tdf)
        st.dataframe(styler, use_container_width=True, height=580)

    # ── A6: By Q9 Organisational Level ───────────────────
    with a6:
        st.markdown("##### Ways of Working by Organisational Level")
        tdf_wow, styler_wow = build_wow_table(filtered, "Q9", q9_levels)
        wow_table_cards(n_total, tdf_wow)
        st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                    "**P** = Place average &nbsp;|&nbsp; "
                    "**Δ** = I − P")
        st.dataframe(styler_wow, use_container_width=True, height=720)

        st.markdown("---")
        st.markdown("##### Sentiment Outcomes by Organisational Level")
        tdf_out, styler_out = build_outcome_table(filtered, "Q9", q9_levels)
        outcome_table_cards(n_total, tdf_out)
        st.dataframe(styler_out, use_container_width=True, height=580)


# ═══════════════════════════════════════════════════════════════════════════════════
# SECTION B
# ═══════════════════════════════════════════════════════════════════════════════════
with sec_b:
    if not directorates:
        st.warning("No directorate data found in this file.")
        st.stop()

    st.markdown("##### Select Directorate")
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
    st.markdown("---")

    b1, b2, b3, b4, b5 = st.tabs([
        "B1 · WoW × Outcomes",
        "B2 · WoW × WoW",
        "B3 · Outcomes × Outcomes",
        "B4 · Ways of Working",
        "B5 · Sentiment Outcomes",
    ])

    # ── B1 ────────────────────────────────────────────────
    with b1:
        if n_dir < 3:
            st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
        else:
            b1_place, b1_ind = st.tabs(["Place (P)", "Individual (I)"])
            with b1_place:
                mat = spearman_matrix(dir_df, WOW_PLACE_COLS, OUTCOME_COLS)
                mat.index   = WOW_THEMES
                mat.columns = OUTCOME_LABELS
                render_heatmap_cards(n_dir, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                use_container_width=True)
            with b1_ind:
                mat = spearman_matrix(dir_df, WOW_IND_COLS, OUTCOME_COLS)
                mat.index   = WOW_THEMES
                mat.columns = OUTCOME_LABELS
                render_heatmap_cards(n_dir, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, OUTCOME_LABELS),
                                use_container_width=True)

    # ── B2 ────────────────────────────────────────────────
    with b2:
        if n_dir < 3:
            st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
        else:
            b2_place, b2_ind = st.tabs(["Place (P)", "Individual (I)"])
            with b2_place:
                mat = make_writable_matrix(spearman_matrix(dir_df, WOW_PLACE_COLS, WOW_PLACE_COLS))
                mat.index = mat.columns = WOW_THEMES
                fill_diagonal_with_nan(mat)
                render_heatmap_cards(n_dir, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                use_container_width=True)
            with b2_ind:
                mat = make_writable_matrix(spearman_matrix(dir_df, WOW_IND_COLS, WOW_IND_COLS))
                mat.index = mat.columns = WOW_THEMES
                fill_diagonal_with_nan(mat)
                render_heatmap_cards(n_dir, mat)
                st.plotly_chart(make_heatmap(mat, WOW_THEMES, WOW_THEMES),
                                use_container_width=True)

    # ── B3 ────────────────────────────────────────────────
    with b3:
        if n_dir < 3:
            st.warning(f"Too few respondents ({n_dir}) for correlation analysis.")
        else:
            mat = make_writable_matrix(spearman_matrix(dir_df, OUTCOME_COLS, OUTCOME_COLS))
            mat.index = mat.columns = OUTCOME_LABELS
            fill_diagonal_with_nan(mat)
            render_heatmap_cards(n_dir, mat)
            st.plotly_chart(make_heatmap(mat, OUTCOME_LABELS, OUTCOME_LABELS),
                            use_container_width=True)

    # ── B4 ────────────────────────────────────────────────
    with b4:
        if not service_areas:
            st.info("No service area breakdown available for this directorate.")
        else:
            tdf, styler = build_wow_table(dir_df, "service_area", service_areas)
            wow_table_cards(n_dir, tdf)
            st.markdown(f"**{selected_dir}** — Ways of Working by Service Area")
            st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                        "**P** = Place average &nbsp;|&nbsp; "
                        "**Δ** = I − P")
            st.dataframe(styler, use_container_width=True, height=720)

    # ── B5 ────────────────────────────────────────────────
    with b5:
        if not service_areas:
            st.info("No service area breakdown available for this directorate.")
        else:
            tdf, styler = build_outcome_table(dir_df, "service_area", service_areas)
            outcome_table_cards(n_dir, tdf)
            st.markdown(f"**{selected_dir}** — Sentiment Outcomes by Service Area")
            st.dataframe(styler, use_container_width=True, height=580)
