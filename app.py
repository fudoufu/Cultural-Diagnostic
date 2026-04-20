import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from factor_analyzer import FactorAnalyzer
import semopy

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
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
}
.metric-card .card-label { color: #5A7080; font-size: 11px; font-weight: 500; margin: 0; }
.metric-card .card-value { color: #0F4C6B; font-size: 15px; font-weight: 700; margin: 2px 0 0; }
.metric-card .card-sub   { color: #1A2B3C; font-size: 12px; font-weight: 500; margin: 1px 0 0; }
.metric-card-lg {
    background: #FFFFFF;
    border: 1px solid #D6E0EA;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.metric-card-lg .card-label { color: #5A7080; font-size: 12px; font-weight: 500; margin: 0; }
.metric-card-lg .card-value { color: #0F4C6B; font-size: 20px; font-weight: 700; margin: 4px 0 0; }
.metric-card-lg .card-sub   { color: #1A2B3C; font-size: 13px; font-weight: 500; margin: 2px 0 0; }
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
[data-testid="stCaptionContainer"] p, .stCaption {
    color: #3A4D5C !important;
    font-size: 13px !important;
}
[data-testid="stSlider"] label, [data-testid="stSlider"] p,
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #1A2B3C !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #1A2B3C !important;
}
details summary, [data-testid="stExpander"] summary p {
    color: #1A2B3C !important;
    font-weight: 500 !important;
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

TREND_COLOURS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
    "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5",
    "#393b79","#637939",
]

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
        textfont={"size": 9, "color": "#1A2B3C"},
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


def make_trend_chart(df: pd.DataFrame, wow_cols: list, wow_labels: list,
                     outcome_col: str, outcome_label: str) -> go.Figure:
    """Trend lines (linear regression) for each WoW theme vs a selected outcome."""
    fig = go.Figure()
    for col, label, colour in zip(wow_cols, wow_labels, TREND_COLOURS):
        valid = df[[col, outcome_col]].dropna()
        if len(valid) < 3:
            continue
        x = valid[col].values.astype(float)
        y = valid[outcome_col].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([1.0, 5.0])
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name=label,
            line=dict(color=colour, width=2),
            hovertemplate=f"{label}<br>WoW score: %{{x:.0f}}<br>{outcome_label}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        font=dict(family="Inter", color="#1A2B3C"),
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#F7F9FC",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title=dict(text="WoW Rating (1–5)", font=dict(color="#1A2B3C", size=12)),
            tickvals=[1, 2, 3, 4, 5],
            range=[0.8, 5.2],
            tickfont=dict(color="#1A2B3C"),
            gridcolor="#E8EEF2",
        ),
        yaxis=dict(
            title=dict(text=outcome_label, font=dict(color="#1A2B3C", size=12)),
            tickvals=[1, 2, 3, 4, 5],
            range=[0.8, 5.2],
            tickfont=dict(color="#1A2B3C"),
            gridcolor="#E8EEF2",
        ),
        legend=dict(
            font=dict(color="#1A2B3C", size=11),
            orientation="v",
            x=1.01, y=1,
            xanchor="left",
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


def render_wow_variance_card(matrix: pd.DataFrame, top_n: int = 5):
    """Show top N WoW themes by collective R² (sum across all outcome columns)."""
    r2_sum = (matrix ** 2).sum(axis=1).sort_values(ascending=False)
    rows_html = "".join(
        f'<p class="card-sub">{wow}</p><p class="card-value">collective R² = {val:.3f}</p>'
        for wow, val in r2_sum.head(top_n).items()
    )
    st.markdown(
        f'<div class="metric-card"><p class="card-label">'
        f'Top {top_n} WoW themes explaining most variation in employee experience</p>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )


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


# ── SEM / EFA helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running EFA…")
def run_efa(outcome_df: pd.DataFrame, n_factors: int):
    clean = outcome_df.dropna()
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="minres")
    fa.fit(clean)
    loadings = pd.DataFrame(
        fa.loadings_,
        index=outcome_df.columns.tolist(),
        columns=[f"LV{i+1}" for i in range(n_factors)],
    )
    return loadings, len(clean)


@st.cache_data(show_spinner="Fitting SEM — this may take a moment…")
def run_sem(fit_df: pd.DataFrame, wow_cols: tuple, factor_map_items: tuple):
    """factor_map_items: tuple of (outcome_col, lv_name) pairs (hashable for cache)."""
    factor_map = dict(factor_map_items)
    lv_names = sorted(set(factor_map.values()))

    meas_lines, valid_lvs = [], []
    for lv in lv_names:
        items = [k for k, v in factor_map.items() if v == lv]
        if len(items) >= 2:
            meas_lines.append(f"{lv} =~ {' + '.join(items)}")
            valid_lvs.append(lv)

    if not valid_lvs:
        return None, None, "No latent variable has ≥2 assigned outcomes.", ""

    struct_lines = [f"{lv} ~ {' + '.join(wow_cols)}" for lv in valid_lvs]
    model_desc = "\n".join(meas_lines + struct_lines)

    try:
        model = semopy.Model(model_desc)
        model.fit(fit_df)
        results = model.inspect(mode="list")
        try:
            fit_stats = semopy.calc_stats(model)
        except Exception:
            fit_stats = None
        return results, fit_stats, None, model_desc
    except Exception as e:
        return None, None, str(e), model_desc


@st.cache_data(show_spinner="Running backward elimination…")
def run_backward_elimination(
    df: pd.DataFrame, outcome_col: str, predictor_cols: tuple, p_threshold: float
):
    import statsmodels.api as sm

    fit_df = df[list(predictor_cols) + [outcome_col]].dropna()
    predictors = list(predictor_cols)
    y = fit_df[outcome_col].values.astype(float)
    elim_log = []
    step = 0

    while predictors:
        X = sm.add_constant(fit_df[predictors].values.astype(float), has_constant="add")
        model = sm.OLS(y, X).fit()
        pvals = pd.Series(model.pvalues[1:], index=predictors)
        max_p = pvals.max()
        worst = pvals.idxmax()

        elim_log.append({
            "Step": step,
            "Predictors in model": len(predictors),
            "Adj. R²": round(model.rsquared_adj, 4),
            "Removed this step": worst if max_p > p_threshold else "—",
            "p-value of removed": round(float(max_p), 4) if max_p > p_threshold else "—",
        })

        if max_p <= p_threshold:
            break

        predictors.remove(worst)
        step += 1

    if not predictors:
        return elim_log, None, None, None, None, None, len(fit_df), []

    X_final = fit_df[predictors].values.astype(float)
    X_const = sm.add_constant(X_final, has_constant="add")
    final = sm.OLS(y, X_const).fit()

    y_std = y.std()
    coef_df = pd.DataFrame({
        "col":       predictors,
        "β":         final.params[1:],
        "std_err":   final.bse[1:],
        "p_value":   final.pvalues[1:],
        "β_std":     [
            final.params[i + 1] * fit_df[predictors[i]].std() / y_std
            for i in range(len(predictors))
        ],
    })

    return (
        elim_log,
        coef_df,
        final.fittedvalues.tolist(),
        final.resid.tolist(),
        float(final.rsquared_adj),
        float(final.rsquared),
        len(fit_df),
        predictors,
    )


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
    "Section A — Council-Wide & By Directorate",
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
                render_wow_variance_card(mat)
                st.plotly_chart(make_heatmap(mat.T, OUTCOME_LABELS, WOW_THEMES),
                                use_container_width=True, key="a1_place")
                st.markdown("---")
                st.markdown("#### Trend Lines: Ways of Working (Place) vs Employee Experience")
                sel_out = st.selectbox("Select outcome", OUTCOME_LABELS, key="a1_place_out")
                out_col = OUTCOME_COLS[OUTCOME_LABELS.index(sel_out)]
                st.plotly_chart(make_trend_chart(filtered, WOW_PLACE_COLS, WOW_THEMES, out_col, sel_out),
                                use_container_width=True, key=f"a1_place_trend_{sel_out}")
            with a1_ind:
                st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Employee Experience")
                mat = spearman_matrix(filtered, WOW_IND_COLS, OUTCOME_COLS)
                mat.index   = WOW_THEMES
                mat.columns = OUTCOME_LABELS
                render_heatmap_cards(n_total, mat)
                render_wow_variance_card(mat)
                st.plotly_chart(make_heatmap(mat.T, OUTCOME_LABELS, WOW_THEMES),
                                use_container_width=True, key="a1_ind")
                st.markdown("---")
                st.markdown("#### Trend Lines: Ways of Working (Individual) vs Employee Experience")
                sel_out = st.selectbox("Select outcome", OUTCOME_LABELS, key="a1_ind_out")
                out_col = OUTCOME_COLS[OUTCOME_LABELS.index(sel_out)]
                st.plotly_chart(make_trend_chart(filtered, WOW_IND_COLS, WOW_THEMES, out_col, sel_out),
                                use_container_width=True, key=f"a1_ind_trend_{sel_out}")

            # ── Advanced Analysis tabs ─────────────────────────────────
            st.markdown("---")
            a1_adv_sem, a1_adv_model = st.tabs([
                "Exploratory Factor Analysis & SEM",
                "Second-Level Modelling",
            ])

            with a1_adv_sem:
                st.caption(
                    "Step 1: EFA identifies hidden clusters in your employee experience outcomes. "
                    "Step 2: SEM then estimates how each WoW theme predicts those clusters, "
                    "accounting for intercorrelations between WoW themes."
                )

                outcome_clean_sem = filtered[OUTCOME_COLS].dropna()
                if len(outcome_clean_sem) < 20:
                    st.warning("Too few complete responses for SEM analysis.")
                else:
                    corr_mat = np.corrcoef(outcome_clean_sem.values.T)
                    eigenvalues = sorted(np.linalg.eigvalsh(corr_mat).tolist(), reverse=True)
                    n_kaiser = max(2, sum(1 for e in eigenvalues if e > 1))

                    st.markdown("#### Step 1 · Scree Plot — How many hidden factors exist in your outcome data?")
                    st.caption(
                        "This graph helps you decide how many distinct themes exist within your employee experience "
                        "questions. Each point is one potential theme (factor), and its height shows how much shared "
                        "information it captures across your outcome questions. The dashed line is a cut-off rule: "
                        "factors above it are capturing more than a single question's worth of information on their "
                        "own, so they're worth keeping. Look for where the curve suddenly goes from steep to flat — "
                        "that's the point where additional factors stop adding meaningful new information. The factors "
                        "to the left of that bend are your real underlying themes. Use the slider below to set how "
                        "many to extract."
                    )
                    fig_scree = go.Figure([go.Scatter(
                        x=list(range(1, len(eigenvalues) + 1)), y=eigenvalues,
                        mode="lines+markers",
                        line=dict(color=PRIMARY, width=2),
                        marker=dict(color=PRIMARY, size=7),
                        hovertemplate="Factor %{x}<br>Eigenvalue = %{y:.3f}<extra></extra>",
                    )])
                    fig_scree.add_hline(
                        y=1, line_dash="dash", line_color="#C0392B",
                        annotation_text="Kaiser criterion (λ = 1)",
                        annotation_font=dict(color="#C0392B", size=11),
                    )
                    fig_scree.update_layout(
                        font=dict(family="Inter", color="#1A2B3C"),
                        paper_bgcolor="#F7F9FC", plot_bgcolor="#F7F9FC",
                        margin=dict(l=10, r=10, t=20, b=10),
                        xaxis=dict(title="Factor", tickvals=list(range(1, len(eigenvalues) + 1)),
                                   tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2"),
                        yaxis=dict(title="Eigenvalue", tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2"),
                        height=300,
                    )
                    st.plotly_chart(fig_scree, use_container_width=True, key="a1_sem_scree")

                    n_factors = st.slider(
                        "Number of latent factors", min_value=2,
                        max_value=min(8, len(OUTCOME_COLS) - 1),
                        value=n_kaiser, key="a1_sem_nfactors",
                        help="Factors with eigenvalue > 1 (Kaiser criterion) are suggested as a starting point",
                    )

                    loadings, n_efa = run_efa(filtered[OUTCOME_COLS], n_factors)
                    loadings_display = loadings.copy()
                    loadings_display.index = OUTCOME_LABELS
                    lv_cols = [f"LV{i+1}" for i in range(n_factors)]

                    st.markdown(f"#### Step 2 · Factor Loadings — Which outcomes cluster together? (n = {n_efa:,})")
                    st.caption(
                        "Each cell shows a loading value — a correlation score between that outcome question and "
                        "that factor, ranging from -1 to +1. A value above 0.7 means that question is a core part "
                        "of what that factor represents. Between 0.4 and 0.7 is a moderate association. Below 0.3 "
                        "means that question isn't really measuring this factor. Read down each column to see which "
                        "outcomes cluster together — that cluster is what gives the factor its meaning, and tells "
                        "you what to name it (e.g. if Intent to stay, Good place to work, and Employer rating all "
                        "load highly on LV1, you might call it 'Retention & Advocacy')."
                    )
                    max_load_pre = loadings_display.abs().max(axis=1)
                    primary_pre = loadings_display.abs().idxmax(axis=1).copy()
                    primary_pre[max_load_pre < 0.3] = "Unassigned"
                    factor_members = {lv: [] for lv in lv_cols}
                    factor_members["Unassigned"] = []
                    for outcome, lv in primary_pre.items():
                        factor_members[lv].append(outcome)

                    card_cols = st.columns(n_factors)
                    for col_ui, lv in zip(card_cols, lv_cols):
                        members = factor_members[lv]
                        items_html = "".join(
                            f'<p class="card-sub">· {m}</p>' for m in members
                        ) if members else '<p class="card-sub"><em>No outcomes assigned</em></p>'
                        with col_ui:
                            st.markdown(
                                f'<div class="metric-card-lg">'
                                f'<p class="card-label">{lv} — unnamed</p>'
                                f'{items_html}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    fig_load = make_heatmap(loadings_display, OUTCOME_LABELS, lv_cols)
                    st.plotly_chart(fig_load, use_container_width=True, key="a1_sem_loadings")

                    max_load = loadings_display.abs().max(axis=1)
                    primary = loadings_display.abs().idxmax(axis=1).copy()
                    primary[max_load < 0.3] = "Unassigned"

                    st.markdown("#### Step 3 · SEM Structural Paths — Which Ways of Working drive each factor?")
                    wow_choice_sem = st.radio(
                        "WoW predictors", ["Place (P)", "Individual (I)"],
                        horizontal=True, key="a1_sem_wow",
                    )
                    wow_cols_sem = tuple(WOW_PLACE_COLS if "Place" in wow_choice_sem else WOW_IND_COLS)

                    raw_primary = pd.Series(primary.values, index=OUTCOME_COLS)
                    factor_map_items = tuple(sorted(
                        (col, lv) for col, lv in raw_primary.items() if lv != "Unassigned"
                    ))

                    fit_df_sem = filtered[OUTCOME_COLS + list(wow_cols_sem)].dropna()

                    if len(fit_df_sem) < 100:
                        st.warning(f"Too few complete cases ({len(fit_df_sem)}) for SEM.")
                    elif not factor_map_items:
                        st.warning("No outcomes assigned to factors. Try reducing the number of factors.")
                    else:
                        results_sem, fit_stats_sem, err_sem, model_desc_sem = run_sem(
                            fit_df_sem, wow_cols_sem, factor_map_items
                        )
                        st.caption(
                            "The SEM estimates a path coefficient (β) from each WoW theme to each latent factor, "
                            "while simultaneously accounting for correlations between WoW themes. A positive β means "
                            "that WoW theme is associated with higher scores on that factor; negative means lower. "
                            "Unlike simple correlation, SEM isolates each WoW theme's unique contribution after "
                            "controlling for all the others."
                        )
                        with st.expander("Model specification"):
                            st.code(model_desc_sem, language="text")

                        if err_sem:
                            st.error(f"SEM could not converge: {err_sem}")
                        else:
                            if fit_stats_sem is not None:
                                try:
                                    fit_row = fit_stats_sem.iloc[0] if hasattr(fit_stats_sem, "iloc") else fit_stats_sem
                                    stat_keys = [k for k in ["CFI", "GFI", "RMSEA", "AIC"] if k in fit_row.index]
                                    fcols = st.columns(max(1, len(stat_keys)))
                                    for fc, sk in zip(fcols, stat_keys):
                                        val = fit_row[sk]
                                        with fc:
                                            st.markdown(
                                                f'<div class="metric-card">'
                                                f'<p class="card-label">{sk}</p>'
                                                f'<p class="card-value">'
                                                f'{"—" if pd.isna(val) else f"{val:.3f}"}'
                                                f'</p></div>',
                                                unsafe_allow_html=True,
                                            )
                                except Exception:
                                    pass

                            st.markdown("#### Model Fit Statistics — How well does the model fit the data?")
                            st.caption(
                                "CFI and GFI > 0.90 indicate acceptable fit (> 0.95 is good). "
                                "RMSEA < 0.08 is acceptable (< 0.05 is good). "
                                "AIC is used for comparing models — lower is better."
                            )

                            paths_sem = results_sem[
                                (results_sem["op"] == "~") &
                                (results_sem["rval"].isin(list(wow_cols_sem)))
                            ].copy()

                            if not paths_sem.empty:
                                col_to_label = dict(zip(wow_cols_sem, WOW_THEMES))
                                paths_sem["wow_label"] = paths_sem["rval"].map(col_to_label)
                                p_col = next(
                                    (c for c in paths_sem.columns if "p-value" in c.lower() or "p_value" in c.lower()),
                                    None,
                                )
                                pivot_est = paths_sem.pivot(index="wow_label", columns="lval", values="Estimate")

                                if p_col:
                                    paths_sem[p_col] = pd.to_numeric(paths_sem[p_col], errors="coerce")
                                    pivot_p = paths_sem.pivot(index="wow_label", columns="lval", values=p_col)
                                    text_vals = [
                                        [
                                            f"{e:.2f}{'***' if (not pd.isna(p) and p < 0.001) else '**' if (not pd.isna(p) and p < 0.01) else '*' if (not pd.isna(p) and p < 0.05) else ''}"
                                            for e, p in zip(row_e, row_p)
                                        ]
                                        for row_e, row_p in zip(pivot_est.values, pivot_p.values)
                                    ]
                                else:
                                    text_vals = [[f"{e:.2f}" for e in row] for row in pivot_est.values]

                                fig_paths = go.Figure(go.Heatmap(
                                    z=pivot_est.values,
                                    x=pivot_est.columns.tolist(),
                                    y=pivot_est.index.tolist(),
                                    text=text_vals,
                                    texttemplate="%{text}",
                                    textfont={"size": 9, "color": "#1A2B3C"},
                                    colorscale=HEATMAP_COLORSCALE,
                                    zmid=0,
                                    colorbar=dict(title="β", tickfont=dict(color="#1A2B3C")),
                                    hovertemplate="WoW: %{y}<br>Factor: %{x}<br>β = %{z:.3f}<extra></extra>",
                                ))
                                fig_paths.update_layout(
                                    font=dict(family="Inter", color="#1A2B3C"),
                                    paper_bgcolor="#F7F9FC", plot_bgcolor="#F7F9FC",
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    xaxis=dict(tickfont=dict(color="#1A2B3C"), side="top"),
                                    yaxis=dict(tickfont=dict(color="#1A2B3C"), autorange="reversed"),
                                    height=max(400, 28 * len(pivot_est)),
                                )
                                st.markdown("#### Path Coefficient Heatmap — WoW themes (rows) × Latent factors (columns)")
                                st.caption(
                                    "Each cell shows the standardised path coefficient (β) from that WoW theme to "
                                    "that factor. Red = positive effect (higher WoW score → higher factor score); "
                                    "blue = negative. Stars show statistical significance: * p<0.05, ** p<0.01, "
                                    "*** p<0.001. Cells without stars are not reliably different from zero — treat "
                                    "them with caution. Remember: LV1, LV2 etc. are named by you based on the "
                                    "factor loadings table above."
                                )
                                st.plotly_chart(fig_paths, use_container_width=True,
                                                key=f"a1_sem_paths_{wow_choice_sem}")

            with a1_adv_model:
                st.markdown("#### Second-Level Modelling — Nested Behavioural Model")
                st.caption(
                    "Finds the smallest combination of WoW themes that together best predict a chosen "
                    "employee experience outcome. Starting with all WoW themes as predictors, the model "
                    "removes them one by one — least significant first — until only those with an "
                    "independent, statistically reliable relationship remain. The key metric is Adjusted R²: "
                    "it tells you how much of the variation in that outcome is explained by the retained "
                    "WoW themes, penalised for adding predictors that don't earn their place."
                )

                # ── Controls ─────────────────────────────────────────────
                slm_c1, slm_c2, slm_c3 = st.columns([3, 3, 2])
                with slm_c1:
                    sel_outcome_slm = st.selectbox(
                        "Employee experience outcome", OUTCOME_LABELS, key="slm_outcome"
                    )
                with slm_c2:
                    wow_choice_slm = st.radio(
                        "WoW predictors", ["Place (P)", "Individual (I)", "Both"],
                        horizontal=True, key="slm_wow"
                    )
                with slm_c3:
                    p_thresh_slm = st.slider(
                        "P-value threshold", 0.01, 0.10, 0.05, 0.01, key="slm_pthresh"
                    )

                outcome_col_slm = OUTCOME_COLS[OUTCOME_LABELS.index(sel_outcome_slm)]
                if "Place" in wow_choice_slm:
                    pred_cols_slm = tuple(WOW_PLACE_COLS)
                    col_to_lbl_slm = dict(zip(WOW_PLACE_COLS, WOW_THEMES))
                elif "Individual" in wow_choice_slm:
                    pred_cols_slm = tuple(WOW_IND_COLS)
                    col_to_lbl_slm = dict(zip(WOW_IND_COLS, WOW_THEMES))
                else:
                    pred_cols_slm = tuple(WOW_ALL_COLS)
                    col_to_lbl_slm = dict(zip(WOW_ALL_COLS, WOW_ALL_LABELS))

                elim_log, coef_df, fitted_slm, resid_slm, adj_r2_slm, r2_slm, n_slm, retained_cols = \
                    run_backward_elimination(filtered, outcome_col_slm, pred_cols_slm, p_thresh_slm)

                if coef_df is None:
                    st.warning("All predictors were eliminated. Try raising the p-value threshold.")
                else:
                    # ── Summary cards ─────────────────────────────────────
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    for col_ui, label, value in [
                        (mc1, "Respondents",          f"n = {n_slm:,}"),
                        (mc2, "Adjusted R²",           f"{adj_r2_slm:.3f}"),
                        (mc3, "R²",                    f"{r2_slm:.3f}"),
                        (mc4, "WoW themes retained",  str(len(retained_cols))),
                    ]:
                        with col_ui:
                            st.markdown(
                                f'<div class="metric-card"><p class="card-label">{label}</p>'
                                f'<p class="card-value">{value}</p></div>',
                                unsafe_allow_html=True,
                            )

                    # ── Elimination path chart ────────────────────────────
                    st.markdown("#### Elimination Path — Adjusted R² at each step")
                    st.caption(
                        "Each point shows the model's Adjusted R² after removing the least significant "
                        "predictor at that step. If the line stays flat or rises as predictors are removed, "
                        "those predictors weren't adding real explanatory value. A meaningful drop signals "
                        "the model has reached its core — further removal would cost too much."
                    )
                    log_df = pd.DataFrame(elim_log)
                    fig_elim = go.Figure(go.Scatter(
                        x=log_df["Predictors in model"],
                        y=log_df["Adj. R²"],
                        mode="lines+markers",
                        line=dict(color=PRIMARY, width=2),
                        marker=dict(color=PRIMARY, size=8),
                        hovertemplate=(
                            "Predictors: %{x}<br>Adj. R² = %{y:.4f}"
                            "<extra></extra>"
                        ),
                    ))
                    fig_elim.update_layout(
                        font=dict(family="Inter", color="#1A2B3C"),
                        paper_bgcolor="#F7F9FC", plot_bgcolor="#F7F9FC",
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(
                            title="Number of predictors in model",
                            autorange="reversed",
                            tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2",
                        ),
                        yaxis=dict(
                            title="Adjusted R²",
                            tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2",
                        ),
                        height=320,
                    )
                    st.plotly_chart(fig_elim, use_container_width=True, key=f"slm_elim_{sel_outcome_slm}_{wow_choice_slm}")

                    # ── Coefficient chart ─────────────────────────────────
                    st.markdown("#### Final Model — Standardised Coefficients (β)")
                    st.caption(
                        "Each bar shows the standardised path coefficient for a retained WoW theme — how "
                        "much the outcome changes (in standard deviation units) for a 1-SD increase in that "
                        "WoW theme's rating, after controlling for all other retained themes. Longer bar = "
                        "stronger independent effect. Red = positive (more of this WoW → better outcome); "
                        "blue = negative."
                    )
                    coef_df["label"] = coef_df["col"].map(col_to_lbl_slm)
                    coef_df["sig"] = coef_df["p_value"].apply(
                        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    )
                    coef_df["text"] = coef_df.apply(lambda r: f"{r['β_std']:.2f}{r['sig']}", axis=1)
                    coef_sorted = coef_df.reindex(coef_df["β_std"].abs().sort_values().index)

                    fig_coef = go.Figure(go.Bar(
                        x=coef_sorted["β_std"],
                        y=coef_sorted["label"],
                        orientation="h",
                        text=coef_sorted["text"],
                        textposition="outside",
                        textfont=dict(color="#1A2B3C", size=11),
                        marker_color=[PRIMARY if v >= 0 else "#2E7096" for v in coef_sorted["β_std"]],
                        hovertemplate="<b>%{y}</b><br>β (std) = %{x:.3f}<extra></extra>",
                    ))
                    fig_coef.update_layout(
                        font=dict(family="Inter", color="#1A2B3C"),
                        paper_bgcolor="#F7F9FC", plot_bgcolor="#F7F9FC",
                        margin=dict(l=10, r=80, t=10, b=10),
                        xaxis=dict(
                            title="Standardised β", zeroline=True, zerolinecolor="#D6E0EA",
                            tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2",
                        ),
                        yaxis=dict(tickfont=dict(color="#1A2B3C")),
                        height=max(300, 36 * len(coef_sorted)),
                    )
                    st.plotly_chart(fig_coef, use_container_width=True, key=f"slm_coef_{sel_outcome_slm}_{wow_choice_slm}")

                    # ── Residual plot ─────────────────────────────────────
                    st.markdown("#### Residual Plot — Model Diagnostic")
                    st.caption(
                        "Each dot is one respondent. The X axis is what the model predicted for them; "
                        "the Y axis is how far off it was (actual minus predicted). A well-specified model "
                        "shows dots randomly scattered around the zero line with no pattern. A funnel shape "
                        "(spread widening left to right) suggests the model is less precise at higher scores. "
                        "A curve suggests the relationship isn't fully linear. Some imperfection is normal "
                        "with survey data."
                    )
                    fig_resid = go.Figure(go.Scatter(
                        x=fitted_slm, y=resid_slm,
                        mode="markers",
                        marker=dict(color=PRIMARY, opacity=0.35, size=5),
                        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
                    ))
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="#C0392B", line_width=1)
                    fig_resid.update_layout(
                        font=dict(family="Inter", color="#1A2B3C"),
                        paper_bgcolor="#F7F9FC", plot_bgcolor="#F7F9FC",
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(
                            title="Predicted value",
                            tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2",
                        ),
                        yaxis=dict(
                            title="Residual (actual − predicted)",
                            tickfont=dict(color="#1A2B3C"), gridcolor="#E8EEF2",
                        ),
                        height=360,
                    )
                    st.plotly_chart(fig_resid, use_container_width=True, key=f"slm_resid_{sel_outcome_slm}_{wow_choice_slm}")

                    # ── Elimination log ───────────────────────────────────
                    with st.expander("Elimination log"):
                        st.caption(
                            "The full step-by-step record of which predictor was removed at each iteration "
                            "and what the Adjusted R² was after that removal. Step 0 is the full model."
                        )
                        st.dataframe(log_df, use_container_width=True, hide_index=True)


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
            a4_chart, a4_table = st.tabs(["Bar Chart", "Table"])
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
            with a4_table:
                tdf, styler = build_wow_table(filtered, "Q1", directorates)
                wow_table_cards(n_total, tdf)
                st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                            "**P** = Place average &nbsp;|&nbsp; "
                            "**Δ** = I − P")
                st.dataframe(styler, use_container_width=True, height=720)

        # ── A5: Sentiment Outcomes descriptive table ──────────
        with a5:
            st.markdown("#### Employee Experience — Average Scores by Directorate")
            a5_chart, a5_table = st.tabs(["Bar Chart", "Table"])
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
            with a5_table:
                tdf, styler = build_outcome_table(filtered, "Q1", directorates)
                outcome_table_cards(n_total, tdf)
                st.dataframe(styler, use_container_width=True, height=580)

        # ── A6: By Q9 Organisational Level ───────────────────
        with a6:
            q9_ordered = sorted(q9_levels,
                                key=lambda x: next(
                                    (i for i, o in enumerate(Q9_ORDER) if o.lower() == x.lower()),
                                    len(Q9_ORDER)))

            st.markdown("#### Ways of Working — Average Scores by Organisational Level")
            a6_wow_chart, a6_wow_table = st.tabs(["Bar Chart", "Table"])
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
            with a6_wow_table:
                tdf_wow, styler_wow = build_wow_table(filtered, "Q9", q9_levels)
                wow_table_cards(n_total, tdf_wow)
                st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                            "**P** = Place average &nbsp;|&nbsp; "
                            "**Δ** = I − P")
                st.dataframe(styler_wow, use_container_width=True, height=720)

            st.markdown("---")
            st.markdown("#### Employee Experience — Average Scores by Organisational Level")
            a6_out_chart, a6_out_table = st.tabs(["Bar Chart", "Table"])
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
            with a6_out_table:
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
                    render_wow_variance_card(mat)
                    st.plotly_chart(make_heatmap(mat.T, OUTCOME_LABELS, WOW_THEMES),
                                    use_container_width=True, key="b1_place")
                    st.markdown("---")
                    st.markdown("#### Trend Lines: Ways of Working (Place) vs Employee Experience")
                    sel_out = st.selectbox("Select outcome", OUTCOME_LABELS, key="b1_place_out")
                    out_col = OUTCOME_COLS[OUTCOME_LABELS.index(sel_out)]
                    st.plotly_chart(make_trend_chart(dir_df, WOW_PLACE_COLS, WOW_THEMES, out_col, sel_out),
                                    use_container_width=True, key=f"b1_place_trend_{sel_out}")
                with b1_ind:
                    st.markdown("#### Correlational Heatmap: Ways of Working (Individual) × Employee Experience")
                    mat = spearman_matrix(dir_df, WOW_IND_COLS, OUTCOME_COLS)
                    mat.index   = WOW_THEMES
                    mat.columns = OUTCOME_LABELS
                    render_heatmap_cards(n_dir, mat)
                    render_wow_variance_card(mat)
                    st.plotly_chart(make_heatmap(mat.T, OUTCOME_LABELS, WOW_THEMES),
                                    use_container_width=True, key="b1_ind")
                    st.markdown("---")
                    st.markdown("#### Trend Lines: Ways of Working (Individual) vs Employee Experience")
                    sel_out = st.selectbox("Select outcome", OUTCOME_LABELS, key="b1_ind_out")
                    out_col = OUTCOME_COLS[OUTCOME_LABELS.index(sel_out)]
                    st.plotly_chart(make_trend_chart(dir_df, WOW_IND_COLS, WOW_THEMES, out_col, sel_out),
                                    use_container_width=True, key=f"b1_ind_trend_{sel_out}")

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
                b4_chart, b4_table = st.tabs(["Bar Chart", "Table"])
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
                with b4_table:
                    tdf, styler = build_wow_table(dir_df, "service_area", service_areas)
                    wow_table_cards(n_dir, tdf)
                    st.markdown("**I** = Individual average &nbsp;|&nbsp; "
                                "**P** = Place average &nbsp;|&nbsp; "
                                "**Δ** = I − P")
                    st.dataframe(styler, use_container_width=True, height=720)

        # ── B5 ────────────────────────────────────────────────
        with b5:
            if not service_areas:
                st.info("No service area breakdown available for this directorate.")
            else:
                st.markdown(f"#### Employee Experience — Average Scores by Service Area ({selected_dir})")
                b5_chart, b5_table = st.tabs(["Bar Chart", "Table"])
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
                with b5_table:
                    tdf, styler = build_outcome_table(dir_df, "service_area", service_areas)
                    outcome_table_cards(n_dir, tdf)
                    st.dataframe(styler, use_container_width=True, height=580)
