"""
FlowLLM Dashboard — Dark-Themed Modern UI
==========================================
A visually impressive single-page Streamlit dashboard comparing
baseline (fixed-cycle) vs LLM-controlled traffic light performance.

Run:  streamlit run Dashboard/app.py
"""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="FlowLLM Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===================================================================
# COLOUR PALETTE
# ===================================================================
BG_DARK   = "#0e1117"
BG_CARD   = "#1a1d24"
BG_CARD2  = "#22262e"
GREEN     = "#00ff88"
GREEN_DIM = "#00cc6a"
GREY      = "#888888"
GREY_LIGHT = "#aaaaaa"
TEXT_WHITE = "#f0f0f0"
TEXT_MUTED = "#8892a0"

# ===================================================================
# CUSTOM CSS — injected into the page
# ===================================================================
st.markdown(f"""
<style>
    /* ---------- Global ---------- */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_WHITE};
    }}
    section[data-testid="stSidebar"] {{
        display: none;
    }}
    /* Hide Streamlit header/footer chrome */
    header[data-testid="stHeader"] {{
        background-color: {BG_DARK};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ---------- Hero ---------- */
    .hero {{
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }}
    .hero h1 {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 3.6rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: {TEXT_WHITE};
        margin-bottom: 0;
    }}
    .hero h1 span {{
        color: {GREEN};
    }}
    .hero .subtitle {{
        font-size: 1.15rem;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
    }}
    .hero .callout {{
        display: inline-block;
        margin-top: 1.2rem;
        padding: 0.6rem 1.6rem;
        border-radius: 40px;
        background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
        border: 1px solid #2e3440;
        font-size: 1.05rem;
        color: {GREEN};
        font-weight: 600;
    }}

    /* ---------- Metric Cards ---------- */
    .metric-row {{
        display: flex;
        gap: 1.2rem;
        margin: 2rem 0;
    }}
    .metric-card {{
        flex: 1;
        background: {BG_CARD};
        border-radius: 16px;
        padding: 1.8rem 1.4rem;
        border: 1px solid #2e3440;
        text-align: center;
    }}
    .metric-card .label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {TEXT_MUTED};
        margin-bottom: 0.8rem;
    }}
    .metric-card .values {{
        display: flex;
        justify-content: center;
        align-items: baseline;
        gap: 1.2rem;
    }}
    .metric-card .val-baseline {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {GREY_LIGHT};
    }}
    .metric-card .arrow {{
        font-size: 1.3rem;
        color: {TEXT_MUTED};
    }}
    .metric-card .val-llm {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {GREEN};
    }}
    .metric-card .improvement {{
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        background: rgba(0, 255, 136, 0.12);
        color: {GREEN};
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .metric-card .improvement.negative {{
        background: rgba(255, 80, 80, 0.12);
        color: #ff5050;
    }}

    /* ---------- Section Headers ---------- */
    .section-header {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: {TEXT_WHITE};
        margin: 2.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2e3440;
    }}

    /* ---------- How-It-Works Cards ---------- */
    .hiw-row {{
        display: flex;
        gap: 1.2rem;
        margin: 1rem 0 2rem 0;
    }}
    .hiw-card {{
        flex: 1;
        background: {BG_CARD};
        border-radius: 14px;
        padding: 1.6rem 1.3rem;
        border: 1px solid #2e3440;
    }}
    .hiw-card .icon {{
        font-size: 2rem;
        margin-bottom: 0.6rem;
    }}
    .hiw-card .title {{
        font-weight: 700;
        font-size: 1.05rem;
        color: {TEXT_WHITE};
        margin-bottom: 0.4rem;
    }}
    .hiw-card .desc {{
        font-size: 0.88rem;
        color: {TEXT_MUTED};
        line-height: 1.45;
    }}

    /* ---------- Stats Row ---------- */
    .stats-row {{
        display: flex;
        gap: 1.2rem;
        margin: 1rem 0;
    }}
    .stat-card {{
        flex: 1;
        background: {BG_CARD};
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid #2e3440;
        text-align: center;
    }}
    .stat-card .stat-num {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {GREEN};
    }}
    .stat-card .stat-label {{
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
    }}

    /* ---------- Footer ---------- */
    .footer {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #2e3440;
    }}
</style>
""", unsafe_allow_html=True)

# ===================================================================
# DATA LOADING
# ===================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASELINE_CSV = os.path.join(PROJECT_ROOT, "Data", "baseline_metrics.csv")
LLM_CSV      = os.path.join(PROJECT_ROOT, "Data", "llm_metrics.csv")


@st.cache_data
def load_data():
    baseline = pd.read_csv(BASELINE_CSV)
    llm      = pd.read_csv(LLM_CSV)
    return baseline, llm


if not os.path.exists(BASELINE_CSV) or not os.path.exists(LLM_CSV):
    st.error("Missing metrics CSVs. Run both controllers first.")
    st.stop()

df_base, df_llm = load_data()

# ===================================================================
# COMPUTE SUMMARY STATS
# ===================================================================
base_avg_wait  = df_base["avg_waiting_time"].mean()
llm_avg_wait   = df_llm["avg_waiting_time"].mean()
wait_pct       = ((base_avg_wait - llm_avg_wait) / base_avg_wait * 100)

base_avg_queue = df_base["total_queue_length"].mean()
llm_avg_queue  = df_llm["total_queue_length"].mean()
queue_pct      = ((base_avg_queue - llm_avg_queue) / base_avg_queue * 100)

base_max_queue = int(df_base["max_queue_length"].max())
llm_max_queue  = int(df_llm["max_queue_length"].max())

# Vehicle throughput (count of rows with vehicle_count > 0 isn't right —
# use max vehicle_count as a proxy for peak vehicles)
base_peak = int(df_base["vehicle_count"].max())
llm_peak  = int(df_llm["vehicle_count"].max())
peak_pct  = ((llm_peak - base_peak) / base_peak * 100) if base_peak else 0

# LLM decisions
llm_actions  = df_llm[df_llm["llm_action"].notna() & (df_llm["llm_action"] != "")]
total_queries = len(llm_actions)
switch_count  = int((llm_actions["llm_action"] == "switch").sum())
keep_count    = int((llm_actions["llm_action"] == "keep").sum())


# ===================================================================
# HELPER: format improvement badge
# ===================================================================
def pct_badge(pct: float, invert: bool = False) -> str:
    """Return an HTML badge. If invert=True, positive % is bad (red)."""
    is_good = pct > 0 if not invert else pct < 0
    display = abs(pct)
    arrow = "+" if pct >= 0 else "-"  # use ASCII arrows only
    cls = "" if is_good else " negative"
    return f'<span class="improvement{cls}">{arrow}{display:.1f}%</span>'


# ===================================================================
# HELPER: build a dark-themed Plotly line chart
# ===================================================================
def dark_line_chart(x_base, y_base, x_llm, y_llm,
                    y_title: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_base, y=y_base,
        name="Baseline",
        line=dict(color=GREY, width=2),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=x_llm, y=y_llm,
        name="LLM Controller",
        line=dict(color=GREEN, width=2.5),
    ))
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MUTED, family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(
            title="Simulation Time (s)",
            gridcolor="#2e3440",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="#2e3440",
            zeroline=False,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(color=TEXT_WHITE),
        ),
        margin=dict(l=50, r=20, t=30, b=50),
        height=height,
        hovermode="x unified",
    )
    return fig


# ===================================================================
# HERO SECTION
# ===================================================================
callout_text = (
    f"{abs(wait_pct):.0f}% {'reduction' if wait_pct > 0 else 'increase'} "
    f"in average wait time using a local LLM"
)

st.markdown(f"""
<div class="hero">
    <h1>Flow<span>LLM</span></h1>
    <div class="subtitle">AI-Powered Traffic Light Optimization</div>
    <div class="callout">{callout_text}</div>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# METRIC CARDS
# ===================================================================
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="label">Avg Wait Time</div>
        <div class="values">
            <span class="val-baseline">{base_avg_wait:.1f}s</span>
            <span class="arrow">&rarr;</span>
            <span class="val-llm">{llm_avg_wait:.1f}s</span>
        </div>
        {pct_badge(wait_pct)}
    </div>
    <div class="metric-card">
        <div class="label">Peak Vehicles</div>
        <div class="values">
            <span class="val-baseline">{base_peak}</span>
            <span class="arrow">&rarr;</span>
            <span class="val-llm">{llm_peak}</span>
        </div>
        {pct_badge(peak_pct, invert=True)}
    </div>
    <div class="metric-card">
        <div class="label">Avg Queue Length</div>
        <div class="values">
            <span class="val-baseline">{base_avg_queue:.1f}</span>
            <span class="arrow">&rarr;</span>
            <span class="val-llm">{llm_avg_queue:.1f}</span>
        </div>
        {pct_badge(queue_pct)}
    </div>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# CHART: Average Waiting Time
# ===================================================================
st.markdown('<div class="section-header">Average Waiting Time Over Simulation</div>',
            unsafe_allow_html=True)

fig_wait = dark_line_chart(
    df_base["sim_time"], df_base["avg_waiting_time"],
    df_llm["sim_time"],  df_llm["avg_waiting_time"],
    y_title="Avg Waiting Time (s)",
)
st.plotly_chart(fig_wait, use_container_width=True)


# ===================================================================
# CHART: Queue Length
# ===================================================================
st.markdown('<div class="section-header">Total Queue Length Over Simulation</div>',
            unsafe_allow_html=True)

fig_queue = dark_line_chart(
    df_base["sim_time"], df_base["total_queue_length"],
    df_llm["sim_time"],  df_llm["total_queue_length"],
    y_title="Halting Vehicles",
)
st.plotly_chart(fig_queue, use_container_width=True)


# ===================================================================
# HOW IT WORKS
# ===================================================================
st.markdown('<div class="section-header">How It Works</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="hiw-row">
    <div class="hiw-card">
        <div class="icon">🛣️</div>
        <div class="title">SUMO Simulation</div>
        <div class="desc">
            A 4-way intersection modelled in SUMO with realistic vehicle flows.
            The simulation runs for 600 seconds with ~700 vehicles spawned
            across all approaches.
        </div>
    </div>
    <div class="hiw-card">
        <div class="icon">🧠</div>
        <div class="title">LLM Decision Engine</div>
        <div class="desc">
            Every 10 seconds, the intersection state is sent to a local
            Llama 3.1 (8B) model via Ollama. The LLM decides whether to
            keep or switch the signal phase and for how long.
        </div>
    </div>
    <div class="hiw-card">
        <div class="icon">📊</div>
        <div class="title">Metrics Collection</div>
        <div class="desc">
            Queue lengths, waiting times, and vehicle counts are logged
            every simulation step. Results are compared against a
            fixed-cycle baseline controller.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# LLM DECISION BREAKDOWN
# ===================================================================
st.markdown('<div class="section-header">LLM Decision Breakdown</div>',
            unsafe_allow_html=True)

# Stats cards
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-num">{total_queries}</div>
        <div class="stat-label">Total Queries</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{switch_count}</div>
        <div class="stat-label">Switch Decisions</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">{keep_count}</div>
        <div class="stat-label">Keep Decisions</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Bar chart — Switch vs Keep
if total_queries > 0:
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Bar(
        x=["Switch", "Keep"],
        y=[switch_count, keep_count],
        marker_color=["#EF553B", "#AB63FA"],
        text=[switch_count, keep_count],
        textposition="outside",
        textfont=dict(color=TEXT_WHITE, size=14, family="Inter, Segoe UI, sans-serif"),
        width=0.45,
    ))
    fig_dec.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MUTED, family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor="#2e3440"),
        yaxis=dict(title="Count", gridcolor="#2e3440", zeroline=False),
        margin=dict(l=50, r=20, t=20, b=40),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_dec, use_container_width=True)


# ===================================================================
# FOOTER
# ===================================================================
st.markdown("""
<div class="footer">
    FlowLLM &mdash; Traffic Light Optimization with Local LLMs &bull;
    Built with SUMO &bull; Ollama &bull; Streamlit
</div>
""", unsafe_allow_html=True)
