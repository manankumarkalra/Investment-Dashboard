import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Portfolio Terminal | 15-Asset Live Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. DESIGN TOKENS
# ============================================================
BG = "#11151C"
SURFACE = "#1B212C"
SURFACE_ALT = "#212938"
BORDER = "#2E3642"
TEXT = "#EDEFF2"
TEXT_MUTED = "#8F99A8"
GOLD = "#C9A227"
GOLD_SOFT = "#E4C567"
TEAL = "#3E8E8A"
ROSE = "#B5544A"
POSITIVE = "#5FA777"
NEGATIVE = "#C15C4F"

SECTOR_PALETTE = ["#C9A227", "#3E8E8A", "#7C8FC9", "#B5544A", "#8CA35E",
                  "#A6784A", "#6B7AA1", "#C97757", "#5FA07E", "#9A8FC2"]

# ============================================================
# 3. GLOBAL STYLE INJECTION
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

.stApp {{
    background: {BG};
}}

section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}

/* Hero header */
.hero-wrap {{
    padding: 28px 32px 24px 32px;
    background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-bottom: 22px;
}}
.hero-eyebrow {{
    font-size: 12.5px;
    letter-spacing: 0.02em;
    color: {GOLD_SOFT};
    margin-bottom: 6px;
    font-weight: 500;
}}
.hero-title {{
    font-family: 'Fraunces', serif;
    font-size: 34px;
    font-weight: 500;
    color: {TEXT};
    margin: 0 0 6px 0;
    line-height: 1.15;
}}
.hero-sub {{
    font-size: 14.5px;
    color: {TEXT_MUTED};
    margin: 0;
}}
.hero-big-number {{
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 500;
    color: {GOLD_SOFT};
    line-height: 1;
}}
.hero-big-label {{
    font-size: 12.5px;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}

/* KPI cards */
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 16px 18px;
    height: 100%;
}}
.kpi-label {{
    font-size: 12px;
    color: {TEXT_MUTED};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Fraunces', serif;
    font-size: 24px;
    font-weight: 500;
    color: {TEXT};
}}
.kpi-delta-pos {{ color: {POSITIVE}; font-size: 12.5px; margin-top: 4px; }}
.kpi-delta-neg {{ color: {NEGATIVE}; font-size: 12.5px; margin-top: 4px; }}

/* Section titles */
.section-title {{
    font-family: 'Fraunces', serif;
    font-size: 19px;
    font-weight: 500;
    color: {TEXT};
    margin: 4px 0 2px 0;
}}
.section-note {{
    font-size: 12.5px;
    color: {TEXT_MUTED};
    margin-bottom: 14px;
}}

/* Persona strip */
.persona-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 14px 16px;
    text-align: left;
}}
.persona-card.active {{
    border: 1px solid {GOLD};
    background: {SURFACE_ALT};
}}
.persona-name {{
    font-size: 12.5px;
    color: {TEXT_MUTED};
    letter-spacing: 0.01em;
}}
.persona-target {{
    font-family: 'Fraunces', serif;
    font-size: 21px;
    color: {TEXT};
    margin: 2px 0;
}}
.persona-cagr {{
    font-size: 12.5px;
    color: {GOLD_SOFT};
}}

hr {{ border-color: {BORDER}; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT_MUTED};
    font-size: 14px;
    padding: 8px 4px;
}}
.stTabs [aria-selected="true"] {{
    color: {GOLD_SOFT} !important;
    border-bottom: 2px solid {GOLD} !important;
}}

.stDataFrame {{ font-size: 13px; }}

footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# Plotly template matching the dashboard's dark, low-saturation palette
pio.templates["terminal_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12.5),
        colorway=SECTOR_PALETTE,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "terminal_dark"

# ============================================================
# 4. MASTER PORTFOLIO CONFIGURATION
# ============================================================
START_DATE = datetime(2026, 8, 31)
CORPUS = 1_00_00_000         # ₹1 Crore Base Capital
HORIZON_YEARS = 3
DEFAULT_RF = 0.069           # 6.90% Risk-Free Rate
DEFAULT_RM = 0.130           # 13.0% Market Return Expectation

PERSONAS = {
    "Aggressive": {"target_corpus": 2_25_00_000, "bond_floor": 0.05, "bond_cap": 0.15},
    "Moderate": {"target_corpus": 2_00_00_000, "bond_floor": 0.15, "bond_cap": 0.35},
    "Conservative": {"target_corpus": 1_75_00_000, "bond_floor": 0.30, "bond_cap": 0.55},
}
for p_name, p_data in PERSONAS.items():
    p_data["target_cagr"] = (p_data["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

INSTRUMENTS_MASTER = [
    {"Ticker": "INDIANB.NS", "Name": "Indian Bank", "Sector": "Banking", "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "HAL.NS", "Name": "Hindustan Aeronautics (HAL)", "Sector": "Defence", "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "CHENNPETRO.NS", "Name": "Chennai Petroleum Corp", "Sector": "Oil & Gas", "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "MAZDOCK.NS", "Name": "Mazagon Dock Shipbuilders", "Sector": "Defence", "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "BSE.NS", "Name": "BSE Ltd.", "Sector": "Financial Services", "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "NATIONALUM.NS", "Name": "National Aluminium Co (NALCO)", "Sector": "Metals", "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "FORCEMOT.NS", "Name": "Force Motors", "Sector": "Automobiles", "CapCategory": "Small/Mid Cap", "Type": "Equity"},
    {"Ticker": "LLOYDSME.NS", "Name": "Lloyds Metals & Energy", "Sector": "Metals", "CapCategory": "Mid/Small Cap", "Type": "Equity"},
    {"Ticker": "APARINDS.NS", "Name": "Apar Industries", "Sector": "Electrical / Industrial", "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "OIL.NS", "Name": "Oil India", "Sector": "Oil & Gas", "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "TVSMOTOR.NS", "Name": "TVS Motor Company", "Sector": "Automobiles", "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "GSEC2029", "Name": "6.03% GOI G-Sec 2029", "Sector": "Sovereign Debt", "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0625},
    {"Ticker": "EBBETF0431.NS", "Name": "BHARAT Bond ETF April 2031", "Sector": "Target Maturity Debt", "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0710},
    {"Ticker": "GOLDBEES.NS", "Name": "Nippon India ETF Gold BeES", "Sector": "Precious Metals", "CapCategory": "Gold ETF", "Type": "Gold ETF"},
    {"Ticker": "SILVERBEES.NS", "Name": "Nippon India Silver ETF", "Sector": "Precious Metals", "CapCategory": "Silver ETF", "Type": "Silver ETF"},
]

# ============================================================
# 5. LIVE MARKET DATA FETCHING & RETURN COMPUTATION ENGINE
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_market_data(lookback_yrs=3):
    tickers_to_fetch = [item["Ticker"] for item in INSTRUMENTS_MASTER if not item["Ticker"].startswith("GSEC")]
    tickers_to_fetch.append("^NSEI")
    raw = yf.download(tickers_to_fetch, period=f"{lookback_yrs}y", interval="1wk", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Failed to download market data from Yahoo Finance.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.dropna(how="all")


def analyze_live_performance(prices_df, rf_rate, mkt_return, lookback_yrs):
    """Returns (df_master, weekly_returns_df) so returns can be reused for correlation/heatmap charts."""
    returns = prices_df.pct_change().dropna(how="all")
    nifty_ret = returns["^NSEI"].dropna() if "^NSEI" in returns.columns else pd.Series(dtype=float)

    results = []
    for item in INSTRUMENTS_MASTER:
        t = item["Ticker"]
        record = item.copy()

        if t == "GSEC2029":
            record["LivePrice"] = 100.0
            record["Actual_CAGR"] = item["FixedYield"]
            record["Beta"] = 0.0
            record["DivYield"] = 0.0
            record["CAPM_Return"] = item["FixedYield"]
            record["Total_Expected_Return"] = item["FixedYield"]
        elif t in prices_df.columns:
            px_series = prices_df[t].dropna()
            if len(px_series) >= 2:
                latest_px = float(px_series.iloc[-1])
                first_px = float(px_series.iloc[0])
                actual_cagr = ((latest_px / first_px) ** (1.0 / lookback_yrs)) - 1.0
            else:
                latest_px, actual_cagr = 100.0, 0.08

            record["LivePrice"] = latest_px
            record["Actual_CAGR"] = actual_cagr

            if t in returns.columns and not nifty_ret.empty:
                combined = pd.concat([returns[t], nifty_ret], axis=1, join="inner").dropna()
                if len(combined) > 20:
                    cov = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])
                    beta = cov[0, 1] / np.var(combined.iloc[:, 1]) if np.var(combined.iloc[:, 1]) > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            record["Beta"] = beta

            try:
                info = yf.Ticker(t).info
                div_yld = float(info.get("dividendYield", 0.0) or 0.0)
                if div_yld > 1.0:
                    div_yld /= 100.0
            except Exception:
                div_yld = 0.015 if item["Type"] == "Equity" else 0.0
            record["DivYield"] = div_yld

            if item["Type"] == "Equity":
                capm_ret = rf_rate + beta * (mkt_return - rf_rate)
                record["CAPM_Return"] = capm_ret
                record["Total_Expected_Return"] = capm_ret + div_yld
            elif item["Type"] == "Bond":
                record["CAPM_Return"] = item.get("FixedYield", 0.071)
                record["Total_Expected_Return"] = item.get("FixedYield", 0.071)
            else:
                record["CAPM_Return"] = actual_cagr
                record["Total_Expected_Return"] = actual_cagr

        results.append(record)

    return pd.DataFrame(results), returns


# ============================================================
# 6. SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### ◆ Portfolio Terminal")
    st.caption("15-instrument live allocation model")
    st.divider()

    with st.expander("Investor Profile", expanded=True):
        selected_persona = st.selectbox("Persona target", list(PERSONAS.keys()), index=1)
        lookback_yrs = st.slider("Historical lookback (years)", min_value=1, max_value=5, value=3)

    with st.expander("CAPM Parameters", expanded=True):
        rf_rate = st.number_input("Risk-free rate (Rf)", value=DEFAULT_RF, step=0.001, format="%.3f")
        mkt_return = st.number_input("Market return (Rm)", value=DEFAULT_RM, step=0.005, format="%.3f")

    st.divider()
    if st.button("↻ Refresh live market data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Cache refreshes automatically every 15 minutes. Base date: 31 Aug 2026.")

# ============================================================
# 7. EXECUTE ENGINE & WEIGHT CALCULATIONS  (logic unchanged)
# ============================================================
try:
    with st.spinner("Downloading live prices & computing actual returns..."):
        prices_df = fetch_live_market_data(lookback_yrs)
        df_master, returns_df = analyze_live_performance(prices_df, rf_rate, mkt_return, lookback_yrs)
except Exception as e:
    st.error("Error executing market analysis engine.")
    st.exception(e)
    st.stop()

eq_mask = df_master["Type"] == "Equity"
clamped_beta = df_master.loc[eq_mask, "Beta"].clip(lower=0.2)
inv_b = 1.0 / clamped_beta
eq_sleeve_weights = inv_b / inv_b.sum()
df_master.loc[eq_mask, "EquitySleeveWeight"] = eq_sleeve_weights.values

eq_expected_return = (df_master.loc[eq_mask, "EquitySleeveWeight"] * df_master.loc[eq_mask, "Total_Expected_Return"]).sum()
eq_realized_cagr = (df_master.loc[eq_mask, "EquitySleeveWeight"] * df_master.loc[eq_mask, "Actual_CAGR"]).sum()

bond_mask = df_master["Type"] == "Bond"
bond_return = df_master.loc[bond_mask, "Total_Expected_Return"].mean()
bond_realized = df_master.loc[bond_mask, "Actual_CAGR"].mean()

gold_row = df_master[df_master["Type"] == "Gold ETF"].iloc[0]
silver_row = df_master[df_master["Type"] == "Silver ETF"].iloc[0]

gold_w, silver_w = 0.04, 0.04
commodities_w = gold_w + silver_w
investable_w = 1.0 - commodities_w

persona_cfg = PERSONAS[selected_persona]
target_cagr = persona_cfg["target_cagr"]

denom = eq_expected_return - bond_return
if abs(denom) > 1e-6:
    eq_w = (target_cagr - (gold_w * gold_row["Actual_CAGR"] + silver_w * silver_row["Actual_CAGR"]) - investable_w * bond_return) / denom
else:
    eq_w = investable_w

min_eq = investable_w - persona_cfg["bond_cap"]
max_eq = investable_w - persona_cfg["bond_floor"]
eq_w = float(np.clip(eq_w, min_eq, max_eq))
bond_w = investable_w - eq_w

weights = []
for idx, row in df_master.iterrows():
    if row["Type"] == "Equity":
        w = row["EquitySleeveWeight"] * eq_w
    elif row["Type"] == "Bond":
        w = bond_w / 2.0
    elif row["Type"] == "Gold ETF":
        w = gold_w
    else:
        w = silver_w
    weights.append(w)

df_master["Portfolio_Weight"] = weights
df_master["Allocated_Amount"] = df_master["Portfolio_Weight"] * CORPUS

actual_realized_portfolio_cagr = (df_master["Portfolio_Weight"] * df_master["Actual_CAGR"]).sum()
actual_capm_expected_return = (df_master["Portfolio_Weight"] * df_master["Total_Expected_Return"]).sum()
projected_corpus_realized = CORPUS * ((1 + actual_realized_portfolio_cagr) ** HORIZON_YEARS)


# ============================================================
# 8. HERO HEADER
# ============================================================
gap_vs_target = actual_realized_portfolio_cagr - target_cagr
gap_class = "kpi-delta-pos" if gap_vs_target >= 0 else "kpi-delta-neg"
gap_word = "ahead of" if gap_vs_target >= 0 else "behind"

hero_l, hero_r = st.columns([2.3, 1])
with hero_l:
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">LIVE PORTFOLIO · BASE DATE 31 AUG 2026 · {lookback_yrs}Y LOOKBACK</div>
        <div class="hero-title">15-Asset Allocation Terminal</div>
        <p class="hero-sub">₹1.00 Cr deployed across 11 equities, 2 sovereign/quasi-sovereign bonds and gold &amp; silver ETFs,
        sized against a <strong style="color:{GOLD_SOFT}">{selected_persona}</strong> 3-year target of ₹{persona_cfg['target_corpus']/1e7:.2f} Cr.</p>
    </div>
    """, unsafe_allow_html=True)
with hero_r:
    st.markdown(f"""
    <div class="hero-wrap" style="text-align:right; height:100%;">
        <div class="hero-big-number">₹{projected_corpus_realized/1e7:.2f} Cr</div>
        <div class="hero-big-label">Projected value in {HORIZON_YEARS}Y at realized CAGR</div>
        <div class="{gap_class}" style="margin-top:10px; font-size:13px;">
            {abs(gap_vs_target):.2%} {gap_word} the {target_cagr:.2%} required rate
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 9. KPI ROW
# ============================================================
def kpi_card(label, value, delta_html=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(kpi_card("Starting Capital", f"₹{CORPUS/1e7:.2f} Cr"), unsafe_allow_html=True)
k2.markdown(kpi_card("Required Persona CAGR", f"{target_cagr:.2%}"), unsafe_allow_html=True)
k3.markdown(kpi_card(
    "Actual Realized CAGR", f"{actual_realized_portfolio_cagr:.2%}",
    f'<div class="{gap_class}">{gap_vs_target:+.2%} vs target</div>'
), unsafe_allow_html=True)
k4.markdown(kpi_card("CAPM Expected Yield", f"{actual_capm_expected_return:.2%}"), unsafe_allow_html=True)
k5.markdown(kpi_card("Equity : Bond : Commodity", f"{eq_w:.0%} · {bond_w:.0%} · {commodities_w:.0%}"), unsafe_allow_html=True)

st.write("")

# Persona comparison strip
p1, p2, p3 = st.columns(3)
for col, (pname, pdata) in zip([p1, p2, p3], PERSONAS.items()):
    active = " active" if pname == selected_persona else ""
    col.markdown(f"""
    <div class="persona-card{active}">
        <div class="persona-name">{pname.upper()}</div>
        <div class="persona-target">₹{pdata['target_corpus']/1e7:.2f} Cr</div>
        <div class="persona-cagr">requires {pdata['target_cagr']:.2%} CAGR · bonds {pdata['bond_floor']:.0%}–{pdata['bond_cap']:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
# 10. MAIN TABS
# ============================================================
tab_overview, tab_holdings, tab_risk, tab_history = st.tabs(
    ["Overview", "Holdings", "Risk & Return", "Historical Trends"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown('<div class="section-title">Sleeve Allocation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Capital split across the four asset sleeves</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame([
            {"Sleeve": "Equities (11)", "Amount": eq_w * CORPUS},
            {"Sleeve": "Bonds (2)", "Amount": bond_w * CORPUS},
            {"Sleeve": "Gold ETF", "Amount": gold_w * CORPUS},
            {"Sleeve": "Silver ETF", "Amount": silver_w * CORPUS},
        ])
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_df["Sleeve"], values=pie_df["Amount"], hole=0.58,
            marker=dict(colors=[GOLD, TEAL, "#A6784A", "#8F99A8"], line=dict(color=SURFACE, width=2)),
            textinfo="label+percent", textfont=dict(size=12.5),
        )])
        fig_pie.update_layout(height=340, showlegend=False,
                               annotations=[dict(text=f"₹{CORPUS/1e7:.2f} Cr", x=0.5, y=0.5,
                                                  font=dict(size=16, family="Fraunces, serif", color=TEXT),
                                                  showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Realized vs. Target Trajectory</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Compounding at realized, CAPM-expected, and persona-required rates</div>', unsafe_allow_html=True)
        yrs = np.arange(0, HORIZON_YEARS + 1)
        realized_path = CORPUS * ((1 + actual_realized_portfolio_cagr) ** yrs)
        capm_path = CORPUS * ((1 + actual_capm_expected_return) ** yrs)
        target_path = CORPUS * ((1 + target_cagr) ** yrs)

        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(x=yrs, y=realized_path, mode="lines+markers",
                                         name=f"Realized ({actual_realized_portfolio_cagr:.2%})",
                                         line=dict(color=GOLD, width=3)))
        fig_growth.add_trace(go.Scatter(x=yrs, y=capm_path, mode="lines+markers",
                                         name=f"CAPM Expected ({actual_capm_expected_return:.2%})",
                                         line=dict(color=TEAL, width=2)))
        fig_growth.add_trace(go.Scatter(x=yrs, y=target_path, mode="lines",
                                         name=f"Target Required ({target_cagr:.2%})",
                                         line=dict(color=TEXT_MUTED, dash="dash")))
        fig_growth.update_layout(
            height=340, xaxis_title="Years elapsed", yaxis_title="Portfolio value (₹)",
            yaxis_tickprefix="₹", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown('<div class="section-title">Capital by Sector</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Sleeve → sector → instrument, sized by capital allocated</div>', unsafe_allow_html=True)
    fig_tree = px.treemap(
        df_master, path=[px.Constant("Portfolio"), "Type", "Sector", "Name"],
        values="Allocated_Amount", color="Type",
        color_discrete_map={"Equity": GOLD, "Bond": TEAL, "Gold ETF": "#A6784A", "Silver ETF": "#8F99A8"},
    )
    fig_tree.update_traces(textfont=dict(size=12), marker=dict(line=dict(color=SURFACE, width=1)))
    fig_tree.update_layout(height=420, margin=dict(l=4, r=4, t=10, b=4))
    st.plotly_chart(fig_tree, use_container_width=True)

# ---------- TAB 2: HOLDINGS ----------
with tab_holdings:
    st.markdown('<div class="section-title">All 15 Instruments</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Live price, realized performance, CAPM yield, and capital allocated</div>', unsafe_allow_html=True)

    display_df = df_master.copy()
    display_df["Live Price"] = display_df["LivePrice"]
    display_df["Actual CAGR"] = display_df["Actual_CAGR"]
    display_df["Dividend Yield"] = display_df["DivYield"]
    display_df["Total Expected Yield"] = display_df["Total_Expected_Return"]

    st.dataframe(
        display_df[[
            "Ticker", "Name", "Type", "Sector", "CapCategory", "Live Price",
            "Actual CAGR", "Beta", "Dividend Yield", "Total Expected Yield",
            "Portfolio_Weight", "Allocated_Amount",
        ]],
        use_container_width=True, hide_index=True,
        column_config={
            "Live Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Actual CAGR": st.column_config.NumberColumn(format="%.2f%%"),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
            "Dividend Yield": st.column_config.NumberColumn(format="%.2f%%"),
            "Total Expected Yield": st.column_config.NumberColumn(format="%.2f%%"),
            "Portfolio_Weight": st.column_config.ProgressColumn(
                "Portfolio Weight", format="%.1f%%", min_value=0, max_value=float(display_df["Portfolio_Weight"].max()) * 1.15,
            ),
            "Allocated_Amount": st.column_config.NumberColumn("Capital Allocated", format="₹%d"),
        },
    )

    csv = df_master.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download holdings as CSV", csv, "portfolio_holdings.csv", "text/csv")

    st.write("")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown('<div class="section-title">Weight by Instrument</div>', unsafe_allow_html=True)
        sorted_df = df_master.sort_values("Portfolio_Weight", ascending=True)
        fig_bar = px.bar(
            sorted_df, x="Portfolio_Weight", y="Name", orientation="h", color="Type",
            color_discrete_map={"Equity": GOLD, "Bond": TEAL, "Gold ETF": "#A6784A", "Silver ETF": "#8F99A8"},
        )
        fig_bar.update_layout(height=460, xaxis_tickformat=".0%", yaxis_title="", xaxis_title="Portfolio weight",
                               legend_title="")
        st.plotly_chart(fig_bar, use_container_width=True)

    with h2:
        st.markdown('<div class="section-title">Capital by Sector</div>', unsafe_allow_html=True)
        sector_df = df_master.groupby("Sector", as_index=False)["Allocated_Amount"].sum().sort_values("Allocated_Amount")
        fig_sector = px.bar(sector_df, x="Allocated_Amount", y="Sector", orientation="h",
                             color="Allocated_Amount", color_continuous_scale=[SURFACE_ALT, GOLD])
        fig_sector.update_layout(height=460, xaxis_title="Capital allocated (₹)", yaxis_title="",
                                  coloraxis_showscale=False)
        st.plotly_chart(fig_sector, use_container_width=True)

# ---------- TAB 3: RISK & RETURN ----------
with tab_risk:
    st.markdown('<div class="section-title">Security Market Line — Equity Sleeve</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Where each stock\'s realized return sits relative to what its Beta implies under CAPM</div>', unsafe_allow_html=True)

    eq_df = df_master[df_master["Type"] == "Equity"].copy()
    beta_range = np.linspace(0, max(eq_df["Beta"].max() * 1.15, 1.5), 50)
    sml = rf_rate + beta_range * (mkt_return - rf_rate)

    fig_sml = go.Figure()
    fig_sml.add_trace(go.Scatter(x=beta_range, y=sml, mode="lines", name="Security Market Line",
                                  line=dict(color=TEXT_MUTED, dash="dash", width=1.5)))
    eq_df["Position"] = np.where(eq_df["Actual_CAGR"] >= eq_df["CAPM_Return"], "Outperforming", "Underperforming")
    fig_sml.add_trace(go.Scatter(
        x=eq_df["Beta"], y=eq_df["Actual_CAGR"], mode="markers+text", text=eq_df["Ticker"],
        textposition="top center", textfont=dict(size=10.5),
        marker=dict(
            size=eq_df["Portfolio_Weight"] * 420 + 10,
            color=np.where(eq_df["Position"] == "Outperforming", POSITIVE, NEGATIVE),
            line=dict(color=SURFACE, width=1),
        ),
        name="Realized CAGR", hovertext=eq_df["Name"],
    ))
    fig_sml.update_layout(height=440, xaxis_title="Beta (systematic risk vs. Nifty 50)",
                           yaxis_title="Return", yaxis_tickformat=".0%", showlegend=True)
    st.plotly_chart(fig_sml, use_container_width=True)
    st.caption("Bubble size = portfolio weight. Green = realized return beat its CAPM-implied return; rust = it fell short.")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="section-title">Beta by Stock</div>', unsafe_allow_html=True)
        beta_sorted = eq_df.sort_values("Beta")
        fig_beta = px.bar(beta_sorted, x="Beta", y="Ticker", orientation="h",
                           color="Beta", color_continuous_scale=[TEAL, SURFACE_ALT, ROSE])
        fig_beta.add_vline(x=1.0, line_dash="dot", line_color=TEXT_MUTED)
        fig_beta.update_layout(height=420, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_beta, use_container_width=True)

    with r2:
        st.markdown('<div class="section-title">Equity Correlation Matrix</div>', unsafe_allow_html=True)
        eq_tickers = [t for t in eq_df["Ticker"] if t in returns_df.columns]
        corr = returns_df[eq_tickers].corr()
        fig_corr = px.imshow(corr, color_continuous_scale=[TEAL, SURFACE_ALT, GOLD], zmin=-1, zmax=1,
                              aspect="auto")
        fig_corr.update_layout(height=420, coloraxis_colorbar=dict(title=""))
        st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Lower average correlation across holdings means more effective diversification within the equity sleeve.")

# ---------- TAB 4: HISTORICAL TRENDS ----------
with tab_history:
    st.markdown('<div class="section-title">Normalized Price Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">All series indexed to 100 at the start of the lookback window, vs. Nifty 50</div>', unsafe_allow_html=True)

    all_names = {row["Ticker"]: row["Name"] for _, row in df_master.iterrows() if row["Ticker"] in prices_df.columns}
    default_pick = [t for t in df_master[df_master["Type"] == "Equity"]["Ticker"].head(5) if t in prices_df.columns]
    picked = st.multiselect(
        "Instruments to plot", options=list(all_names.keys()) + (["^NSEI"] if "^NSEI" in prices_df.columns else []),
        default=default_pick + (["^NSEI"] if "^NSEI" in prices_df.columns else []),
        format_func=lambda t: "Nifty 50" if t == "^NSEI" else all_names.get(t, t),
    )

    if picked:
        norm_df = prices_df[picked].dropna(how="all")
        norm_df = norm_df / norm_df.iloc[0] * 100
        fig_hist = go.Figure()
        for i, col in enumerate(picked):
            label = "Nifty 50" if col == "^NSEI" else all_names.get(col, col)
            style = dict(color=TEXT_MUTED, dash="dot", width=2) if col == "^NSEI" else dict(width=2, color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)])
            fig_hist.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode="lines", name=label, line=style))
        fig_hist.update_layout(height=460, yaxis_title="Indexed value (start = 100)", xaxis_title="")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Select at least one instrument to plot.")

    st.markdown('<div class="section-title">Fixed Income &amp; Commodities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Gold and Silver ETF price history over the lookback window</div>', unsafe_allow_html=True)
    fi_tickers = [t for t in ["GOLDBEES.NS", "SILVERBEES.NS"] if t in prices_df.columns]
    if fi_tickers:
        fig_fi = go.Figure()
        colors_fi = {"GOLDBEES.NS": GOLD, "SILVERBEES.NS": "#B9BEC7"}
        for t in fi_tickers:
            series = prices_df[t].dropna()
            fig_fi.add_trace(go.Scatter(x=series.index, y=series, mode="lines",
                                         name=all_names.get(t, t), line=dict(color=colors_fi.get(t, TEAL), width=2)))
        fig_fi.update_layout(height=340, yaxis_title="Price (₹)", xaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

st.write("")
st.caption(
    "For educational/simulation purposes only. Beta and CAPM-expected returns are estimated from "
    f"{lookback_yrs}Y of weekly data against the Nifty 50; realized CAGR reflects live market prices at the time this page loaded."
)
