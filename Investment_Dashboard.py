import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT SETUP & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="15-Asset Live Market Portfolio Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #0066cc;
        padding: 14px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 10px;
    }
    .stDataFrame { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. MASTER PORTFOLIO CONFIGURATION
# ============================================================
START_DATE = datetime(2026, 8, 31)
CORPUS = 1_00_00_000         # ₹1 Crore Base Capital
HORIZON_YEARS = 3
DEFAULT_RF = 0.069           # 6.90% Risk-Free Rate
DEFAULT_RM = 0.130           # 13.0% Market Return Expectation

# Required Persona Target Targets
PERSONAS = {
    "Aggressive": {"target_corpus": 2_25_00_000, "bond_floor": 0.05, "bond_cap": 0.15},
    "Moderate": {"target_corpus": 2_00_00_000, "bond_floor": 0.15, "bond_cap": 0.35},
    "Conservative": {"target_corpus": 1_75_00_000, "bond_floor": 0.30, "bond_cap": 0.55},
}

for p_name, p_data in PERSONAS.items():
    p_data["target_cagr"] = (p_data["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

# Master 15 Instruments Metadata
INSTRUMENTS_MASTER = [
    # 11 Equities
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
    
    # 2 Bonds
    {"Ticker": "GSEC2029", "Name": "6.03% GOI G-Sec 2029", "Sector": "Sovereign Debt", "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0625},
    {"Ticker": "EBBETF0431.NS", "Name": "BHARAT Bond ETF April 2031", "Sector": "Target Maturity Debt", "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0710},
    
    # 2 Commodities / Gold & Silver ETFs
    {"Ticker": "GOLDBEES.NS", "Name": "Nippon India ETF Gold BeES", "Sector": "Precious Metals", "CapCategory": "Gold ETF", "Type": "Gold ETF"},
    {"Ticker": "SILVERBEES.NS", "Name": "Nippon India Silver ETF", "Sector": "Precious Metals", "CapCategory": "Silver ETF", "Type": "Silver ETF"},
]

# ============================================================
# 3. LIVE MARKET DATA FETCHING & RETURN COMPUTATION ENGINE
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_market_data(lookback_yrs=3):
    tickers_to_fetch = [item["Ticker"] for item in INSTRUMENTS_MASTER if not item["Ticker"].startswith("GSEC")]
    tickers_to_fetch.append("^NSEI")  # Nifty 50 Benchmark
    
    raw = yf.download(tickers_to_fetch, period=f"{lookback_yrs}y", interval="1wk", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Failed to download market data from Yahoo Finance.")
        
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return prices.dropna(how="all")

def analyze_live_performance(prices_df, rf_rate, mkt_return, lookback_yrs):
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
            
            # Beta calculation against Nifty 50
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
            
            # Fetch Live Dividend Yield via yfinance
            try:
                info = yf.Ticker(t).info
                div_yld = float(info.get("dividendYield", 0.0) or 0.0)
                if div_yld > 1.0: div_yld /= 100.0
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
            else:  # Gold / Silver ETFs
                record["CAPM_Return"] = actual_cagr
                record["Total_Expected_Return"] = actual_cagr
                
        results.append(record)
        
    return pd.DataFrame(results)

# ============================================================
# 4. SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("⚡ Portfolio & Data Controls")
selected_persona = st.sidebar.selectbox("Investor Persona Target", list(PERSONAS.keys()), index=1)
lookback_yrs = st.sidebar.slider("Historical Lookback Window (Years)", min_value=1, max_value=5, value=3)

st.sidebar.divider()
st.sidebar.subheader("📈 CAPM Parameters")
rf_rate = st.sidebar.number_input("Risk-Free Rate (Rf)", value=DEFAULT_RF, step=0.001, format="%.3f")
mkt_return = st.sidebar.number_input("Market Return (Rm)", value=DEFAULT_RM, step=0.005, format="%.3f")

if st.sidebar.button("🔄 Fetch Fresh Market Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# 5. EXECUTE ENGINE & WEIGHT CALCULATIONS
# ============================================================
try:
    with st.spinner("Downloading live prices & computing actual returns..."):
        prices_df = fetch_live_market_data(lookback_yrs)
        df_master = analyze_live_performance(prices_df, rf_rate, mkt_return, lookback_yrs)
except Exception as e:
    st.error("Error executing market analysis engine.")
    st.exception(e)
    st.stop()

# Equity Inverse-Beta Weighting Scheme
eq_mask = df_master["Type"] == "Equity"
clamped_beta = df_master.loc[eq_mask, "Beta"].clip(lower=0.2)
inv_b = 1.0 / clamped_beta
eq_sleeve_weights = inv_b / inv_b.sum()
df_master.loc[eq_mask, "EquitySleeveWeight"] = eq_sleeve_weights.values

# Calculate Expected & Realized Returns for Equity Sleeve
eq_expected_return = (df_master.loc[eq_mask, "EquitySleeveWeight"] * df_master.loc[eq_mask, "Total_Expected_Return"]).sum()
eq_realized_cagr = (df_master.loc[eq_mask, "EquitySleeveWeight"] * df_master.loc[eq_mask, "Actual_CAGR"]).sum()

# Bond & Commodity Fixed Returns
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

# Macro Asset Allocation Solver
denom = eq_expected_return - bond_return
if abs(denom) > 1e-6:
    eq_w = (target_cagr - (gold_w * gold_row["Actual_CAGR"] + silver_w * silver_row["Actual_CAGR"]) - investable_w * bond_return) / denom
else:
    eq_w = investable_w

min_eq = investable_w - persona_cfg["bond_cap"]
max_eq = investable_w - persona_cfg["bond_floor"]
eq_w = float(np.clip(eq_w, min_eq, max_eq))
bond_w = investable_w - eq_w

# Final Weights Assignment Across All 15 Assets
weights = []
for idx, row in df_master.iterrows():
    if row["Type"] == "Equity":
        w = row["EquitySleeveWeight"] * eq_w
    elif row["Type"] == "Bond":
        w = bond_w / 2.0
    elif row["Type"] == "Gold ETF":
        w = gold_w
    else:  # Silver ETF
        w = silver_w
    weights.append(w)

df_master["Portfolio_Weight"] = weights
df_master["Allocated_Amount"] = df_master["Portfolio_Weight"] * CORPUS

# Aggregate Portfolio Returns
actual_realized_portfolio_cagr = (df_master["Portfolio_Weight"] * df_master["Actual_CAGR"]).sum()
actual_capm_expected_return = (df_master["Portfolio_Weight"] * df_master["Total_Expected_Return"]).sum()
projected_corpus_realized = CORPUS * ((1 + actual_realized_portfolio_cagr) ** HORIZON_YEARS)

# ============================================================
# 6. DASHBOARD DISPLAY & METRICS
# ============================================================
st.title("⚡ 15-Asset Live Market Portfolio Dashboard")
st.caption(f"Portfolio Base Date: **August 31, 2026** | Persona Target: **{selected_persona}** | Lookback Window: **{lookback_yrs} Years**")

# Top KPI Metric Cards
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Starting Capital", f"₹{CORPUS/1e7:.2f} Cr")
m2.metric("Required Persona CAGR", f"{target_cagr:.2%}")
m3.metric("Actual Realized CAGR", f"{actual_realized_portfolio_cagr:.2%}", delta=f"{(actual_realized_portfolio_cagr - target_cagr):.2%} vs Target")
m4.metric("CAPM Expected Yield", f"{actual_capm_expected_return:.2%}")
m5.metric("Projected 3Y Value", f"₹{projected_corpus_realized/1e7:.2f} Cr")

st.divider()

# Core Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Asset Sleeve Capital Allocation")
    pie_df = pd.DataFrame([
        {"Sleeve": "11 Equities Sleeve", "Amount": eq_w * CORPUS},
        {"Sleeve": "2 Bonds Sleeve", "Amount": bond_w * CORPUS},
        {"Sleeve": "Gold BeES ETF", "Amount": gold_w * CORPUS},
        {"Sleeve": "Silver ETF", "Amount": silver_w * CORPUS},
    ])
    fig_pie = px.pie(pie_df, names="Sleeve", values="Amount", hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Actual Realized vs Target Trajectory")
    yrs = np.arange(0, HORIZON_YEARS + 1)
    realized_path = CORPUS * ((1 + actual_realized_portfolio_cagr) ** yrs)
    capm_path = CORPUS * ((1 + actual_capm_expected_return) ** yrs)
    target_path = CORPUS * ((1 + target_cagr) ** yrs)
    
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=yrs, y=realized_path, mode="lines+markers", name=f"Actual Realized ({actual_realized_portfolio_cagr:.2%})", line=dict(color="#0066cc", width=3)))
    fig_growth.add_trace(go.Scatter(x=yrs, y=capm_path, mode="lines+markers", name=f"CAPM Expected ({actual_capm_expected_return:.2%})", line=dict(color="#2ca02c", width=2)))
    fig_growth.add_trace(go.Scatter(x=yrs, y=target_path, mode="lines", name=f"Target Required ({target_cagr:.2%})", line=dict(color="#ff7f0e", dash="dash")))
    
    fig_growth.update_layout(
        height=360,
        xaxis_title="Years Elapsed",
        yaxis_title="Portfolio Corpus (₹)",
        yaxis_tickprefix="₹",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_growth, use_container_width=True)

# Master Data Breakdown Table
st.subheader("Live Performance & Risk Metrics (All 15 Instruments)")

display_df = df_master.copy()
display_df["Live Price"] = display_df["LivePrice"].map(lambda x: f"₹{x:,.2f}")
display_df["Actual CAGR"] = display_df["Actual_CAGR"].map(lambda x: f"{x:.2%}")
display_df["Beta"] = display_df["Beta"].map(lambda x: f"{x:.2f}")
display_df["Dividend Yield"] = display_df["DivYield"].map(lambda x: f"{x:.2%}")
display_df["CAPM Return"] = display_df["CAPM_Return"].map(lambda x: f"{x:.2%}")
display_df["Total Expected Yield"] = display_df["Total_Expected_Return"].map(lambda x: f"{x:.2%}")
display_df["Portfolio Weight"] = display_df["Portfolio_Weight"].map(lambda x: f"{x:.2%}")
display_df["Capital Allocated"] = display_df["Allocated_Amount"].map(lambda x: f"₹{x:,.0f}")

st.dataframe(
    display_df[[
        "Ticker", "Name", "Type", "Sector", "CapCategory", "Live Price", 
        "Actual CAGR", "Beta", "Dividend Yield", "Total Expected Yield", 
        "Portfolio Weight", "Capital Allocated"
    ]],
    use_container_width=True,
    hide_index=True
)
