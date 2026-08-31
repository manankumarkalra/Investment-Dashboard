import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMLIT APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="15-Asset Portfolio Risk & Allocation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #0066cc;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .stDataFrame { font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. MASTER PORTFOLIO CONFIGURATION & ASSUMPTIONS
# ============================================================
START_DATE = datetime(2026, 8, 31)
CORPUS = 1_00_00_000         # ₹1 Crore Base Investment
HORIZON_YEARS = 3
RISK_FREE_RATE = 0.069       # 6.90% Risk-Free Rate Assumption
MARKET_RETURN = 0.130         # 13.0% Expected Benchmark Market Return
LOOKBACK_YEARS = 5

# Investor Persona Target Definitions
PERSONAS = {
    "Aggressive": {"target_corpus": 2_25_00_000, "bond_floor": 0.05, "bond_cap": 0.15},
    "Moderate": {"target_corpus": 2_00_00_000, "bond_floor": 0.15, "bond_cap": 0.35},
    "Conservative": {"target_corpus": 1_75_00_000, "bond_floor": 0.30, "bond_cap": 0.55},
}

for p_name, p_data in PERSONAS.items():
    p_data["target_cagr"] = (p_data["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

# User-Defined 15 Portfolio Instruments
SHORTLISTED_EQUITIES = [
    {"Ticker": "INDIANB.NS", "Name": "Indian Bank", "Sector": "Banking", "CapCategory": "Large Cap"},
    {"Ticker": "HAL.NS", "Name": "Hindustan Aeronautics (HAL)", "Sector": "Defence", "CapCategory": "Large Cap"},
    {"Ticker": "CHENNPETRO.NS", "Name": "Chennai Petroleum Corp", "Sector": "Oil & Gas", "CapCategory": "Mid Cap"},
    {"Ticker": "MAZDOCK.NS", "Name": "Mazagon Dock Shipbuilders", "Sector": "Defence", "CapCategory": "Mid Cap"},
    {"Ticker": "BSE.NS", "Name": "BSE Ltd.", "Sector": "Financial Services", "CapCategory": "Large Cap"},
    {"Ticker": "NATIONALUM.NS", "Name": "National Aluminium Co (NALCO)", "Sector": "Metals", "CapCategory": "Mid Cap"},
    {"Ticker": "FORCEMOT.NS", "Name": "Force Motors", "Sector": "Automobiles", "CapCategory": "Small/Mid Cap"},
    {"Ticker": "LLOYDSME.NS", "Name": "Lloyds Metals & Energy", "Sector": "Metals", "CapCategory": "Mid/Small Cap"},
    {"Ticker": "APARINDS.NS", "Name": "Apar Industries", "Sector": "Electrical / Industrial", "CapCategory": "Mid Cap"},
    {"Ticker": "OIL.NS", "Name": "Oil India", "Sector": "Oil & Gas", "CapCategory": "Large Cap"},
    {"Ticker": "TVSMOTOR.NS", "Name": "TVS Motor Company", "Sector": "Automobiles", "CapCategory": "Large Cap"},
]

SHORTLISTED_BONDS = [
    {"Ticker": "GSEC2029", "Name": "6.03% GOI G-Sec 2029", "Sector": "Sovereign Bond", "CapCategory": "Fixed Income", "AnnualYield": 0.0625},
    {"Ticker": "EBBETF0431.NS", "Name": "BHARAT Bond ETF April 2031", "Sector": "Target-Maturity Debt", "CapCategory": "Fixed Income", "AnnualYield": 0.0710},
]

SHORTLISTED_COMMODITIES = [
    {"Ticker": "GOLDBEES.NS", "Name": "Nippon India ETF Gold BeES", "Sector": "Precious Metals", "CapCategory": "Gold ETF", "Type": "Gold"},
    {"Ticker": "SILVERBEES.NS", "Name": "Nippon India Silver ETF", "Sector": "Precious Metals", "CapCategory": "Silver ETF", "Type": "Silver"},
]

# ============================================================
# 3. DATA FETCHING & CAPM CALCULATION ENGINE
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(tickers):
    clean_tickers = [t for t in tickers if not t.startswith("GSEC")]
    raw = yf.download(clean_tickers, period=f"{LOOKBACK_YEARS}y", interval="1wk", auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
    else:
        data = raw[["Close"]] if "Close" in raw.columns else raw
    return data.dropna(how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_portfolio_equities(equity_meta):
    tickers = [item["Ticker"] for item in equity_meta]
    px_hist = fetch_ticker_data(tickers)
    idx_hist = fetch_ticker_data(["^NSEI"])
    
    stock_ret = px_hist.pct_change().dropna(how="all")
    idx_ret = idx_hist.iloc[:, 0].pct_change().dropna()
    
    rows = []
    for item in equity_meta:
        t = item["Ticker"]
        if t in stock_ret.columns:
            joined = pd.concat([stock_ret[t], idx_ret], axis=1, join="inner").dropna()
            s, m = joined.iloc[:, 0], joined.iloc[:, 1]
            beta = np.cov(s, m)[0, 1] / np.var(m) if np.var(m) > 0 else 1.0
            ann_vol = s.std() * np.sqrt(52)
        else:
            beta, ann_vol = 1.0, 0.25
            
        capm_ret = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)
        
        try:
            info = yf.Ticker(t).info
            live_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or px_hist[t].dropna().iloc[-1])
            pe_ratio = float(info.get("trailingPE", 0.0))
            eps = float(info.get("trailingEps", 0.0))
            div_yield = float(info.get("dividendYield", 0.0) or 0.0)
            if div_yield > 1.0: div_yield /= 100.0
        except Exception:
            live_price, pe_ratio, eps, div_yield = 500.0, 20.0, 25.0, 0.015
            
        rows.append({
            "Ticker": t,
            "Name": item["Name"],
            "Sector": item["Sector"],
            "CapCategory": item["CapCategory"],
            "LivePrice": live_price,
            "PE": pe_ratio,
            "EPS": eps,
            "Beta": beta,
            "Volatility": ann_vol,
            "CAPM_Return": capm_ret,
            "DividendYield": div_yield,
            "TotalExpectedReturn": capm_ret + div_yield
        })
        
    return pd.DataFrame(rows).set_index("Ticker")

# ============================================================
# 4. SIDEBAR & USER INPUTS
# ============================================================
st.sidebar.title("🎮 Portfolio Controls")
selected_persona = st.sidebar.selectbox("Investor Persona Target", list(PERSONAS.keys()), index=1)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Economic Baseline")
st.sidebar.write(f"**Base Investment Date:** {START_DATE.strftime('%d-%b-%Y')}")
st.sidebar.write(f"**Risk-Free Rate ($R_f$):** {RISK_FREE_RATE:.2%}")
st.sidebar.write(f"**Expected Market Return ($E(R_m)$):** {MARKET_RETURN:.2%}")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# 5. PORTFOLIO ALLOCATION CALCULATION
# ============================================================
try:
    with st.spinner("Analyzing 15 instruments & calculating risk weights..."):
        equity_df = evaluate_portfolio_equities(SHORTLISTED_EQUITIES)
except Exception as err:
    st.error("Error retrieving market metrics.")
    st.exception(err)
    st.stop()

# Inverse-Beta Equity Weighting Scheme
clamped_beta = equity_df["Beta"].clip(lower=0.2)
inv_beta = 1.0 / clamped_beta
raw_eq_weights = inv_beta / inv_beta.sum()
equity_df["EquitySleeveWeight"] = raw_eq_weights

equity_total_ret = (equity_df["EquitySleeveWeight"] * equity_df["TotalExpectedReturn"]).sum()

# Bond & Commodity Returns
bonds_df = pd.DataFrame(SHORTLISTED_BONDS)
bond_ret = bonds_df["AnnualYield"].mean()

gold_hist = fetch_ticker_data(["GOLDBEES.NS"])
silver_hist = fetch_ticker_data(["SILVERBEES.NS"])
gold_ret = ((gold_hist.iloc[-1,0] / gold_hist.iloc[0,0])**(1/LOOKBACK_YEARS) - 1) if not gold_hist.empty else 0.085
silver_ret = ((silver_hist.iloc[-1,0] / silver_hist.iloc[0,0])**(1/LOOKBACK_YEARS) - 1) if not silver_hist.empty else 0.095

gold_weight, silver_weight = 0.04, 0.04
commodities_weight = gold_weight + silver_weight
investable_weight = 1.0 - commodities_weight

# Solve for Equity vs. Debt Split based on Target CAGR
persona_cfg = PERSONAS[selected_persona]
target_cagr = persona_cfg["target_cagr"]

denom = equity_total_ret - bond_ret
if abs(denom) > 1e-6:
    eq_weight = (target_cagr - (gold_weight * gold_ret + silver_weight * silver_ret) - investable_weight * bond_ret) / denom
else:
    eq_weight = investable_weight

min_eq = investable_weight - persona_cfg["bond_cap"]
max_eq = investable_weight - persona_cfg["bond_floor"]
eq_weight = float(np.clip(eq_weight, min_eq, max_eq))
bond_weight = investable_weight - eq_weight

modelled_cagr = (eq_weight * equity_total_ret) + (bond_weight * bond_ret) + (gold_weight * gold_ret) + (silver_weight * silver_ret)
projected_3y = CORPUS * ((1 + modelled_cagr) ** HORIZON_YEARS)

# ============================================================
# 6. DASHBOARD DISPLAY
# ============================================================
st.title("📊 15-Asset Portfolio Risk & Allocation Dashboard")
st.caption(f"Portfolio Base Date: **August 31, 2026** | Persona Target: **{selected_persona}**")

# Top KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Starting Capital", f"₹{CORPUS/1e7:.2f} Cr")
c2.metric("Target (3-Year)", f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr")
c3.metric("Required CAGR", f"{target_cagr:.2%}")
c4.metric("Modelled CAGR", f"{modelled_cagr:.2%}")
c5.metric("Projected Corpus", f"₹{projected_3y/1e7:.2f} Cr")

st.divider()

# Core Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Asset Class Allocation Split")
    labels = ["11 Equities Sleeve", "2 Bonds Sleeve", "Gold BeES ETF", "Silver ETF"]
    values = [eq_weight * CORPUS, bond_weight * CORPUS, gold_weight * CORPUS, silver_weight * CORPUS]
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, textinfo="percent+label")])
    fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("3-Year Growth Projection Path")
    years = np.arange(0, HORIZON_YEARS + 1)
    model_path = CORPUS * ((1 + modelled_cagr) ** years)
    target_path = CORPUS * ((1 + target_cagr) ** years)
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=years, y=model_path, mode="lines+markers", name="Modelled Trajectory"))
    fig_line.add_trace(go.Scatter(x=years, y=target_path, mode="lines", name="Target Benchmark", line=dict(dash="dash")))
    fig_line.update_layout(
        height=360,
        xaxis_title="Years Elapsed",
        yaxis_title="Portfolio Value (₹)",
        yaxis_tickprefix="₹",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Master Data Table for All 15 Instruments
st.subheader("Complete 15-Instrument Portfolio Breakdown")

# Format Equity Master Table
eq_table = equity_df.reset_index()[[
    "Ticker", "Name", "Sector", "CapCategory", "LivePrice", "PE", "Beta", 
    "CAPM_Return", "DividendYield", "TotalExpectedReturn"
]].copy()

eq_table["Portfolio Weight"] = equity_df["EquitySleeveWeight"].values * eq_weight
eq_table["Allocated Amount (₹)"] = eq_table["Portfolio Weight"] * CORPUS
eq_table["Type"] = "Equity"

# Format Bonds & Commodities Tables
bond_rows = []
for b in SHORTLISTED_BONDS:
    w = (bond_weight / len(SHORTLISTED_BONDS))
    bond_rows.append({
        "Ticker": b["Ticker"], "Name": b["Name"], "Sector": b["Sector"], "CapCategory": b["CapCategory"],
        "LivePrice": 100.0, "PE": 0.0, "Beta": 0.0, "CAPM_Return": b["AnnualYield"],
        "DividendYield": 0.0, "TotalExpectedReturn": b["AnnualYield"],
        "Portfolio Weight": w, "Allocated Amount (₹)": w * CORPUS, "Type": "Bond"
    })

comm_rows = [
    {
        "Ticker": "GOLDBEES.NS", "Name": "Nippon India ETF Gold BeES", "Sector": "Precious Metals", "CapCategory": "Gold ETF",
        "LivePrice": float(gold_hist.iloc[-1,0]) if not gold_hist.empty else 60.0, "PE": 0.0, "Beta": 0.15,
        "CAPM_Return": gold_ret, "DividendYield": 0.0, "TotalExpectedReturn": gold_ret,
        "Portfolio Weight": gold_weight, "Allocated Amount (₹)": gold_weight * CORPUS, "Type": "Gold ETF"
    },
    {
        "Ticker": "SILVERBEES.NS", "Name": "Nippon India Silver ETF", "Sector": "Precious Metals", "CapCategory": "Silver ETF",
        "LivePrice": float(silver_hist.iloc[-1,0]) if not silver_hist.empty else 75.0, "PE": 0.0, "Beta": 0.25,
        "CAPM_Return": silver_ret, "DividendYield": 0.0, "TotalExpectedReturn": silver_ret,
        "Portfolio Weight": silver_weight, "Allocated Amount (₹)": silver_weight * CORPUS, "Type": "Silver ETF"
    }
]

full_df = pd.concat([eq_table, pd.DataFrame(bond_rows), pd.DataFrame(comm_rows)], ignore_index=True)

# Format numerical columns for table view
display_df = full_df.copy()
display_df["LivePrice"] = display_df["LivePrice"].map(lambda x: f"₹{x:,.2f}")
display_df["Beta"] = display_df["Beta"].map(lambda x: f"{x:.2f}")
display_df["CAPM_Return"] = display_df["CAPM_Return"].map(lambda x: f"{x:.2%}")
display_df["DividendYield"] = display_df["DividendYield"].map(lambda x: f"{x:.2%}")
display_df["TotalExpectedReturn"] = display_df["TotalExpectedReturn"].map(lambda x: f"{x:.2%}")
display_df["Portfolio Weight"] = display_df["Portfolio Weight"].map(lambda x: f"{x:.2%}")
display_df["Allocated Amount (₹)"] = display_df["Allocated Amount (₹)"].map(lambda x: f"₹{x:,.0f}")

st.dataframe(display_df[[
    "Ticker", "Name", "Type", "Sector", "CapCategory", "Beta", 
    "CAPM_Return", "DividendYield", "TotalExpectedReturn", "Portfolio Weight", "Allocated Amount (₹)"
]], use_container_width=True, hide_index=True)
