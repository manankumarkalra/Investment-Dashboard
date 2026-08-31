%%writefile app.py
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# --- Global assumptions ---
CORPUS          = 1_00_00_000      # ₹1 Cr starting corpus
HORIZON_YEARS   = 3
RISK_FREE_RATE  = 0.069            # Rf: current 10Y India G-Sec yield (~6.9%). Update from RBI/CCIL if needed.
MARKET_RETURN   = 0.13             # Rm: long-run expected Nifty 500 total return (price CAGR + dividend yield)
LOOKBACK_YEARS  = 5                # history window used to estimate beta / volatility / realised CAGR

PERSONAS = {
    "Aggressive":   {"target_corpus": 2_25_00_000},
    "Moderate":     {"target_corpus": 2_00_00_000},
    "Conservative": {"target_corpus": 1_75_00_000},
}
for name, p in PERSONAS.items():
    p["target_cagr"] = (p["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

# --- Asset Universe ---
universe = pd.DataFrame([
    dict(Ticker="INDIANB.NS",    Name="Indian Bank",                     AssetClass="Equity", Sector="Banking",                    MarketCapCr=117286.89, PE=9.45,  EPS=92.47),
    dict(Ticker="HAL.NS",        Name="Hindustan Aeronautics",           AssetClass="Equity", Sector="Defence / Aerospace",         MarketCapCr=327699.75, PE=34.70, EPS=138.75),
    dict(Ticker="CHENNPETRO.NS", Name="Chennai Petroleum Corporation",   AssetClass="Equity", Sector="Oil & Gas",                   MarketCapCr=21027.03,  PE=4.92,  EPS=277.69),
    dict(Ticker="MAZDOCK.NS",    Name="Mazagon Dock Shipbuilders",       AssetClass="Equity", Sector="Defence / Shipbuilding",      MarketCapCr=102377.84, PE=39.88, EPS=62.63),
    dict(Ticker="BSE.NS",        Name="BSE Ltd.",                        AssetClass="Equity", Sector="Financial Services",          MarketCapCr=134925.97, PE=50.17, EPS=65.48),
    dict(Ticker="NATIONALUM.NS", Name="National Aluminium Company",      AssetClass="Equity", Sector="Metals",                      MarketCapCr=73970.35,  PE=10.28, EPS=36.78),
    dict(Ticker="FORCEMOT.NS",   Name="Force Motors",                    AssetClass="Equity", Sector="Automobiles",                 MarketCapCr=23207.35,  PE=21.25, EPS=824.82),
    dict(Ticker="LLOYDSME.NS",   Name="Lloyds Metals & Energy",          AssetClass="Equity", Sector="Metals",                      MarketCapCr=103102.04, PE=24.63, EPS=72.60),
    dict(Ticker="APARINDS.NS",   Name="Apar Industries",                 AssetClass="Equity", Sector="Electrical / Industrial",     MarketCapCr=69947.11,  PE=61.89, EPS=284.94),
    dict(Ticker="OIL.NS",        Name="Oil India",                       AssetClass="Equity", Sector="Oil & Gas",                   MarketCapCr=77150.01,  PE=12.12, EPS=40.03),
    dict(Ticker="TVSMOTOR.NS",   Name="TVS Motor Company",               AssetClass="Equity", Sector="Automobiles",                 MarketCapCr=207420.66, PE=50.75, EPS=85.17),
    dict(Ticker=None,            Name="6.03% GOI G-Sec 2029",            AssetClass="Bond",   Sector="Sovereign",                   MarketCapCr=None, PE=None, EPS=None),
    dict(Ticker="EBBETF0431.NS", Name="BHARAT Bond ETF - April 2031",    AssetClass="Bond",   Sector="AAA PSU/CPSE",                MarketCapCr=None, PE=None, EPS=None),
    dict(Ticker="GOLDBEES.NS",   Name="Nippon India ETF Gold BeES",      AssetClass="Gold",   Sector="Gold",                        MarketCapCr=None, PE=None, EPS=None),
    dict(Ticker="SILVERBEES.NS", Name="Nippon India Silver ETF",         AssetClass="Silver", Sector="Silver",                      MarketCapCr=None, PE=None, EPS=None),
]).set_index("Ticker", drop=False)

# --- Data Fetching & CAPM Calculations ---
NIFTY500_TICKER = "^CRSLDX" # This is just a proxy, actual NIFTY500 index may vary, if unavailable, ^NSEI will be used

equity_tickers = universe[universe.AssetClass == "Equity"]["Ticker"].tolist()

@st.cache_data
def fetch_history(tickers, years=LOOKBACK_YEARS):
    data = yf.download(tickers, period=f"{years}y", interval="1wk", auto_adjust=True, progress=False)["Close"]
    return data.dropna(how="all")

px_hist = fetch_history(equity_tickers)
try:
    idx_hist = fetch_history([NIFTY500_TICKER]).iloc[:, 0]
except Exception:
    idx_hist = fetch_history(["^NSEI"]).iloc[:, 0]

stock_ret = px_hist.pct_change().dropna()
idx_ret   = idx_hist.pct_change().dropna()

rows = []
for t in equity_tickers:
    common = stock_ret[t].align(idx_ret, join="inner")
    s, m = common[0].dropna(), common[1].dropna()
    s, m = s.align(m, join="inner")
    beta = np.cov(s, m)[0, 1] / np.var(m) if np.var(m) != 0 else np.nan
    ann_vol = s.std() * np.sqrt(52)
    ke = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)

    hist_px = px_hist[t].dropna()
    yrs = (hist_px.index[-1] - hist_px.index[0]).days / 365.25
    realised_cagr = (hist_px.iloc[-1] / hist_px.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan

    try:
        info = yf.Ticker(t).info
        div_yield = info.get("dividendYield", 0) or 0
        last_price = info.get("currentPrice") or hist_px.iloc[-1]
    except Exception:
        div_yield, last_price = 0, hist_px.iloc[-1]

    rows.append(dict(Ticker=t, LivePrice=last_price, Beta=beta, AnnVolatility=ann_vol,
                      CAPM_Return=ke, Realised5Y_CAGR=realised_cagr, DividendYield=div_yield))

capm_df = pd.DataFrame(rows).set_index("Ticker")
capm_df["RiskBucket"] = pd.cut(capm_df.Beta, bins=[-np.inf, 0.8, 1.2, np.inf],
                                labels=["Low (Defensive)", "Medium (Core)", "High (Aggressive)"])
capm_df = capm_df.join(universe[["Name", "Sector", "PE", "EPS"]])

MIN_W, MAX_W = 0.04, 0.14   # per-stock floor/cap inside the equity sleeve

inv_beta = 1 / capm_df["Beta"].clip(lower=0.2)   # floor beta at 0.2 to avoid explosive weights
raw_w = inv_beta / inv_beta.sum()
raw_w = raw_w.clip(lower=MIN_W, upper=MAX_W)
capm_df["EquitySleeveWeight"] = raw_w / raw_w.sum()   # renormalise to 100% of the equity sleeve

equity_expected_return = (capm_df["EquitySleeveWeight"] * capm_df["CAPM_Return"]).sum()
equity_expected_divyld = (capm_df["EquitySleeveWeight"] * capm_df["DividendYield"] / 100).sum() # Corrected: Divide DividendYield by 100

# --- Gold/Silver/Bond Returns ---
GOLD_SILVER_ANCHOR = 0.08

gold_ticker = universe[universe.AssetClass == "Gold"].index[0]
silver_ticker = universe[universe.AssetClass == "Silver"].index[0]

gold_silver_px_hist = fetch_history([gold_ticker, silver_ticker])
gold_px_hist = gold_silver_px_hist[gold_ticker].dropna()
silver_px_hist = gold_silver_px_hist[silver_ticker].dropna()

gold_yrs = (gold_px_hist.index[-1] - gold_px_hist.index[0]).days / 365.25
gold_return = (gold_px_hist.iloc[-1] / gold_px_hist.iloc[0]) ** (1 / gold_yrs) - 1 if gold_yrs > 0 else np.nan

silver_yrs = (silver_px_hist.index[-1] - silver_px_hist.index[0]).days / 365.25
silver_return = (silver_px_hist.iloc[-1] / silver_px_hist.iloc[0]) ** (1 / silver_yrs) - 1 if silver_yrs > 0 else np.nan

gold_silver_return = (gold_return + silver_return) / 2

bond_blended_return = 0.07 # Placeholder, actual calculation from bond instruments/ETFs is pending

# --- Allocation Results ---
persona_bond_floor = {"Aggressive": 0.05, "Moderate": 0.15, "Conservative": 0.30}
persona_bond_cap    = {"Aggressive": 0.15, "Moderate": 0.35, "Conservative": 0.55}

alloc_results = {}
for name, p in PERSONAS.items():
    target = p["target_cagr"]
    investable = 1 - GOLD_SILVER_ANCHOR   # split between equity & bond
    num = target - GOLD_SILVER_ANCHOR * gold_silver_return - investable * bond_blended_return
    den = (equity_expected_return - bond_blended_return)
    equity_w = num / den
    equity_w = np.clip(equity_w, investable - persona_bond_cap[name], investable - persona_bond_floor[name])
    bond_w = investable - equity_w

    achieved = equity_w * equity_expected_return + bond_w * bond_blended_return + GOLD_SILVER_ANCHOR * gold_silver_return
    feasible = abs(achieved - target) < 0.005

    alloc_results[name] = dict(EquityWeight=equity_w, BondWeight=bond_w,
                                GoldSilverWeight=GOLD_SILVER_ANCHOR,
                                TargetCAGR=target, AchievedCAGR=achieved, Feasible=feasible)

alloc_df = pd.DataFrame(alloc_results).T
alloc_df[["EquityWeight","BondWeight","GoldSilverWeight","TargetCAGR","AchievedCAGR"]] = \
    alloc_df[["EquityWeight","BondWeight","GoldSilverWeight","TargetCAGR","AchievedCAGR"]].astype(float).round(4)

def build_full_weights(persona):
    a = alloc_results[persona]
    w = capm_df["EquitySleeveWeight"] * a["EquityWeight"]
    w.name = persona
    bond_rows = pd.Series({
        "GSEC2029": a["BondWeight"] * 0.5,
        "EBBETF0431.NS": a["BondWeight"] * 0.5,
        "GOLDBEES.NS": a["GoldSilverWeight"] * 0.5,
        "SILVERBEES.NS": a["GoldSilverWeight"] * 0.5,
    }, name=persona)
    return pd.concat([w, bond_rows])

full_weights = pd.concat([build_full_weights(p) for p in PERSONAS], axis=1)
full_weights_rupees = (full_weights * CORPUS).round(0)

names_map = universe["Name"].to_dict()
names_map["GSEC2029"] = "6.03% GOI G-Sec 2029"

# --- Streamlit Dashboard ---
st.set_page_config(layout="wide", page_title="Portfolio Allocation Dashboard")
st.title("Persona-Based Portfolio Allocation")
st.markdown("This dashboard provides a persona-based asset allocation strategy for a ₹1 Crore corpus over a 3-year horizon, assuming investment on **August 31st, 2026**.")

selected_persona = st.sidebar.selectbox(
    "Select a Persona:",
    list(PERSONAS.keys())
)

a = alloc_results[selected_persona]

st.sidebar.subheader("Persona Details")
st.sidebar.write(f"**Target Corpus:** ₹{PERSONAS[selected_persona]['target_corpus']:,.0f}")
st.sidebar.write(f"**Target CAGR:** {a['TargetCAGR']*100:.1f}%")
st.sidebar.write(f"**Achieved CAGR:** {a['AchievedCAGR']*100:.1f}%")
st.sidebar.write(f"**Feasible:** {'Yes' if a['Feasible'] else 'No'}")


# Create two columns for the charts
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{selected_persona} — Instrument Allocation")
    w = full_weights[selected_persona]
    rupees = (w * CORPUS)
    labels = [names_map.get(i, i) for i in w.index]

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=rupees.values, hole=0.45,
                                    textinfo="label+percent", showlegend=True)])
    fig_pie.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader(f"{selected_persona} — Corpus Growth ({HORIZON_YEARS}Y)")
    years = np.arange(0, HORIZON_YEARS + 1)
    growth = CORPUS * (1 + a["AchievedCAGR"]) ** years
    target_growth = CORPUS * (1 + a["TargetCAGR"]) ** years

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=years, y=growth, mode="lines+markers", name="Modelled Path"))
    fig_line.add_trace(go.Scatter(x=years, y=target_growth, mode="lines", name="Target Path", line=dict(dash="dash")))
    fig_line.update_layout(height=400, template="plotly_white",
                           xaxis_title="Year", yaxis_title="Corpus (₹)",
                           margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_line, use_container_width=True)

st.subheader("Contribution to Return Breakdown")
breakdown = pd.DataFrame({
    "Component": ["Equity capital gains", "Equity dividends", "Bond interest (YTM)", "Gold/Silver appreciation"],
    "Contribution to return": [
        a["EquityWeight"] * (equity_expected_return - equity_expected_divyld),
        a["EquityWeight"] * equity_expected_divyld,
        a["BondWeight"] * bond_blended_return,
        a["GoldSilverWeight"] * gold_silver_return,
    ]})
breakdown["Contribution to return"] = (breakdown["Contribution to return"] * 100).round(2).astype(str) + "%"
st.dataframe(breakdown.style.hide(axis="index"), use_container_width=True)

st.subheader("Rupee Allocation per Instrument")
rupee_alloc_df = full_weights_rupees[[selected_persona]].rename(index=names_map).rename(columns={selected_persona: "₹ Allocation"})
st.dataframe(rupee_alloc_df, use_container_width=True)
