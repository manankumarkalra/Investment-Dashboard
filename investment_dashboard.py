
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ============================================================
# STREAMLIT CONFIG
# MUST BE THE FIRST STREAMLIT COMMAND
# ============================================================
st.set_page_config(
    page_title="Investment Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL ASSUMPTIONS
# ============================================================
CORPUS = 1_00_00_000          # ₹1 crore
HORIZON_YEARS = 3
RISK_FREE_RATE = 0.069        # 6.9% modelling assumption
MARKET_RETURN = 0.13          # 13.0% modelling assumption
LOOKBACK_YEARS = 5

PERSONAS = {
    "Aggressive": {"target_corpus": 2_25_00_000},
    "Moderate": {"target_corpus": 2_00_00_000},
    "Conservative": {"target_corpus": 1_75_00_000},
}

for name, p in PERSONAS.items():
    p["target_cagr"] = (p["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

# ============================================================
# FIXED 15-INSTRUMENT UNIVERSE
# 11 EQUITIES + 2 BONDS + 1 GOLD + 1 SILVER
# ============================================================
universe = pd.DataFrame([
    # ---- Equity (11) ----
    dict(Ticker="INDIANB.NS",    Name="Indian Bank",                   AssetClass="Equity", Sector="Banking",                MarketCapCr=117286.89, PE=9.45,  EPS=92.47),
    dict(Ticker="HAL.NS",        Name="Hindustan Aeronautics",         AssetClass="Equity", Sector="Defence / Aerospace",     MarketCapCr=327699.75, PE=34.70, EPS=138.75),
    dict(Ticker="CHENNPETRO.NS", Name="Chennai Petroleum Corporation", AssetClass="Equity", Sector="Oil & Gas",               MarketCapCr=21027.03,  PE=4.92,  EPS=277.69),
    dict(Ticker="MAZDOCK.NS",    Name="Mazagon Dock Shipbuilders",     AssetClass="Equity", Sector="Defence / Shipbuilding",  MarketCapCr=102377.84, PE=39.88, EPS=62.63),
    dict(Ticker="BSE.NS",        Name="BSE Ltd.",                      AssetClass="Equity", Sector="Financial Services",      MarketCapCr=134925.97, PE=50.17, EPS=65.48),
    dict(Ticker="NATIONALUM.NS", Name="National Aluminium Company",    AssetClass="Equity", Sector="Metals",                  MarketCapCr=73970.35,  PE=10.28, EPS=36.78),
    dict(Ticker="FORCEMOT.NS",   Name="Force Motors",                 AssetClass="Equity", Sector="Automobiles",             MarketCapCr=23207.35,  PE=21.25, EPS=824.82),
    dict(Ticker="LLOYDSME.NS",   Name="Lloyds Metals & Energy",       AssetClass="Equity", Sector="Metals",                  MarketCapCr=103102.04, PE=24.63, EPS=72.60),
    dict(Ticker="APARINDS.NS",   Name="Apar Industries",              AssetClass="Equity", Sector="Electrical / Industrial", MarketCapCr=69947.11,  PE=61.89, EPS=284.94),
    dict(Ticker="OIL.NS",        Name="Oil India",                    AssetClass="Equity", Sector="Oil & Gas",               MarketCapCr=77150.01,  PE=12.12, EPS=40.03),
    dict(Ticker="TVSMOTOR.NS",   Name="TVS Motor Company",            AssetClass="Equity", Sector="Automobiles",             MarketCapCr=207420.66, PE=50.75, EPS=85.17),

    # ---- Fixed Income (2) ----
    dict(Ticker=None,            Name="6.03% GOI G-Sec 2029",          AssetClass="Bond", Sector="Sovereign", MarketCapCr=np.nan, PE=np.nan, EPS=np.nan),
    dict(Ticker="EBBETF0431.NS", Name="BHARAT Bond ETF - April 2031", AssetClass="Bond", Sector="AAA PSU/CPSE", MarketCapCr=np.nan, PE=np.nan, EPS=np.nan),

    # ---- Precious Metals (2) ----
    dict(Ticker="GOLDBEES.NS",   Name="Nippon India ETF Gold BeES",   AssetClass="Gold",   Sector="Gold",   MarketCapCr=np.nan, PE=np.nan, EPS=np.nan),
    dict(Ticker="SILVERBEES.NS", Name="Nippon India Silver ETF",      AssetClass="Silver", Sector="Silver", MarketCapCr=np.nan, PE=np.nan, EPS=np.nan),
]).set_index("Ticker", drop=False)

NIFTY500_TICKERS = ["^CRSLDX", "^NSEI"]

# ============================================================
# DATA HELPERS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(tickers, years=LOOKBACK_YEARS):
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period=f"{years}y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    # yfinance gives different shapes for one vs many tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            data = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        data = raw[["Close"]].copy()
        data.columns = [tickers[0]]

    return data.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def get_current_and_dividend(ticker, fallback_price):
    if not ticker:
        return fallback_price, 0.0

    try:
        info = yf.Ticker(ticker).info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or fallback_price

        # Yahoo Finance normally returns dividendYield as a decimal
        # (e.g. 0.025 for 2.5%), not 2.5.
        dividend_yield = info.get("dividendYield")
        if dividend_yield is None:
            dividend_yield = 0.0

        dividend_yield = float(dividend_yield)
        if dividend_yield > 1.0:
            dividend_yield /= 100.0

        return float(current_price), dividend_yield
    except Exception:
        return float(fallback_price), 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def build_market_data(equity_tickers):
    px_hist = fetch_history(equity_tickers)
    if px_hist.empty:
        raise RuntimeError("Yahoo Finance returned no equity price history.")

    # Benchmark: try NIFTY 500 first, then NIFTY 50.
    idx_hist = pd.Series(dtype=float)
    benchmark_used = None

    for benchmark in NIFTY500_TICKERS:
        tmp = fetch_history([benchmark])
        if not tmp.empty:
            idx_hist = tmp.iloc[:, 0].dropna()
            if not idx_hist.empty:
                benchmark_used = benchmark
                break

    if idx_hist.empty:
        raise RuntimeError("Could not download a benchmark index from Yahoo Finance.")

    stock_ret = px_hist.pct_change().dropna(how="all")
    idx_ret = idx_hist.pct_change().dropna()

    rows = []
    for ticker in equity_tickers:
        if ticker not in stock_ret.columns:
            continue

        joined = pd.concat([stock_ret[ticker], idx_ret], axis=1, join="inner").dropna()
        if len(joined) < 30:
            continue

        s = joined.iloc[:, 0]
        m = joined.iloc[:, 1]

        market_var = np.var(m)
        beta = np.cov(s, m)[0, 1] / market_var if market_var > 0 else np.nan
        ann_vol = s.std() * np.sqrt(52)
        capm_return = RISK_FREE_RATE + beta * (MARKET_RETURN - RISK_FREE_RATE)

        hist_px = px_hist[ticker].dropna()
        if len(hist_px) >= 2:
            years = (hist_px.index[-1] - hist_px.index[0]).days / 365.25
            realised_cagr = (
                (hist_px.iloc[-1] / hist_px.iloc[0]) ** (1 / years) - 1
                if years > 0 and hist_px.iloc[0] > 0
                else np.nan
            )
            fallback_price = float(hist_px.iloc[-1])
        else:
            realised_cagr = np.nan
            fallback_price = np.nan

        live_price, dividend_yield = get_current_and_dividend(ticker, fallback_price)

        rows.append({
            "Ticker": ticker,
            "LivePrice": live_price,
            "Beta": beta,
            "AnnVolatility": ann_vol,
            "CAPM_Return": capm_return,
            "Realised5Y_CAGR": realised_cagr,
            "DividendYield": dividend_yield,
        })

    capm_df = pd.DataFrame(rows)
    if capm_df.empty:
        raise RuntimeError("No equity-level market data could be calculated.")

    capm_df = capm_df.set_index("Ticker")
    capm_df["RiskBucket"] = pd.cut(
        capm_df["Beta"],
        bins=[-np.inf, 0.8, 1.2, np.inf],
        labels=["Low (Defensive)", "Medium (Core)", "High (Aggressive)"]
    )

    capm_df = capm_df.join(universe[["Name", "Sector", "PE", "EPS"]])
    return capm_df, benchmark_used


# ============================================================
# PORTFOLIO CALCULATION
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_portfolio():
    equity_tickers = universe[universe.AssetClass == "Equity"]["Ticker"].dropna().tolist()
    capm_df, benchmark_used = build_market_data(tuple(equity_tickers))

    # Risk-adjusted equity sleeve weights:
    # inverse-beta score, bounded and renormalised.
    MIN_W, MAX_W = 0.04, 0.14

    beta_for_weighting = capm_df["Beta"].clip(lower=0.2)
    inv_beta = 1 / beta_for_weighting
    raw_w = inv_beta / inv_beta.sum()
    raw_w = raw_w.clip(lower=MIN_W, upper=MAX_W)
    equity_sleeve_weight = raw_w / raw_w.sum()

    capm_df["EquitySleeveWeight"] = equity_sleeve_weight

    equity_expected_capm = (
        capm_df["EquitySleeveWeight"] * capm_df["CAPM_Return"]
    ).sum()

    equity_expected_dividend = (
        capm_df["EquitySleeveWeight"] * capm_df["DividendYield"]
    ).sum()

    # Separate precious-metal return estimates.
    metals_hist = fetch_history(["GOLDBEES.NS", "SILVERBEES.NS"])
    metal_returns = {}

    for ticker in ["GOLDBEES.NS", "SILVERBEES.NS"]:
        if ticker in metals_hist.columns:
            px = metals_hist[ticker].dropna()
            if len(px) >= 2:
                yrs = (px.index[-1] - px.index[0]).days / 365.25
                metal_returns[ticker] = (
                    (px.iloc[-1] / px.iloc[0]) ** (1 / yrs) - 1
                    if yrs > 0 and px.iloc[0] > 0
                    else np.nan
                )
            else:
                metal_returns[ticker] = np.nan
        else:
            metal_returns[ticker] = np.nan

    gold_return = metal_returns.get("GOLDBEES.NS", np.nan)
    silver_return = metal_returns.get("SILVERBEES.NS", np.nan)

    # Fixed income assumptions for the locked instruments.
    # These are explicit modelling inputs, not live bond pricing.
    gsec_return = 0.0625
    bharat_bond_return = 0.0710
    bond_blended_return = (gsec_return + bharat_bond_return) / 2

    # Current design assumption: 4% gold + 4% silver.
    GOLD_WEIGHT = 0.04
    SILVER_WEIGHT = 0.04

    persona_bond_floor = {
        "Aggressive": 0.05,
        "Moderate": 0.15,
        "Conservative": 0.30,
    }
    persona_bond_cap = {
        "Aggressive": 0.15,
        "Moderate": 0.35,
        "Conservative": 0.55,
    }

    alloc_results = {}

    # Use separate gold and silver returns. If missing, fall back to 0%.
    gold_return_used = 0.0 if pd.isna(gold_return) else float(gold_return)
    silver_return_used = 0.0 if pd.isna(silver_return) else float(silver_return)

    metals_return = (
        GOLD_WEIGHT * gold_return_used +
        SILVER_WEIGHT * silver_return_used
    )
    fixed_metals_weight = GOLD_WEIGHT + SILVER_WEIGHT
    investable = 1 - fixed_metals_weight

    for persona, p in PERSONAS.items():
        target = p["target_cagr"]

        denominator = equity_expected_capm - bond_blended_return
        if abs(denominator) < 1e-9:
            equity_weight = investable
        else:
            numerator = (
                target
                - metals_return
                - investable * bond_blended_return
            )
            equity_weight = numerator / denominator

        min_equity = investable - persona_bond_cap[persona]
        max_equity = investable - persona_bond_floor[persona]
        equity_weight = float(np.clip(equity_weight, min_equity, max_equity))
        bond_weight = investable - equity_weight

        achieved = (
            equity_weight * equity_expected_capm
            + bond_weight * bond_blended_return
            + metals_return
        )

        alloc_results[persona] = {
            "EquityWeight": equity_weight,
            "BondWeight": bond_weight,
            "GoldWeight": GOLD_WEIGHT,
            "SilverWeight": SILVER_WEIGHT,
            "TargetCAGR": target,
            "AchievedCAGR": achieved,
            "Feasible": abs(achieved - target) < 0.005,
        }

    def build_full_weights(persona):
        a = alloc_results[persona]

        equity_weights = capm_df["EquitySleeveWeight"] * a["EquityWeight"]

        other_weights = pd.Series({
            "GSEC2029": a["BondWeight"] * 0.50,
            "EBBETF0431.NS": a["BondWeight"] * 0.50,
            "GOLDBEES.NS": a["GoldWeight"],
            "SILVERBEES.NS": a["SilverWeight"],
        })

        return pd.concat([equity_weights, other_weights])

    full_weights = pd.concat(
        {persona: build_full_weights(persona) for persona in PERSONAS},
        axis=1
    )

    # Ensure all 15 instruments exist in every persona column.
    expected_index = [
        *equity_tickers,
        "GSEC2029",
        "EBBETF0431.NS",
        "GOLDBEES.NS",
        "SILVERBEES.NS",
    ]
    full_weights = full_weights.reindex(expected_index).fillna(0.0)

    return (
        capm_df,
        benchmark_used,
        equity_expected_capm,
        equity_expected_dividend,
        bond_blended_return,
        gold_return_used,
        silver_return_used,
        alloc_results,
        full_weights,
    )


# ============================================================
# UI
# ============================================================
st.title("📊 Persona-Based Investment Portfolio Dashboard")
st.caption(
    "₹1 crore corpus • 3-year horizon • 11 equities + 2 bonds + 1 gold + 1 silver"
)

with st.sidebar:
    st.header("Portfolio Controls")
    selected_persona = st.selectbox(
        "Select investor persona",
        list(PERSONAS.keys()),
        index=1,
    )

    st.divider()
    st.subheader("Model assumptions")
    st.write(f"Risk-free rate: **{RISK_FREE_RATE:.1%}**")
    st.write(f"Market return assumption: **{MARKET_RETURN:.1%}**")
    st.write(f"Historical lookback: **{LOOKBACK_YEARS} years**")

    if st.button("🔄 Refresh live market data"):
        st.cache_data.clear()
        st.rerun()

# ============================================================
# LOAD MODEL
# ============================================================
try:
    with st.spinner("Downloading live market data and calculating risk metrics..."):
        (
            capm_df,
            benchmark_used,
            equity_expected_capm,
            equity_expected_dividend,
            bond_blended_return,
            gold_return,
            silver_return,
            alloc_results,
            full_weights,
        ) = calculate_portfolio()
except Exception as exc:
    st.error("The dashboard could not build the live market dataset.")
    st.exception(exc)
    st.stop()

a = alloc_results[selected_persona]
target = PERSONAS[selected_persona]["target_corpus"]
expected_final = CORPUS * (1 + a["AchievedCAGR"]) ** HORIZON_YEARS
required_cagr = a["TargetCAGR"]

# ============================================================
# KPI CARDS
# ============================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Starting Corpus", "₹1.00 Cr")
k2.metric("Target Corpus", f"₹{target/1e7:.2f} Cr")
k3.metric("Required CAGR", f"{required_cagr:.1%}")
k4.metric("Modelled CAGR", f"{a['AchievedCAGR']:.1%}")
k5.metric("Modelled 3Y Corpus", f"₹{expected_final/1e7:.2f} Cr")

# ============================================================
# ASSET ALLOCATION / CORPUS GROWTH
# ============================================================
left, right = st.columns(2)

with left:
    st.subheader(f"{selected_persona} — Asset Allocation")

    weights = full_weights[selected_persona].copy()
    names = universe["Name"].to_dict()
    names["GSEC2029"] = "6.03% GOI G-Sec 2029"
    weights.index = [names.get(i, i) for i in weights.index]

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=weights.index,
                values=(weights * CORPUS).values,
                hole=0.45,
                textinfo="percent",
                hovertemplate="%{label}<br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig_pie.update_layout(
        height=430,
        template="plotly_white",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="v"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader(f"{selected_persona} — 3-Year Corpus Projection")

    years = np.arange(0, HORIZON_YEARS + 1)
    modelled_path = CORPUS * (1 + a["AchievedCAGR"]) ** years
    target_path = CORPUS * (1 + a["TargetCAGR"]) ** years

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=years,
            y=modelled_path,
            mode="lines+markers",
            name="Modelled",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=years,
            y=target_path,
            mode="lines",
            name="Target",
            line=dict(dash="dash"),
        )
    )
    fig_line.update_layout(
        height=430,
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="Corpus (₹)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis_tickprefix="₹",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ============================================================
# ASSET CLASS SUMMARY
# ============================================================
st.subheader("Asset-Class Allocation")

asset_summary = pd.DataFrame({
    "Asset Class": ["Equity", "Bonds", "Gold", "Silver"],
    "Weight": [
        a["EquityWeight"],
        a["BondWeight"],
        a["GoldWeight"],
        a["SilverWeight"],
    ],
})
asset_summary["Allocation (₹)"] = asset_summary["Weight"] * CORPUS
asset_summary["Weight"] = asset_summary["Weight"].map(lambda x: f"{x:.1%}")
asset_summary["Allocation (₹)"] = asset_summary["Allocation (₹)"].map(lambda x: f"₹{x:,.0f}")
st.dataframe(asset_summary, use_container_width=True, hide_index=True)

# ============================================================
# CAPM / RISK TABLE
# ============================================================
st.subheader("Equity Risk & CAPM Analysis")

risk_table = capm_df.reset_index().rename(columns={
    "Ticker": "Ticker",
    "Name": "Company",
    "Sector": "Sector",
    "LivePrice": "Live Price (₹)",
    "Beta": "Beta",
    "AnnVolatility": "Annual Volatility",
    "CAPM_Return": "CAPM Expected Return",
    "Realised5Y_CAGR": "Realised 5Y CAGR",
    "DividendYield": "Dividend Yield",
    "RiskBucket": "Risk Bucket",
})

risk_table["Live Price (₹)"] = risk_table["Live Price (₹)"].round(2)
risk_table["Beta"] = risk_table["Beta"].round(2)
risk_table["Annual Volatility"] = risk_table["Annual Volatility"].map(lambda x: f"{x:.1%}")
risk_table["CAPM Expected Return"] = risk_table["CAPM Expected Return"].map(lambda x: f"{x:.1%}")
risk_table["Realised 5Y CAGR"] = risk_table["Realised 5Y CAGR"].map(
    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
)
risk_table["Dividend Yield"] = risk_table["Dividend Yield"].map(lambda x: f"{x:.1%}")

st.dataframe(
    risk_table[[
        "Company", "Sector", "Live Price (₹)", "Beta",
        "Annual Volatility", "CAPM Expected Return",
        "Realised 5Y CAGR", "Dividend Yield", "Risk Bucket"
    ]],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# RISK / RETURN SCATTER
# ============================================================
st.subheader("Expected Return vs Risk")

scatter = go.Figure()
for risk_bucket in capm_df["RiskBucket"].dropna().unique():
    subset = capm_df[capm_df["RiskBucket"] == risk_bucket]
    scatter.add_trace(
        go.Scatter(
            x=subset["AnnVolatility"],
            y=subset["CAPM_Return"],
            mode="markers+text",
            text=subset["Name"],
            textposition="top center",
            name=str(risk_bucket),
            hovertemplate=(
                "%{text}<br>"
                "Volatility: %{x:.1%}<br>"
                "CAPM Return: %{y:.1%}<extra></extra>"
            ),
        )
    )

scatter.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Annualised Volatility",
    yaxis_title="CAPM Expected Return",
)
st.plotly_chart(scatter, use_container_width=True)

# ============================================================
# INSTRUMENT-LEVEL ALLOCATION
# ============================================================
st.subheader(f"{selected_persona} — ₹ Allocation by Instrument")

alloc_table = pd.DataFrame({
    "Instrument": [names.get(i, i) for i in full_weights.index],
    "Weight": full_weights[selected_persona].values,
})
alloc_table["Allocation (₹)"] = alloc_table["Weight"] * CORPUS
alloc_table["Weight"] = alloc_table["Weight"].map(lambda x: f"{x:.2%}")
alloc_table["Allocation (₹)"] = alloc_table["Allocation (₹)"].map(lambda x: f"₹{x:,.0f}")

st.dataframe(alloc_table, use_container_width=True, hide_index=True)

# ============================================================
# RETURN BREAKDOWN
# ============================================================
st.subheader("Contribution to Expected Return")

equity_capital_gain = max(equity_expected_capm - equity_expected_dividend, 0.0)
breakdown = pd.DataFrame({
    "Component": [
        "Equity capital gains (CAPM)",
        "Equity dividends",
        "Bond interest / yield assumption",
        "Gold appreciation",
        "Silver appreciation",
    ],
    "Contribution": [
        a["EquityWeight"] * equity_capital_gain,
        a["EquityWeight"] * equity_expected_dividend,
        a["BondWeight"] * bond_blended_return,
        a["GoldWeight"] * gold_return,
        a["SilverWeight"] * silver_return,
    ],
})
breakdown["Contribution"] = breakdown["Contribution"].map(lambda x: f"{x:.2%}")
st.dataframe(breakdown, use_container_width=True, hide_index=True)

# ============================================================
# MODEL NOTES
# ============================================================
with st.expander("Methodology & limitations"):
    st.markdown(
        """
        **Selection universe:** 11 fixed equities, 2 fixed-income instruments,
        1 gold ETF and 1 silver ETF.

        **CAPM:** Expected equity return is estimated as
        Risk-free rate + Beta × (Market return − Risk-free rate).

        **Equity sleeve weighting:** inverse-beta weighting, subject to a
        4% minimum and 14% maximum per stock, then normalised.

        **Dividends:** Yahoo Finance dividend yield is treated as a decimal
        rate and included separately from capital gains.

        **Bonds:** the two bond instruments currently use explicit modelling
        assumptions (6.25% for the G-Sec and 7.10% for the Bharat Bond ETF)
        rather than live bond-yield pulls.

        **Gold and silver:** historical annualised price returns are estimated
        separately from Yahoo Finance. They are not assumed to be equivalent.

        **Important:** This is an analytical model, not investment advice.
        Historical returns and model assumptions do not guarantee future results.
        """
    )

st.caption(
    f"Benchmark used for beta estimation: {benchmark_used}. "
    "Click 'Refresh live market data' in the sidebar to clear the cache."
)
