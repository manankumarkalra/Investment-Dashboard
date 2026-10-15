import json
import os
import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
import yfinance as yf
import requests
from scipy.optimize import minimize

# Optional dependency guard: the app will still start if Streamlit Cloud
# has not installed the auto-refresh package yet.
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

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

# The browser session refreshes once per day while the dashboard is open
# when streamlit-autorefresh is installed. Cache TTLs also expire daily.
if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=24 * 60 * 60 * 1000, key="daily_dashboard_refresh")
TODAY = pd.Timestamp.now().normalize()

# ============================================================
# 2. LIGHT DESIGN TOKENS
# ============================================================
BG = "#F7F8FA"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F1F4F8"
BORDER = "#D9DEE7"
TEXT = "#172033"
TEXT_MUTED = "#687386"
GOLD = "#B8860B"
GOLD_SOFT = "#9A7300"
TEAL = "#177E89"
ROSE = "#B54A4A"
POSITIVE = "#2E8B57"
NEGATIVE = "#C94C4C"

SECTOR_PALETTE = [
    "#B8860B", "#177E89", "#5B6FB5", "#B54A4A", "#708238",
    "#9A6B3A", "#66728A", "#C66B4E", "#4D8D6E", "#8B72A8"
]

# ============================================================
# 3. GLOBAL LIGHT STYLE
# ============================================================
st.markdown(
    f"""
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

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}}

.hero-wrap {{
    padding: 28px 32px 24px 32px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F3F6FA 100%);
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(25, 38, 58, 0.04);
}}

.hero-eyebrow {{
    font-size: 12.5px;
    letter-spacing: 0.02em;
    color: {GOLD_SOFT};
    margin-bottom: 6px;
    font-weight: 600;
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
    color: {TEXT};
    line-height: 1;
}}

.hero-big-label {{
    font-size: 12.5px;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}

.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 18px;
    height: 100%;
    box-shadow: 0 2px 8px rgba(25, 38, 58, 0.03);
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

.persona-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
}}

.persona-card.active {{
    border: 1px solid {GOLD};
    background: #FFFDF5;
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

.info-box {{
    background: #F8FAFC;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px 14px;
    color: {TEXT_MUTED};
    font-size: 12.5px;
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

footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
</style>
""",
    unsafe_allow_html=True,
)

pio.templates["portfolio_light"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12.5),
        colorway=SECTOR_PALETTE,
        xaxis=dict(
            gridcolor="#E7EBF0",
            zerolinecolor="#D9DEE7",
            linecolor="#D9DEE7",
        ),
        yaxis=dict(
            gridcolor="#E7EBF0",
            zerolinecolor="#D9DEE7",
            linecolor="#D9DEE7",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "portfolio_light"

# ============================================================
# 4. MASTER PORTFOLIO CONFIGURATION
# ============================================================
# THREE DISTINCT DATES (never mix them up):
#   INVESTMENT_DATE  - fixed simulated purchase date.
#   latest_market_date (computed at run time) - the "current valuation date".
#   TARGET_DATE      - fixed end of the 3-year horizon.
INVESTMENT_DATE = pd.Timestamp("2026-08-31")
TARGET_DATE = pd.Timestamp("2029-08-31")
CORPUS = 1_00_00_000
HORIZON_YEARS = 3
DEFAULT_RF = 0.069
DEFAULT_RM = 0.130
assert TARGET_DATE == INVESTMENT_DATE + pd.DateOffset(years=HORIZON_YEARS), (
    "TARGET_DATE must equal INVESTMENT_DATE + HORIZON_YEARS"
)

# The ORIGINAL ₹1 Cr portfolio is established once, as of INVESTMENT_DATE, using
# these inception settings, and is then frozen in a snapshot file.
# Sidebar inputs (persona / lookback / Rf / Rm) only drive the CURRENT
# recommendation and modelled returns - never the original holdings.
INCEPTION_PERSONA = "Moderate"
INCEPTION_LOOKBACK_YRS = 5
INCEPTION_RF = DEFAULT_RF
INCEPTION_RM = DEFAULT_RM

# Data-quality thresholds
ENTRY_MAX_STALENESS_DAYS = 7     # entry price may be at most this many days before INVESTMENT_DATE
CURRENT_STALE_DAYS = 5           # current price older than this (vs latest market date) is flagged STALE
HISTORY_BUFFER_DAYS = 10

# G-Sec has no Yahoo price feed: it is carried at indicative par (100) and,
# optionally, accrues its coupon as simple interest (actual/365).
GSEC_TICKER = "GSEC2029"
GSEC_PAR_PRICE = 100.0
GSEC_COUPON_RATE = 0.0603        # taken from the instrument name "6.03% GOI G-Sec 2029"
GSEC_ACCRUE_COUPON = True
GSEC_STATUS = "Indicative par price (no market feed)"

STATUS_ENTITLED = "Entitled / return accrued"
STATUS_UPCOMING = "Declared / upcoming"

try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()
SNAPSHOT_PATH = os.path.join(APP_DIR, "inception_holdings.json")
AUTOSAVE_PATH = os.path.join(APP_DIR, ".inception_holdings_autosave.json")
SNAPSHOT_SCHEMA_VERSION = 1

# Persona target = objective only.
# It is displayed as the person's desired outcome and used to
# show required CAGR / target path. It is NOT used to solve weights.
PERSONAS = {
    "Aggressive": {
        "target_corpus": 2_25_00_000,
        "min_equity": 0.65, "max_equity": 0.90,
        "min_debt": 0.05, "max_debt": 0.20,
        "min_gold": 0.02, "max_gold": 0.10,
        "min_silver": 0.02, "max_silver": 0.10,
        "risk_label": "Highest target / highest risk budget",
    },
    "Moderate": {
        "target_corpus": 2_00_00_000,
        "min_equity": 0.50, "max_equity": 0.80,
        "min_debt": 0.10, "max_debt": 0.35,
        "min_gold": 0.05, "max_gold": 0.15,
        "min_silver": 0.02, "max_silver": 0.10,
        "risk_label": "Balanced target / medium risk budget",
    },
    "Conservative": {
        "target_corpus": 1_75_00_000,
        "min_equity": 0.35, "max_equity": 0.65,
        "min_debt": 0.20, "max_debt": 0.55,
        "min_gold": 0.05, "max_gold": 0.20,
        "min_silver": 0.02, "max_silver": 0.10,
        "risk_label": "Lower target / lower risk budget",
    },
}

for p_name, p_data in PERSONAS.items():
    p_data["target_cagr"] = (p_data["target_corpus"] / CORPUS) ** (1 / HORIZON_YEARS) - 1

# ============================================================
# LOCKED 15-INSTRUMENT UNIVERSE
# ============================================================
INSTRUMENTS_MASTER = [
    {"Ticker": "INDIANB.NS", "Name": "Indian Bank", "Sector": "Banking",
     "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "HAL.NS", "Name": "Hindustan Aeronautics (HAL)", "Sector": "Defence",
     "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "CHENNPETRO.NS", "Name": "Chennai Petroleum Corp", "Sector": "Oil & Gas",
     "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "MAZDOCK.NS", "Name": "Mazagon Dock Shipbuilders", "Sector": "Defence",
     "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "BSE.NS", "Name": "BSE Ltd.", "Sector": "Financial Services",
     "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "NATIONALUM.NS", "Name": "National Aluminium Co (NALCO)", "Sector": "Metals",
     "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "FORCEMOT.NS", "Name": "Force Motors", "Sector": "Automobiles",
     "CapCategory": "Small/Mid Cap", "Type": "Equity"},
    {"Ticker": "LLOYDSME.NS", "Name": "Lloyds Metals & Energy", "Sector": "Metals",
     "CapCategory": "Mid/Small Cap", "Type": "Equity"},
    {"Ticker": "APARINDS.NS", "Name": "Apar Industries", "Sector": "Electrical / Industrial",
     "CapCategory": "Mid Cap", "Type": "Equity"},
    {"Ticker": "OIL.NS", "Name": "Oil India", "Sector": "Oil & Gas",
     "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "TVSMOTOR.NS", "Name": "TVS Motor Company", "Sector": "Automobiles",
     "CapCategory": "Large Cap", "Type": "Equity"},
    {"Ticker": "GSEC2029", "Name": "6.03% GOI G-Sec 2029", "Sector": "Sovereign Debt",
     "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0625},
    {"Ticker": "EBBETF0431.NS", "Name": "BHARAT Bond ETF April 2031", "Sector": "Target Maturity Debt",
     "CapCategory": "Fixed Income", "Type": "Bond", "FixedYield": 0.0710},
    {"Ticker": "GOLDBEES.NS", "Name": "Nippon India ETF Gold BeES", "Sector": "Precious Metals",
     "CapCategory": "Gold ETF", "Type": "Gold ETF"},
    {"Ticker": "SILVERBEES.NS", "Name": "Nippon India Silver ETF", "Sector": "Precious Metals",
     "CapCategory": "Silver ETF", "Type": "Silver ETF"},
]
ALL_TICKERS = [item["Ticker"] for item in INSTRUMENTS_MASTER]
YAHOO_TICKERS = [t for t in ALL_TICKERS if t != GSEC_TICKER]
EQUITY_TICKERS = [item["Ticker"] for item in INSTRUMENTS_MASTER if item["Type"] == "Equity"]

# ============================================================
# 5. DATA ENGINE
# ============================================================
def _clean_index(df):
    """Timezone-naive, midnight-normalised, de-duplicated, sorted DatetimeIndex."""
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _extract_field(raw, field):
    """Pull one price field (e.g. 'Close', 'Adj Close') out of a yfinance frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            return raw[field]
        if field in raw.columns.get_level_values(1):
            return raw.xs(field, level=1, axis=1)
        return None
    return None


@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def fetch_price_history():
    """
    ONE daily download covering the inception lookback window up to the latest
    available trading day. Returns (close_df, adj_close_df):

      close_df     - raw (split-adjusted, NOT dividend-adjusted) closes. Used for
                     entry prices, current prices and portfolio valuation, so that
                     later dividends cannot retroactively change the 31 Aug entry
                     price (and dividends are not double counted).
      adj_close_df - dividend-adjusted closes. Used only for return statistics
                     (beta, volatility, historical CAGR).

    Exceptions are deliberately NOT swallowed here so Streamlit never caches a
    failed/empty download.
    """
    tickers = YAHOO_TICKERS + ["^NSEI"]
    start = (
        INVESTMENT_DATE
        - pd.DateOffset(years=INCEPTION_LOOKBACK_YRS)
        - pd.Timedelta(days=HISTORY_BUFFER_DAYS)
    )
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    close = _extract_field(raw, "Close")
    adj = _extract_field(raw, "Adj Close")
    if close is None:
        raise RuntimeError("Yahoo Finance response did not contain 'Close' prices.")
    if adj is None:
        adj = close.copy()

    close = _clean_index(close).dropna(how="all")
    adj = _clean_index(adj).dropna(how="all")
    return close, adj


@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def fetch_dividend_series(ticker):
    """Per-share dividend history (point-in-time facts). Raises on failure so a
    failed call is never cached."""
    d = yf.Ticker(ticker).dividends
    if d is None:
        raise RuntimeError(f"No dividend series returned for {ticker}.")
    d = d.copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    return d.astype(float)


def get_dividend_map():
    """ticker -> dividend Series, or None when the retrieval failed."""
    out = {}
    for t in EQUITY_TICKERS:
        try:
            out[t] = fetch_dividend_series(t)
        except Exception:
            out[t] = None
    return out


@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def _fetch_market_cap_cr(ticker):
    info = yf.Ticker(ticker).info or {}
    mc = info.get("marketCap")
    if mc is None:
        raise RuntimeError(f"No market cap for {ticker}.")
    return float(mc) / 1e7  # ₹ crore


def get_market_caps():
    """Display-only (never used in any calculation)."""
    out = {}
    for t in EQUITY_TICKERS:
        try:
            out[t] = _fetch_market_cap_cr(t)
        except Exception:
            out[t] = np.nan
    return out


def asof_series(series, as_of_date):
    s = series.dropna().copy()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    s = s.loc[s.index <= as_of_date].sort_index()
    return s


def compute_beta(asset_returns, benchmark_returns):
    combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < 20:
        return 1.0
    mkt_var = np.var(combined.iloc[:, 1], ddof=1)
    if mkt_var <= 0:
        return 1.0
    return float(np.cov(combined.iloc[:, 0], combined.iloc[:, 1], ddof=1)[0, 1] / mkt_var)


def compute_trailing_dividend_yield(div_series, price, as_of):
    """Trailing-12-month dividends / price, using dividend facts dated on or
    before `as_of` only (point-in-time, so it cannot drift between reruns)."""
    if div_series is None or price is None:
        return 0.0
    if not np.isfinite(price) or price <= 0:
        return 0.0
    as_of = pd.Timestamp(as_of)
    window = div_series[
        (div_series.index > as_of - pd.DateOffset(years=1)) & (div_series.index <= as_of)
    ]
    y = float(window.sum()) / float(price)
    return float(np.clip(y, 0.0, 0.15))


def select_entry_prices(close_df, investment_date, tickers):
    """
    For every instrument pick the LAST valid close on or before the investment
    date, record that date, and verify it. Never falls back to a current price.
    """
    rows = []
    for t in tickers:
        if t == GSEC_TICKER:
            rows.append({
                "Ticker": t, "EntryPrice": GSEC_PAR_PRICE,
                "EntryPriceDate": investment_date, "EntryStatus": GSEC_STATUS,
            })
            continue
        s = asof_series(close_df[t], investment_date) if t in close_df.columns else pd.Series(dtype=float)
        if s.empty:
            rows.append({"Ticker": t, "EntryPrice": np.nan,
                         "EntryPriceDate": pd.NaT, "EntryStatus": "MISSING (no price on or before investment date)"})
            continue
        d = pd.Timestamp(s.index[-1])
        p = float(s.iloc[-1])
        if (not np.isfinite(p)) or p <= 0 or d > investment_date:
            status = "INVALID"
        else:
            age = (investment_date - d).days
            status = "OK" if age <= ENTRY_MAX_STALENESS_DAYS else f"STALE ({age} days before investment date)"
        rows.append({"Ticker": t, "EntryPrice": p, "EntryPriceDate": d, "EntryStatus": status})
    return pd.DataFrame(rows)


def add_return_decomposition(df):
    df = df.copy()
    df["CapitalGainExpected"] = np.where(
        df["Type"].eq("Equity"),
        df["CAPM_Return"],
        np.where(df["Type"].isin(["Gold ETF", "Silver ETF"]), df["Actual_CAGR"], 0.0),
    )
    df["ExpectedIncomeYield"] = np.where(
        df["Type"].eq("Equity"),
        df["DivYield"],
        np.where(df["Type"].eq("Bond"), df["Total_Expected_Return"], 0.0),
    )
    return df


def analyze_instruments(close_df, adj_df, div_map, as_of, lookback_yrs, rf_rate, mkt_return):
    """
    Risk/return statistics for the 15 instruments using ONLY data dated on or
    before `as_of`, over a window of `lookback_yrs` ending at `as_of`.

      as_of = INVESTMENT_DATE      -> inception analysis (frozen into the snapshot)
      as_of = latest market date   -> current analysis (recommendation / projection)
    """
    as_of = pd.Timestamp(as_of)
    window_start = as_of - pd.DateOffset(years=int(lookback_yrs))
    adj_win = adj_df.loc[(adj_df.index >= window_start) & (adj_df.index <= as_of)]
    weekly_prices = adj_win.resample("W-FRI").last().dropna(how="all")
    weekly_returns = weekly_prices.pct_change(fill_method=None).dropna(how="all")
    nifty_returns = (
        weekly_returns["^NSEI"].dropna()
        if "^NSEI" in weekly_returns.columns else pd.Series(dtype=float)
    )

    records = []
    for item in INSTRUMENTS_MASTER:
        t = item["Ticker"]
        rec = item.copy()

        if t == GSEC_TICKER:
            rec.update({
                "Actual_CAGR": item["FixedYield"], "Beta": 0.0, "Volatility": 0.0,
                "DivYield": 0.0, "CAPM_Return": item["FixedYield"],
                "Total_Expected_Return": item["FixedYield"], "AnalysisOK": True,
            })
            records.append(rec)
            continue

        px_adj = weekly_prices[t].dropna() if t in weekly_prices.columns else pd.Series(dtype=float)
        raw_px = asof_series(close_df[t], as_of) if t in close_df.columns else pd.Series(dtype=float)

        if px_adj.empty or raw_px.empty:
            rec.update({
                "Actual_CAGR": 0.0, "Beta": 1.0, "Volatility": 0.0, "DivYield": 0.0,
                "CAPM_Return": 0.0, "Total_Expected_Return": 0.0, "AnalysisOK": False,
            })
            records.append(rec)
            continue

        latest_adj = float(px_adj.iloc[-1])
        latest_raw = float(raw_px.iloc[-1])

        if len(px_adj) >= 2:
            elapsed_years = max((px_adj.index[-1] - px_adj.index[0]).days / 365.25, 1 / 52)
            actual_cagr = (latest_adj / float(px_adj.iloc[0])) ** (1 / elapsed_years) - 1
        else:
            actual_cagr = 0.0

        if t in weekly_returns.columns and not nifty_returns.empty:
            beta = compute_beta(weekly_returns[t].dropna(), nifty_returns)
            vol = float(weekly_returns[t].dropna().std() * np.sqrt(52))
        else:
            beta = 1.0
            vol = 0.0

        if item["Type"] == "Equity":
            div_yield = compute_trailing_dividend_yield(div_map.get(t), latest_raw, as_of)
            capm_return = rf_rate + beta * (mkt_return - rf_rate)
            total_expected = capm_return + div_yield
        elif item["Type"] == "Bond":
            div_yield = 0.0
            capm_return = item.get("FixedYield", 0.0)
            total_expected = capm_return
        else:
            # Gold/silver: realised price CAGR is the modelling anchor.
            div_yield = 0.0
            capm_return = actual_cagr
            total_expected = actual_cagr

        rec.update({
            "Actual_CAGR": actual_cagr, "Beta": beta, "Volatility": vol,
            "DivYield": div_yield, "CAPM_Return": capm_return,
            "Total_Expected_Return": total_expected, "AnalysisOK": True,
        })
        records.append(rec)

    return add_return_decomposition(pd.DataFrame(records)), weekly_returns


def _annualized_risk_vector(out, returns_df):
    """Annualized risk proxy for the 15 instruments."""
    risks = []
    for _, row in out.iterrows():
        ticker = row["Ticker"]
        if ticker == GSEC_TICKER:
            risks.append(0.03)
        elif ticker in returns_df.columns:
            s = returns_df[ticker].dropna()
            risks.append(float(s.std() * np.sqrt(52)) if len(s) >= 10 else 0.12)
        else:
            risks.append(0.12)
    return np.clip(np.asarray(risks, dtype=float), 0.02, 1.50)


def build_persona_weights(df_analysis, persona_name, returns_df):
    """
    Optimizer (UNCHANGED LOGIC). Returns a RECOMMENDED weight set only.
    It no longer computes units or values: units belong to the locked
    inception snapshot and are never touched by the optimizer.

    The investor's target corpus is converted into a required CAGR.
    The optimizer then:
      - requires portfolio expected return >= required CAGR;
      - minimizes a risk/concentration objective;
      - respects persona-specific equity/debt/gold/silver guardrails.

    If the target is mathematically infeasible under the current assumptions,
    the maximum-achievable-return portfolio is returned instead.
    """
    cfg = PERSONAS[persona_name]
    out = df_analysis.copy().reset_index(drop=True)

    expected = pd.to_numeric(
        out["Total_Expected_Return"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    risks = _annualized_risk_vector(out, returns_df)
    types = out["Type"].to_numpy()

    eq_idx = np.where(types == "Equity")[0]
    bond_idx = np.where(types == "Bond")[0]
    gold_idx = np.where(types == "Gold ETF")[0]
    silver_idx = np.where(types == "Silver ETF")[0]

    target_cagr = float(cfg["target_cagr"])

    def objective(w):
        risk_term = np.sum((w * risks) ** 2)
        concentration_term = 0.015 * np.sum(w ** 2)
        return float(risk_term + concentration_term)

    common_constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},

        {"type": "ineq", "fun": lambda w: np.sum(w[eq_idx]) - cfg["min_equity"]},
        {"type": "ineq", "fun": lambda w: cfg["max_equity"] - np.sum(w[eq_idx])},

        {"type": "ineq", "fun": lambda w: np.sum(w[bond_idx]) - cfg["min_debt"]},
        {"type": "ineq", "fun": lambda w: cfg["max_debt"] - np.sum(w[bond_idx])},

        {"type": "ineq", "fun": lambda w: np.sum(w[gold_idx]) - cfg["min_gold"]},
        {"type": "ineq", "fun": lambda w: cfg["max_gold"] - np.sum(w[gold_idx])},

        {"type": "ineq", "fun": lambda w: np.sum(w[silver_idx]) - cfg["min_silver"]},
        {"type": "ineq", "fun": lambda w: cfg["max_silver"] - np.sum(w[silver_idx])},
    ]

    bounds = []
    for typ in types:
        if typ == "Equity":
            bounds.append((0.02, 0.22))
        elif typ == "Bond":
            bounds.append((0.02, 0.30))
        elif typ == "Gold ETF":
            bounds.append((cfg["min_gold"], cfg["max_gold"]))
        else:
            bounds.append((cfg["min_silver"], cfg["max_silver"]))

    w0 = np.zeros(len(out), dtype=float)
    w0[eq_idx] = cfg["min_equity"] / len(eq_idx)
    w0[bond_idx] = cfg["min_debt"] / len(bond_idx)
    w0[gold_idx] = cfg["min_gold"] / len(gold_idx)
    w0[silver_idx] = cfg["min_silver"] / len(silver_idx)

    remainder = 1.0 - w0.sum()
    if remainder > 0:
        w0[eq_idx] += remainder / len(eq_idx)

    max_return_res = minimize(
        lambda w: -float(np.dot(w, expected)),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=common_constraints,
        options={"maxiter": 1500, "ftol": 1e-10},
    )
    if not max_return_res.success:
        raise RuntimeError(
            f"Could not establish the feasible portfolio set: {max_return_res.message}"
        )

    max_achievable_return = float(np.dot(max_return_res.x, expected))
    target_feasible = max_achievable_return >= target_cagr - 1e-6

    if target_feasible:
        target_constraints = common_constraints + [
            {"type": "ineq", "fun": lambda w: float(np.dot(w, expected)) - target_cagr}
        ]
        result = minimize(
            objective,
            max_return_res.x,
            method="SLSQP",
            bounds=bounds,
            constraints=target_constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Target-constrained optimization failed: {result.message}")
        weights = np.asarray(result.x, dtype=float)
        solver_status = "Target is feasible; weights minimize risk while maintaining the required CAGR."
    else:
        weights = np.asarray(max_return_res.x, dtype=float)
        solver_status = "Target is not feasible; weights maximize expected return within the current risk guardrails."

    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()

    out["Portfolio_Weight"] = weights
    out["Allocated_Amount"] = out["Portfolio_Weight"] * CORPUS

    achieved_expected_return = float(np.dot(weights, expected))
    projected_value = float(CORPUS * (1.0 + achieved_expected_return) ** HORIZON_YEARS)

    meta = {
        "target_feasible": target_feasible,
        "required_cagr": target_cagr,
        "achieved_expected_return": achieved_expected_return,
        "projected_value": projected_value,
        "target_corpus": cfg["target_corpus"],
        "target_gap_cagr": achieved_expected_return - target_cagr,
        "target_gap_value": projected_value - cfg["target_corpus"],
        "max_achievable_return": max_achievable_return,
        "solver_status": solver_status,
    }
    return out, meta


# ============================================================
# 5b. INCEPTION SNAPSHOT (the ORIGINAL ₹1 Cr investment)
# ============================================================
def validate_snapshot(snap):
    """Return a list of problems (empty list = snapshot is valid)."""
    problems = []
    try:
        if snap.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            problems.append(f"Unsupported schema_version {snap.get('schema_version')!r}.")
        if pd.Timestamp(snap["investment_date"]) != INVESTMENT_DATE:
            problems.append(
                f"Snapshot investment_date {snap['investment_date']} != {INVESTMENT_DATE.date()}."
            )
        if abs(float(snap["corpus"]) - CORPUS) > 0.5:
            problems.append(f"Snapshot corpus {snap['corpus']} != {CORPUS}.")

        rows = snap["holdings"]
        tickers = [r["Ticker"] for r in rows]
        if len(tickers) != len(set(tickers)):
            problems.append("Duplicate tickers in snapshot.")
        if set(tickers) != set(ALL_TICKERS):
            problems.append(
                "Snapshot instruments do not match the 15-instrument universe "
                f"(missing: {sorted(set(ALL_TICKERS) - set(tickers))}, "
                f"unexpected: {sorted(set(tickers) - set(ALL_TICKERS))})."
            )

        total_w, total_alloc = 0.0, 0.0
        for r in rows:
            t = r["Ticker"]
            w = float(r["Portfolio_Weight"])
            alloc = float(r["Allocated_Amount"])
            price = float(r["EntryPrice"])
            units = float(r["Units"])
            d = pd.Timestamp(r["EntryPriceDate"])
            if not (np.isfinite(w) and w >= 0):
                problems.append(f"{t}: invalid weight {w}.")
            if not (np.isfinite(price) and price > 0):
                problems.append(f"{t}: invalid entry price {price}.")
            if not (np.isfinite(units) and units > 0):
                problems.append(f"{t}: invalid units {units}.")
            if d > INVESTMENT_DATE:
                problems.append(f"{t}: entry price date {d.date()} is after the investment date.")
            if abs(w * CORPUS - alloc) > 1.0:
                problems.append(f"{t}: allocation does not equal weight × corpus.")
            if abs(units * price - alloc) > max(1.0, 1e-6 * abs(alloc)):
                problems.append(f"{t}: units × entry price does not reconcile to the allocation.")
            total_w += w
            total_alloc += alloc
        if abs(total_w - 1.0) > 1e-6:
            problems.append(f"Weights sum to {total_w:.8f}, not 1.")
        if abs(total_alloc - CORPUS) > 1.0:
            problems.append(f"Allocations sum to ₹{total_alloc:,.2f}, not ₹{CORPUS:,}.")
    except Exception as exc:
        problems.append(f"Snapshot is malformed: {exc}")
    return problems


def create_inception_snapshot(close_df, adj_df, div_map):
    """
    Establish the original portfolio from data dated on/before INVESTMENT_DATE.
    Returns (snapshot_dict, []) or (None, [problems]). Nothing is created unless
    EVERY instrument has a verified entry price and a complete analysis.
    """
    problems = []
    entry_df = select_entry_prices(close_df, INVESTMENT_DATE, ALL_TICKERS)
    for _, r in entry_df.iterrows():
        if r["EntryStatus"] not in ("OK", GSEC_STATUS):
            problems.append(f"{r['Ticker']}: entry price {r['EntryStatus']}.")

    df_an, wk = analyze_instruments(
        close_df, adj_df, div_map, INVESTMENT_DATE,
        INCEPTION_LOOKBACK_YRS, INCEPTION_RF, INCEPTION_RM,
    )
    for _, r in df_an.iterrows():
        if not bool(r["AnalysisOK"]):
            problems.append(f"{r['Ticker']}: no usable price history as of {INVESTMENT_DATE.date()}.")
    for t in EQUITY_TICKERS:
        if div_map.get(t) is None:
            problems.append(f"{t}: dividend history could not be retrieved.")
    if problems:
        return None, problems

    try:
        reco, meta = build_persona_weights(df_an, INCEPTION_PERSONA, wk)
    except Exception as exc:
        return None, [f"Optimizer failed while establishing the original portfolio: {exc}"]

    merged = entry_df.merge(
        reco[["Ticker", "Portfolio_Weight", "Allocated_Amount"]], on="Ticker", how="left"
    )
    merged["Units"] = merged["Allocated_Amount"] / merged["EntryPrice"]
    name_map = {i["Ticker"]: i["Name"] for i in INSTRUMENTS_MASTER}
    type_map = {i["Ticker"]: i["Type"] for i in INSTRUMENTS_MASTER}

    holdings = []
    for _, r in merged.iterrows():
        holdings.append({
            "Ticker": r["Ticker"],
            "Name": name_map[r["Ticker"]],
            "Type": type_map[r["Ticker"]],
            "Portfolio_Weight": float(r["Portfolio_Weight"]),
            "Allocated_Amount": float(r["Allocated_Amount"]),
            "EntryPrice": float(r["EntryPrice"]),
            "EntryPriceDate": pd.Timestamp(r["EntryPriceDate"]).date().isoformat(),
            "EntryStatus": str(r["EntryStatus"]),
            "Units": float(r["Units"]),
        })

    snap = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "investment_date": INVESTMENT_DATE.date().isoformat(),
        "corpus": CORPUS,
        "persona": INCEPTION_PERSONA,
        "lookback_yrs": INCEPTION_LOOKBACK_YRS,
        "rf_rate": INCEPTION_RF,
        "mkt_return": INCEPTION_RM,
        "optimizer_status": meta["solver_status"],
        "holdings": holdings,
    }
    snap_problems = validate_snapshot(snap)
    if snap_problems:
        return None, snap_problems
    return snap, []


def snapshot_to_df(snap):
    h = pd.DataFrame(snap["holdings"])
    h["EntryPriceDate"] = pd.to_datetime(h["EntryPriceDate"])
    meta = pd.DataFrame(INSTRUMENTS_MASTER)[["Ticker", "Name", "Sector", "CapCategory", "Type"]]
    return meta.merge(
        h[["Ticker", "Portfolio_Weight", "Allocated_Amount", "EntryPrice",
           "EntryPriceDate", "EntryStatus", "Units"]],
        on="Ticker", how="left", validate="one_to_one",
    )


def load_inception_snapshot():
    """
    Returns (snapshot | None, kind, problems).
      kind 'repo'     - inception_holdings.json committed next to this file (permanent)
      kind 'autosave' - best-effort local copy (lost when the Streamlit container restarts)
    An existing-but-invalid file is reported, never silently replaced.
    """
    for path, kind in ((SNAPSHOT_PATH, "repo"), (AUTOSAVE_PATH, "autosave")):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    snap = json.load(fh)
            except Exception as exc:
                return None, kind, [f"Could not read {os.path.basename(path)}: {exc}"]
            return snap, kind, validate_snapshot(snap)
    return None, None, []


def save_snapshot_file(snap, path):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2)
        return True
    except Exception:
        return False


# ============================================================
# 5c. CURRENT VALUATION
# ============================================================
def mark_to_market(df, close_df, as_of_date):
    """
    Value the ORIGINAL units at the latest valid price on or before `as_of_date`.
    Records the price date and a status per instrument. Missing prices are
    left as NaN and flagged - they are never replaced with entry/other prices.
    """
    out = df.copy()
    cur_p, cur_d, prev_p, status = [], [], [], []
    for _, r in out.iterrows():
        t = r["Ticker"]
        if t == GSEC_TICKER:
            cur_p.append(GSEC_PAR_PRICE)
            cur_d.append(pd.Timestamp(as_of_date))
            prev_p.append(GSEC_PAR_PRICE)
            status.append(GSEC_STATUS)
            continue
        s = asof_series(close_df[t], as_of_date) if t in close_df.columns else pd.Series(dtype=float)
        if s.empty:
            cur_p.append(np.nan); cur_d.append(pd.NaT); prev_p.append(np.nan); status.append("MISSING")
            continue
        p = float(s.iloc[-1])
        d = pd.Timestamp(s.index[-1])
        if (not np.isfinite(p)) or p <= 0:
            cur_p.append(np.nan); cur_d.append(d); prev_p.append(np.nan); status.append("INVALID")
            continue
        lag = (pd.Timestamp(as_of_date) - d).days
        cur_p.append(p)
        cur_d.append(d)
        prev_p.append(float(s.iloc[-2]) if len(s) >= 2 else p)
        status.append("OK" if lag <= CURRENT_STALE_DAYS else f"STALE ({lag} days old)")

    out["CurrentPrice"] = cur_p
    out["PriceDate"] = pd.to_datetime(pd.Series(cur_d, index=out.index))
    out["PreviousClose"] = prev_p
    out["PriceStatus"] = status

    out["CurrentValue"] = out["Units"] * out["CurrentPrice"]
    out["DailyPriceChange"] = out["CurrentPrice"] - out["PreviousClose"]
    out["DailyPriceChangePct"] = np.where(
        out["PreviousClose"].notna() & (out["PreviousClose"] != 0),
        out["DailyPriceChange"] / out["PreviousClose"],
        0.0,
    )

    days_held = max((pd.Timestamp(as_of_date) - INVESTMENT_DATE).days, 0)
    if GSEC_ACCRUE_COUPON:
        out["AccruedInterest"] = np.where(
            out["Ticker"].eq(GSEC_TICKER),
            out["Allocated_Amount"] * GSEC_COUPON_RATE * days_held / 365.0,
            0.0,
        )
    else:
        out["AccruedInterest"] = 0.0
    return out


def valuation_issues(df):
    missing = df.loc[df["PriceStatus"].isin(["MISSING", "INVALID"]), "Name"].tolist()
    stale = df.loc[df["PriceStatus"].astype(str).str.startswith("STALE"), "Name"].tolist()
    return missing, stale


# ============================================================
# 5d. DIVIDENDS (NSE primary, Yahoo fallback)
# ============================================================
def _nse_session():
    """Create a browser-like NSE session for official public API calls."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=15)
    except Exception:
        pass
    return session


def _parse_dividend_amount(purpose):
    """Extract dividend per share from NSE's purpose text."""
    text = str(purpose or "")
    patterns = [
        r"Dividend\s*-\s*Rs\.?\s*([0-9]+(?:\.[0-9]+)?)\s*Per Share",
        r"Interim Dividend\s*-\s*Rs\.?\s*([0-9]+(?:\.[0-9]+)?)\s*Per Share",
        r"Dividend\s*-\s*Re\.?\s*([0-9]+(?:\.[0-9]+)?)\s*Per Share",
        r"Interim Dividend\s*-\s*Re\.?\s*([0-9]+(?:\.[0-9]+)?)\s*Per Share",
        r"Dividend[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*Per Share",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return float(m.group(1))
    return np.nan


def _normalize_nse_date(value):
    return pd.to_datetime(value, errors="coerce", dayfirst=False)


DIVIDEND_COLUMNS = [
    "Symbol", "NSE Purpose", "Dividend / Share", "Ex-Date", "Record Date",
    "NSE Announcement Date", "NSE Announcement Subject", "Source",
]
DIVIDEND_DATE_COLUMNS = ["Ex-Date", "Record Date", "NSE Announcement Date"]


@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def fetch_nse_dividends(symbols, start_date, end_date):
    """
    PRIMARY dividend feed: NSE public corporate-actions + corporate-announcements APIs.

    Only events with an ex-date STRICTLY AFTER start_date (the investment date)
    are kept: shares bought at the investment-date close are not entitled to a
    dividend whose ex-date is that same day.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    session = _nse_session()

    rows = []

    for symbol in symbols:
        actions_url = (
            "https://www.nseindia.com/api/corporates-corporateActions"
            f"?index=equities&symbol={symbol}"
        )
        actions = []
        try:
            r = session.get(actions_url, timeout=15)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                actions = payload.get("data", payload.get("corporateActions", []))
            elif isinstance(payload, list):
                actions = payload
        except Exception:
            actions = []

        announcements_url = (
            "https://www.nseindia.com/api/corporate-announcements"
            f"?index=equities&symbol={symbol}"
        )
        announcements = []
        try:
            r = session.get(announcements_url, timeout=15)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                announcements = payload.get("data", [])
            elif isinstance(payload, list):
                announcements = payload
        except Exception:
            announcements = []

        dividend_actions = []
        for action in actions:
            if not isinstance(action, dict):
                continue

            purpose = action.get("subject", action.get("purpose", ""))
            if "DIVIDEND" not in str(purpose).upper():
                continue

            ex_date = _normalize_nse_date(action.get("exDate", action.get("ex_date")))
            record_date = _normalize_nse_date(
                action.get("recDate", action.get("recordDate", action.get("record_date")))
            )
            amount = _parse_dividend_amount(purpose)

            if pd.isna(ex_date) and pd.isna(record_date):
                continue

            action_date = ex_date if pd.notna(ex_date) else record_date
            # strictly after the investment date; not beyond the fetch horizon
            if pd.notna(action_date) and (action_date <= start_date or action_date > end_date):
                continue

            dividend_actions.append({
                "Symbol": symbol,
                "NSE Purpose": purpose,
                "Dividend / Share": amount,
                "Ex-Date": ex_date,
                "Record Date": record_date,
                "Source": "NSE Corporate Actions API",
            })

        for row in dividend_actions:
            relevant_ann = []
            for ann in announcements:
                if not isinstance(ann, dict):
                    continue
                subject = str(ann.get("desc", ann.get("subject", "")))
                if "DIVIDEND" not in subject.upper():
                    continue

                ann_date = _normalize_nse_date(
                    ann.get("an_dt", ann.get("sort_date", ann.get("broadcastDate")))
                )
                if pd.notna(ann_date):
                    if start_date - pd.Timedelta(days=30) <= ann_date <= end_date:
                        relevant_ann.append((ann_date, subject))

            if relevant_ann:
                ex_date = row["Ex-Date"]
                if pd.notna(ex_date):
                    prior = [x for x in relevant_ann if x[0] <= ex_date]
                    chosen = max(prior, key=lambda x: x[0]) if prior else min(
                        relevant_ann, key=lambda x: abs((x[0] - ex_date).days)
                    )
                else:
                    chosen = max(relevant_ann, key=lambda x: x[0])
                row["NSE Announcement Date"] = chosen[0]
                row["NSE Announcement Subject"] = chosen[1]
            else:
                row["NSE Announcement Date"] = pd.NaT
                row["NSE Announcement Subject"] = ""

        rows.extend(dividend_actions)

    if not rows:
        return pd.DataFrame(columns=DIVIDEND_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Symbol", "Ex-Date", "Dividend / Share"], keep="last")
    return df.sort_values(["Symbol", "Ex-Date"], ascending=[True, False])


@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def fetch_yahoo_dividends_fallback(symbols, start_date, end_date):
    """Fallback only for symbols where NSE does not return an action."""
    rows = []
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    for symbol in symbols:
        ticker = symbol + ".NS"
        try:
            div = yf.Ticker(ticker).dividends
            div.index = pd.to_datetime(div.index)
            if getattr(div.index, "tz", None) is not None:
                div.index = div.index.tz_localize(None)
            div.index = div.index.normalize()
            # strictly after the investment date
            div = div[(div.index > start_date) & (div.index <= end_date)]
        except Exception:
            div = pd.Series(dtype=float)

        for event_date, amount in div.items():
            rows.append({
                "Symbol": symbol,
                "NSE Purpose": "Yahoo Finance dividend action",
                "Dividend / Share": float(amount),
                "Ex-Date": pd.Timestamp(event_date),
                "Record Date": pd.NaT,
                "NSE Announcement Date": pd.NaT,
                "NSE Announcement Subject": "",
                "Source": "Yahoo Finance fallback",
            })

    if not rows:
        return pd.DataFrame(columns=DIVIDEND_COLUMNS)
    return pd.DataFrame(rows)


def build_dividend_tracker(units_by_ticker, investment_date, fetch_end, as_of):
    """
    One consolidated, auditable dividend ledger.

    Events are attributed using the ORIGINAL units. An event counts toward
    portfolio return only when its ex-date is after the investment date and on
    or before `as_of` (the current valuation date); later ex-dates are listed
    as "Declared / upcoming" and are NOT added to value.
    """
    symbols = [t.replace(".NS", "") for t in EQUITY_TICKERS]

    nse_df = fetch_nse_dividends(symbols, investment_date, fetch_end)

    covered = set(nse_df["Symbol"].dropna().astype(str)) if not nse_df.empty else set()
    missing_symbols = [s for s in symbols if s not in covered]
    yahoo_df = fetch_yahoo_dividends_fallback(missing_symbols, investment_date, fetch_end)

    if nse_df.empty:
        tracker = yahoo_df.copy()
    elif yahoo_df.empty:
        tracker = nse_df.copy()
    else:
        tracker = pd.concat([nse_df, yahoo_df], ignore_index=True)

    for col in DIVIDEND_COLUMNS:
        if col not in tracker.columns:
            if col == "Dividend / Share":
                tracker[col] = np.nan
            elif col in DIVIDEND_DATE_COLUMNS:
                tracker[col] = pd.NaT
            else:
                tracker[col] = ""

    ticker_by_symbol = {t.replace(".NS", ""): t for t in ALL_TICKERS}
    name_by_ticker = {i["Ticker"]: i["Name"] for i in INSTRUMENTS_MASTER}
    tracker["Ticker"] = tracker["Symbol"].map(ticker_by_symbol)
    tracker["Company"] = tracker["Ticker"].map(name_by_ticker)
    tracker["Units"] = tracker["Ticker"].map(units_by_ticker)
    tracker["Dividend Entitlement"] = (
        pd.to_numeric(tracker["Dividend / Share"], errors="coerce")
        * pd.to_numeric(tracker["Units"], errors="coerce")
    )

    ex = pd.to_datetime(tracker["Ex-Date"], errors="coerce")
    tracker["Status"] = np.where(
        ex.notna() & (ex > pd.Timestamp(investment_date)) & (ex <= pd.Timestamp(as_of)),
        STATUS_ENTITLED,
        STATUS_UPCOMING,
    )
    tracker["NSE Verified"] = np.where(
        tracker["Source"].astype(str).str.startswith("NSE"), "Yes", "Fallback",
    )

    return tracker.sort_values(
        ["Ex-Date", "Company"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


# ============================================================
# 6. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ◆ Portfolio Terminal")
    st.caption("15-instrument live allocation model")
    st.divider()

    with st.expander("Investor Objective", expanded=True):
        selected_persona = st.selectbox(
            "Persona target",
            list(PERSONAS.keys()),
            index=1,
            help="The target is the investor's desired 3-year outcome. It drives the target comparison and the CURRENT recommendation; it never changes the original 31 Aug 2026 holdings.",
        )
        lookback_yrs = st.slider(
            "Historical lookback (years)",
            min_value=1,
            max_value=5,
            value=5,
            help="Used for the CURRENT beta/volatility/return analysis and recommendation. The original holdings were set with a fixed 5-year lookback ending 31 Aug 2026.",
        )

    with st.expander("CAPM Parameters", expanded=True):
        rf_rate = st.number_input(
            "Risk-free rate (Rf)",
            value=DEFAULT_RF,
            step=0.001,
            format="%.3f",
        )
        mkt_return = st.number_input(
            "Market return (Rm)",
            value=DEFAULT_RM,
            step=0.005,
            format="%.3f",
        )

    st.divider()
    if st.button("↻ Refresh live market data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    refresh_note = (
        "Daily auto-refresh enabled; NSE corporate actions are checked each day."
        if AUTO_REFRESH_AVAILABLE
        else "Daily auto-refresh unavailable in this environment; use the refresh button. "
             "Market-data cache still expires daily."
    )
    st.caption("Investment date: 31 Aug 2026 · " + refresh_note)
    st.caption("Dividend source priority: NSE Corporate Actions → Yahoo fallback.")
    st.caption("Original holdings are locked at inception. Persona, lookback, Rf and Rm only update the current recommendation and modelled returns.")

# ============================================================
# 7. RUN DATA ENGINE
# ============================================================
# 7a. Load the locked inception snapshot (if one exists).
snapshot, snapshot_kind, snapshot_problems = load_inception_snapshot()
if snapshot_problems:
    st.error(
        "The inception snapshot file failed validation. The dashboard has stopped "
        "instead of silently replacing your original holdings."
    )
    for _problem in snapshot_problems:
        st.write("- " + _problem)
    st.caption(
        "Fix the file, or delete it deliberately if you really want the original "
        "portfolio to be rebuilt from 31 Aug 2026 data."
    )
    st.stop()

# 7b. Market data.
try:
    with st.spinner("Downloading market data & computing risk/return metrics..."):
        close_df, adj_df = fetch_price_history()
        div_map = get_dividend_map()
        market_caps = get_market_caps()
        _holding_cols = [c for c in close_df.columns if c != "^NSEI"]
        if not _holding_cols:
            raise RuntimeError("No instrument prices were returned.")
        latest_market_date = pd.Timestamp(close_df[_holding_cols].dropna(how="all").index.max())
except Exception as e:
    st.error("Error downloading market data.")
    st.exception(e)
    st.stop()

if latest_market_date < INVESTMENT_DATE:
    st.error(
        f"Market data currently ends on {latest_market_date.strftime('%d %b %Y')}, before the "
        f"investment date ({INVESTMENT_DATE.strftime('%d %b %Y')}). The simulated investment "
        f"cannot be established yet."
    )
    st.stop()

# 7c. First run only: establish the original portfolio from data dated on or
# before 31 Aug 2026. Nothing is created unless every instrument has a verified
# entry price and complete analysis.
if snapshot is None:
    snapshot, _new_problems = create_inception_snapshot(close_df, adj_df, div_map)
    if _new_problems:
        st.error(
            "The original 31 Aug 2026 portfolio could not be established safely, so nothing "
            "was created. Fix the issue below (or press Refresh) and reload."
        )
        for _problem in _new_problems:
            st.write("- " + _problem)
        st.stop()
    snapshot_kind = "autosave" if save_snapshot_file(snapshot, AUTOSAVE_PATH) else "provisional"

# 7d. Current analysis + valuation of the ORIGINAL units.
try:
    holdings_df = snapshot_to_df(snapshot)
    df_cur, returns_df = analyze_instruments(
        close_df, adj_df, div_map, latest_market_date, lookback_yrs, rf_rate, mkt_return,
    )
    _analysis_cols = [
        "Ticker", "Beta", "Volatility", "Actual_CAGR", "DivYield", "CAPM_Return",
        "Total_Expected_Return", "CapitalGainExpected", "ExpectedIncomeYield", "AnalysisOK",
    ]
    df_master = holdings_df.merge(
        df_cur[_analysis_cols], on="Ticker", how="left", validate="one_to_one"
    )
    df_master["MarketCapCr"] = df_master["Ticker"].map(market_caps)
    df_master = mark_to_market(df_master, close_df, latest_market_date)
except Exception as e:
    st.error("Error executing market analysis engine.")
    st.exception(e)
    st.stop()

data_health_df = pd.DataFrame({
    "Instrument": df_master["Name"],
    "Entry price date": df_master["EntryPriceDate"].dt.strftime("%d %b %Y"),
    "Entry price (₹)": df_master["EntryPrice"],
    "Current price date": df_master["PriceDate"].dt.strftime("%d %b %Y"),
    "Current price (₹)": df_master["CurrentPrice"],
    "Entry price status": df_master["EntryStatus"],
    "Current price status": df_master["PriceStatus"],
})

missing_prices, stale_prices = valuation_issues(df_master)
if missing_prices:
    st.error(
        "Valuation stopped: no valid current price for " + ", ".join(missing_prices) + ". "
        "Showing a total without these holdings would misstate the portfolio value and P&L, "
        "and substituting the entry price would hide real gains/losses. "
        "Press Refresh, or check the ticker on Yahoo Finance."
    )
    st.dataframe(data_health_df, use_container_width=True, hide_index=True)
    st.stop()
if stale_prices:
    st.warning(
        "These holdings have a last traded price older than "
        f"{CURRENT_STALE_DAYS} days and are valued at that last price: "
        + ", ".join(stale_prices) + "."
    )

persona_cfg = PERSONAS[selected_persona]
target_cagr = persona_cfg["target_cagr"]

# Current recommendation (separate from the original holdings; never used for valuation).
reco_df, optimization_meta, reco_error = None, None, ""
try:
    reco_df, optimization_meta = build_persona_weights(df_cur, selected_persona, returns_df)
except Exception as e:
    reco_error = str(e)

# ============================================================
# 8. DIVIDENDS + PORTFOLIO VALUE
# ============================================================
units_by_ticker = df_master.set_index("Ticker")["Units"].to_dict()
dividend_fetch_end = max(TODAY, latest_market_date) + pd.Timedelta(days=90)
dividend_error = ""
try:
    div_hist = build_dividend_tracker(
        units_by_ticker, INVESTMENT_DATE, dividend_fetch_end, latest_market_date,
    )
except Exception as e:
    dividend_error = str(e)
    div_hist = pd.DataFrame(columns=DIVIDEND_COLUMNS + [
        "Ticker", "Company", "Units", "Dividend Entitlement", "Status", "NSE Verified",
    ])

_counted = div_hist["Status"].eq(STATUS_ENTITLED)
dividend_entitlement = float(
    pd.to_numeric(div_hist.loc[_counted, "Dividend Entitlement"], errors="coerce").fillna(0).sum()
)
unparsed_dividends = int(
    pd.to_numeric(div_hist["Dividend / Share"], errors="coerce").isna().sum()
)

df_master["DividendEntitlement"] = 0.0
if _counted.any():
    dividend_by_ticker = div_hist.loc[_counted].groupby("Ticker")["Dividend Entitlement"].sum()
    df_master["DividendEntitlement"] = (
        df_master["Ticker"].map(dividend_by_ticker).fillna(0.0)
    )

# Total value = ORIGINAL units x latest valid price + dividends whose ex-date has
# passed + modelled G-Sec accrued coupon.
df_master["TotalCurrentValue"] = (
    df_master["CurrentValue"] + df_master["DividendEntitlement"] + df_master["AccruedInterest"]
)
df_master["UnrealizedPnL"] = df_master["CurrentValue"] - df_master["Allocated_Amount"]
df_master["TotalInvestmentGain"] = df_master["TotalCurrentValue"] - df_master["Allocated_Amount"]
df_master["TotalInvestmentGainPct"] = np.where(
    df_master["Allocated_Amount"] != 0,
    df_master["TotalInvestmentGain"] / df_master["Allocated_Amount"],
    0.0,
)
df_master["CurrentWeight"] = (
    df_master["TotalCurrentValue"] / df_master["TotalCurrentValue"].sum()
)

# ============================================================
# 9. PORTFOLIO METRICS
# ============================================================
current_portfolio_value = float(df_master["TotalCurrentValue"].sum())
invested_capital = float(CORPUS)
current_pnl = current_portfolio_value - invested_capital
current_pnl_pct = current_pnl / invested_capital

# ------------------------------------------------------------
# DATES: investment date (fixed) / current valuation date (latest market data)
# / target date (fixed) / remaining horizon
# ------------------------------------------------------------
target_date = TARGET_DATE
remaining_days = max((TARGET_DATE - latest_market_date).days, 0)
remaining_years = remaining_days / 365.25
elapsed_years_now = min(
    max((latest_market_date - INVESTMENT_DATE).days / 365.25, 0.0), float(HORIZON_YEARS)
)

# Modelled return of the ACTUAL holdings: current market-value weights x current
# expected returns (current beta / CAPM / dividend yield).
portfolio_expected_return = float(
    (df_master["CurrentWeight"] * df_master["Total_Expected_Return"]).sum()
)

if remaining_years > 0 and current_portfolio_value > 0:
    live_required_cagr = (
        (persona_cfg["target_corpus"] / current_portfolio_value)
        ** (1.0 / remaining_years)
        - 1.0
    )
else:
    live_required_cagr = (
        0.0 if current_portfolio_value >= persona_cfg["target_corpus"] else np.inf
    )

live_expected_end_value = (
    current_portfolio_value
    * (1.0 + portfolio_expected_return) ** remaining_years
    if remaining_years > 0
    else current_portfolio_value
)

live_projected_gain = live_expected_end_value - current_portfolio_value
live_projected_gain_pct = (
    live_projected_gain / current_portfolio_value
    if current_portfolio_value > 0 else 0.0
)
live_target_surplus_deficit = live_expected_end_value - persona_cfg["target_corpus"]

live_target_achievable = (
    current_portfolio_value >= persona_cfg["target_corpus"]
    or (
        np.isfinite(live_required_cagr)
        and portfolio_expected_return >= live_required_cagr
    )
)
portfolio_realized_cagr = float(
    (df_master["Portfolio_Weight"] * df_master["Actual_CAGR"]).sum()
)

# Return components (current market-value weights, so they add up to the
# modelled return shown in the KPI row).
equity_mask = df_master["Type"].eq("Equity")
bond_mask = df_master["Type"].eq("Bond")
gold_mask = df_master["Type"].eq("Gold ETF")
silver_mask = df_master["Type"].eq("Silver ETF")

equity_cap_gain = float(
    (df_master.loc[equity_mask, "CurrentWeight"] * df_master.loc[equity_mask, "CapitalGainExpected"]).sum()
)
equity_dividends = float(
    (df_master.loc[equity_mask, "CurrentWeight"] * df_master.loc[equity_mask, "DivYield"]).sum()
)
bond_income = float(
    (df_master.loc[bond_mask, "CurrentWeight"] * df_master.loc[bond_mask, "Total_Expected_Return"]).sum()
)
gold_return = float(
    (df_master.loc[gold_mask, "CurrentWeight"] * df_master.loc[gold_mask, "Total_Expected_Return"]).sum()
)
silver_return = float(
    (df_master.loc[silver_mask, "CurrentWeight"] * df_master.loc[silver_mask, "Total_Expected_Return"]).sum()
)

yrs = np.arange(0, HORIZON_YEARS + 1)
target_path = CORPUS * ((1 + target_cagr) ** yrs)
expected_final_value = (
    optimization_meta["projected_value"] if optimization_meta else np.nan
)

return_breakdown = pd.DataFrame({
    "Component": [
        "Equity capital gains",
        "Equity dividends",
        "Bond interest",
        "Gold appreciation",
        "Silver appreciation",
    ],
    "Contribution": [
        equity_cap_gain,
        equity_dividends,
        bond_income,
        gold_return,
        silver_return,
    ],
})

# ============================================================
# 9. HERO
# ============================================================
hero_l, hero_r = st.columns([2.3, 1])

with hero_l:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-eyebrow">LIVE PORTFOLIO · AS OF {latest_market_date.strftime("%d %b %Y").upper()} · INVESTED {INVESTMENT_DATE.strftime("%d %b %Y").upper()} · {lookback_yrs}Y LOOKBACK</div>
            <div class="hero-title">15-Asset Allocation Terminal</div>
            <p class="hero-sub">
                ₹1.00 Cr invested on {INVESTMENT_DATE.strftime("%d %b %Y")} across 11 equities, 2 bonds, Gold ETF and Silver ETF
                (original holdings set under the <strong style="color:{GOLD_SOFT}">{snapshot.get("persona", INCEPTION_PERSONA)}</strong> profile and locked).
                Target comparisons use the selected <strong style="color:{GOLD_SOFT}">{selected_persona}</strong> objective;
                the optimizer only produces a separate, current recommendation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_r:
    st.markdown(
        f"""
        <div class="hero-wrap" style="text-align:right; height:100%;">
            <div class="hero-big-number">₹{current_portfolio_value/1e7:.2f} Cr</div>
            <div class="hero-big-label">Current portfolio value at {latest_market_date.strftime("%d %b %Y")}</div>
            <div class="{"kpi-delta-pos" if current_pnl >= 0 else "kpi-delta-neg"}"
                 style="margin-top:10px; font-size:13px;">
                P&amp;L since {INVESTMENT_DATE.strftime("%d %b %Y")}: ₹{current_pnl/1e5:.2f} Lakh ({current_pnl_pct:+.2%})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Snapshot status (is the original portfolio permanent yet?)
_snapshot_created = str(snapshot.get("created_at_utc", ""))[:10]
if snapshot_kind == "repo":
    st.caption(
        f"Original holdings locked from inception_holdings.json (created {_snapshot_created}; "
        f"{snapshot.get('persona', INCEPTION_PERSONA)} profile, {snapshot.get('lookback_yrs', INCEPTION_LOOKBACK_YRS)}Y lookback). "
        f"Sidebar inputs do not change them."
    )
else:
    st.warning(
        "Original holdings were established from 31 Aug 2026 data but are NOT permanent yet: "
        "this server's disk is wiped whenever the app restarts or redeploys. Download the file "
        "below and commit it to your GitHub repo root (next to Investment_Dashboard.py) as "
        "inception_holdings.json to lock them permanently."
    )
    st.download_button(
        "⬇ Download inception_holdings.json",
        data=json.dumps(snapshot, indent=2),
        file_name="inception_holdings.json",
        mime="application/json",
    )

# ============================================================
# 10. KPI ROW
# ============================================================
def kpi_card(label, value, delta_html=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(
    kpi_card("Current Portfolio Value", f"₹{current_portfolio_value/1e7:.2f} Cr"),
    unsafe_allow_html=True,
)
k2.markdown(
    kpi_card("Investor Target", f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr"),
    unsafe_allow_html=True,
)
k3.markdown(
    kpi_card(
        "CAGR Needed From Today",
        f"{live_required_cagr:.2%}" if np.isfinite(live_required_cagr) else "N/A",
        f'<div class="{ "kpi-delta-pos" if live_target_achievable else "kpi-delta-neg" }">'
        f'{"On track" if live_target_achievable else "At risk"}</div>',
    ),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi_card(
        "Current Modelled Return",
        f"{portfolio_expected_return:.2%}",
        '<div class="kpi-delta-pos">Original holdings · current market/risk data</div>',
    ),
    unsafe_allow_html=True,
)
k5.markdown(
    kpi_card(
        "Projected Value at Target Date",
        f"₹{live_expected_end_value/1e7:.2f} Cr",
        f'<div class="{ "kpi-delta-pos" if live_target_surplus_deficit >= 0 else "kpi-delta-neg" }">'
        f'{"₹{:,.0f} above".format(live_target_surplus_deficit) if live_target_surplus_deficit >= 0 else "₹{:,.0f} below".format(abs(live_target_surplus_deficit))} target</div>',
    ),
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    f'<div class="info-box"><strong>Investment date</strong> {INVESTMENT_DATE.strftime("%d %b %Y")} (fixed) · '
    f'<strong>Current valuation date</strong> {latest_market_date.strftime("%d %b %Y")} (latest market data) · '
    f'<strong>Target date</strong> {TARGET_DATE.strftime("%d %b %Y")} (fixed) · '
    f'<strong>Remaining horizon</strong> {remaining_years:.2f} years ({remaining_days} days). '
    f'Prices can be intraday while the market is open.</div>',
    unsafe_allow_html=True,
)
st.write("")

# Prominent "actual amount" projection card.
proj_l, proj_r = st.columns([1.5, 1])
with proj_l:
    st.markdown(
        f"""
        <div class="hero-wrap" style="margin-top:6px;">
            <div class="hero-eyebrow">LIVE 3-YEAR PROJECTION</div>
            <div class="hero-big-number">₹{live_expected_end_value/1e7:.2f} Cr</div>
            <div class="hero-big-label">
                Estimated portfolio value on {target_date.strftime("%d %b %Y")}
                if today's portfolio value and modelled annual return continue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with proj_r:
    st.markdown(
        f"""
        <div class="hero-wrap" style="margin-top:6px;">
            <div class="hero-eyebrow">PROJECTED GAIN BY TARGET DATE</div>
            <div class="hero-big-number">₹{live_projected_gain/1e5:.2f} L</div>
            <div class="hero-big-label">
                Projected gain of {live_projected_gain_pct:.2%} from today's
                ₹{current_portfolio_value/1e7:.2f} Cr over the remaining horizon.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Real reconciliation: ORIGINAL units x ENTRY prices must equal the ₹1 Cr invested.
entry_value_check = float((df_master["Units"] * df_master["EntryPrice"]).sum())
reconciliation_gap = entry_value_check - CORPUS
st.caption(
    f"Inception reconciliation: units × entry prices = ₹{entry_value_check:,.0f} vs ₹{CORPUS:,} invested "
    f"(difference ₹{reconciliation_gap:,.2f}) across {len(df_master)} instruments. "
    f"Units are fixed from the {INVESTMENT_DATE.strftime('%d %b %Y')} entry prices; "
    f"market value ₹{df_master['CurrentValue'].sum():,.0f} + dividends ₹{dividend_entitlement:,.0f} "
    f"+ G-Sec accrued coupon ₹{df_master['AccruedInterest'].sum():,.0f} = ₹{current_portfolio_value:,.0f}."
)
if abs(reconciliation_gap) > 1.0:
    st.error("Inception units × entry prices do not reconcile to ₹1 Cr. Check the snapshot file.")

with st.expander("Data integrity: entry prices & current prices per instrument"):
    st.dataframe(
        data_health_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Current price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )
    st.caption(
        "Entry price = last close on or before the investment date (never a current price). "
        "The G-Sec has no market price feed and is carried at indicative par (₹100)"
        + (f", accruing its {GSEC_COUPON_RATE:.2%} coupon as simple interest." if GSEC_ACCRUE_COUPON else ".")
    )
    if dividend_error:
        st.warning(f"Dividend ledger could not be built ({dividend_error}); dividends are NOT included in the value.")

st.write("")

if optimization_meta is None:
    st.info(
        "The current recommendation could not be computed "
        f"({reco_error}). Portfolio valuation above is unaffected."
    )
elif optimization_meta["target_feasible"]:
    st.success(
        f"Current recommendation: the {selected_persona} objective of "
        f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr is feasible under the current "
        f"return assumptions and risk guardrails. A fresh optimization would minimize risk while "
        f"maintaining at least {target_cagr:.2%} expected annual return (see the Recommendation tab)."
    )
else:
    st.warning(
        f"Current recommendation: the {selected_persona} target of ₹{persona_cfg['target_corpus']/1e7:.2f} Cr "
        f"is not feasible from inception under the current assumptions and risk guardrails. "
        f"A fresh optimization would show a maximum achievable outcome of "
        f"₹{expected_final_value/1e7:.2f} Cr."
    )

if live_target_achievable:
    st.success(
        f"Live target check: current portfolio value is ₹{current_portfolio_value/1e7:.2f} Cr. "
        f"With {remaining_years:.2f} years remaining until {target_date.strftime('%d %b %Y')}, "
        f"the portfolio needs {live_required_cagr:.2%} CAGR from today to reach the fixed "
        f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr target. "
        f"Current modelled return is {portfolio_expected_return:.2%}; the target is currently on track."
    )
else:
    st.warning(
        f"Live target check: current portfolio value is ₹{current_portfolio_value/1e7:.2f} Cr. "
        f"With {remaining_years:.2f} years remaining until {target_date.strftime('%d %b %Y')}, "
        f"the portfolio needs {live_required_cagr:.2%} CAGR from today to reach the fixed "
        f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr target, "
        f"versus a current modelled return of {portfolio_expected_return:.2%}. "
        f"The target is currently at risk."
    )

# ============================================================
# 11. PERSONA STRIP
# ============================================================
p1, p2, p3 = st.columns(3)
for col, (pname, pdata) in zip([p1, p2, p3], PERSONAS.items()):
    active = " active" if pname == selected_persona else ""
    col.markdown(
        f"""
        <div class="persona-card{active}">
            <div class="persona-name">{pname.upper()}</div>
            <div class="persona-target">₹{pdata['target_corpus']/1e7:.2f} Cr</div>
            <div class="persona-cagr">
                target CAGR {pdata['target_cagr']:.2%} ·
                equity {pdata['min_equity']:.0%}–{pdata['max_equity']:.0%} ·
                debt {pdata['min_debt']:.0%}–{pdata['max_debt']:.0%} ·
                gold {pdata['min_gold']:.0%}–{pdata['max_gold']:.0%} ·
                silver {pdata['min_silver']:.0%}–{pdata['max_silver']:.0%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# 12. INTERACTIVE INVESTMENT SELECTOR
# ============================================================
st.markdown('<div class="section-title">Investment Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Select an instrument to update the detailed view, metrics and charts.</div>',
    unsafe_allow_html=True,
)

instrument_options = df_master["Name"].tolist()
selected_instrument = st.selectbox(
    "Choose an investment",
    instrument_options,
    index=0,
)
selected_row = df_master.loc[df_master["Name"].eq(selected_instrument)].iloc[0]

i1, i2, i3, i4 = st.columns(4)
i1.metric("Current Price", f"₹{selected_row['CurrentPrice']:,.2f}" if pd.notna(selected_row["CurrentPrice"]) else "N/A")
i2.metric("Original Weight", f"{selected_row['Portfolio_Weight']:.2%}")
i3.metric("Allocated", f"₹{selected_row['Allocated_Amount']/1e5:.2f} L")
i4.metric("Current Value", f"₹{selected_row['TotalCurrentValue']/1e5:.2f} L" if pd.notna(selected_row["TotalCurrentValue"]) else "N/A")

detail_left, detail_right = st.columns([1, 1.4])

with detail_left:
    details = {
        "Asset Class": selected_row["Type"],
        "Market Cap": (
            f"₹{selected_row['MarketCapCr']:,.0f} Cr"
            if pd.notna(selected_row["MarketCapCr"]) else "N/A"
        ),
        "Sector / Exposure": selected_row["Sector"],
        "Entry Price": f"₹{selected_row['EntryPrice']:,.2f}" if pd.notna(selected_row["EntryPrice"]) else "N/A",
        "Entry Price Date": str(pd.Timestamp(selected_row["EntryPriceDate"]).date()) if pd.notna(selected_row["EntryPriceDate"]) else "N/A",
        "Cap Category": selected_row["CapCategory"],
        "Current Price Date": str(pd.Timestamp(selected_row["PriceDate"]).date()) if pd.notna(selected_row["PriceDate"]) else "N/A",
        "Units (fixed)": f"{selected_row['Units']:,.4f}" if pd.notna(selected_row["Units"]) else "N/A",
        "Daily Price Change": f"{selected_row['DailyPriceChangePct']:+.2%}",
        "Dividend Earned / Entitled": f"₹{selected_row['DividendEntitlement']:,.0f}",
        "Accrued Coupon (modelled)": f"₹{selected_row['AccruedInterest']:,.0f}",
        "Beta": f"{selected_row['Beta']:.2f}",
        "Volatility": f"{selected_row['Volatility']:.2%}",
        "Historical CAGR": f"{selected_row['Actual_CAGR']:.2%}",
        "Expected Capital Gain": f"{selected_row['CapitalGainExpected']:.2%}",
        "Dividend / Interest Yield": f"{selected_row['ExpectedIncomeYield']:.2%}",
        "Expected Total Return": f"{selected_row['Total_Expected_Return']:.2%}",
    }
    detail_df = pd.DataFrame(details.items(), columns=["Metric", "Value"])
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

with detail_right:
    if selected_row["Ticker"] != GSEC_TICKER and selected_row["Ticker"] in close_df.columns:
        _series = close_df[selected_row["Ticker"]].dropna()
        _series = _series[_series.index >= latest_market_date - pd.DateOffset(years=int(lookback_yrs))]
        if not _series.empty:
            fig_selected = go.Figure()
            fig_selected.add_trace(
                go.Scatter(
                    x=_series.index,
                    y=_series.values,
                    mode="lines",
                    name=selected_instrument,
                    line=dict(color=GOLD, width=2.5),
                )
            )
            if pd.notna(selected_row["EntryPrice"]) and pd.notna(selected_row["EntryPriceDate"]):
                fig_selected.add_trace(
                    go.Scatter(
                        x=[selected_row["EntryPriceDate"]],
                        y=[selected_row["EntryPrice"]],
                        mode="markers",
                        name="Entry",
                        marker=dict(size=11, color=TEAL),
                    )
                )
            fig_selected.update_layout(
                height=330,
                title=f"{selected_instrument} — price history",
                yaxis_title="Price (₹)",
                xaxis_title="",
            )
            st.plotly_chart(fig_selected, use_container_width=True)
    else:
        st.markdown(
            '<div class="info-box">The sovereign G-Sec is represented using its indicative par price / yield assumption rather than a Yahoo Finance price series.</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ============================================================
# 13. MAIN TABS
# ============================================================
tab_overview, tab_holdings, tab_reco, tab_risk, tab_history = st.tabs(
    ["Overview", "Holdings", "Recommendation", "Risk & Return", "Historical Trends"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown('<div class="section-title">Asset-Class Allocation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">How the original ₹1 Cr was divided across equity, debt, gold and silver</div>', unsafe_allow_html=True)

        pie_df = pd.DataFrame([
            {"Sleeve": "Equity (11)", "Amount": df_master.loc[equity_mask, "Allocated_Amount"].sum()},
            {"Sleeve": "Debt (2)", "Amount": df_master.loc[bond_mask, "Allocated_Amount"].sum()},
            {"Sleeve": "Gold", "Amount": df_master.loc[gold_mask, "Allocated_Amount"].sum()},
            {"Sleeve": "Silver", "Amount": df_master.loc[silver_mask, "Allocated_Amount"].sum()},
        ])

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=pie_df["Sleeve"],
                    values=pie_df["Amount"],
                    hole=0.58,
                    marker=dict(
                        colors=[GOLD, TEAL, "#A6784A", "#8E99A8"],
                        line=dict(color=SURFACE, width=2),
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=12.5, color=TEXT),
                )
            ]
        )
        fig_pie.update_layout(
            height=340,
            showlegend=False,
            annotations=[
                dict(
                    text=f"₹{current_portfolio_value/1e7:.2f} Cr",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16, family="Fraunces, serif", color=TEXT),
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Target vs Expected Trajectory</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-note">Current portfolio: ₹{current_portfolio_value/1e7:.2f} Cr · projected value at {target_date.strftime("%d %b %Y")}: ₹{live_expected_end_value/1e7:.2f} Cr · fixed target: ₹{persona_cfg["target_corpus"]/1e7:.2f} Cr.</div>',
            unsafe_allow_html=True,
        )

        fig_growth = go.Figure()
        live_elapsed = np.linspace(0, max(remaining_years, 0.0), HORIZON_YEARS + 1)
        live_projection_path = (
            current_portfolio_value * (1.0 + portfolio_expected_return) ** live_elapsed
        )

        fig_growth.add_trace(
            go.Scatter(
                x=[elapsed_years_now + x for x in live_elapsed],
                y=live_projection_path,
                mode="lines+markers",
                name=f"Live Projection ({portfolio_expected_return:.2%})",
                line=dict(color=GOLD, width=3),
            )
        )
        fig_growth.add_trace(
            go.Scatter(
                x=yrs,
                y=target_path,
                mode="lines",
                name=f"Target Required ({target_cagr:.2%})",
                line=dict(color=TEXT_MUTED, dash="dash"),
            )
        )
        fig_growth.add_trace(
            go.Scatter(
                x=[elapsed_years_now],
                y=[current_portfolio_value],
                mode="markers",
                name="Current Portfolio Value",
                marker=dict(size=12, color=TEAL),
            )
        )

        fig_growth.update_layout(
            height=340,
            xaxis_title="Years elapsed since 31 Aug 2026",
            yaxis_title="Portfolio value (₹)",
            yaxis_tickprefix="₹",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown('<div class="section-title">Return Composition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Annualised weighted contribution (current market-value weights) from capital gains and income sources.</div>',
        unsafe_allow_html=True,
    )

    rb_plot = return_breakdown.copy()
    fig_return = px.bar(
        rb_plot,
        x="Contribution",
        y="Component",
        orientation="h",
        text=rb_plot["Contribution"].map(lambda x: f"{x:.2%}"),
    )
    fig_return.update_traces(marker_color=TEAL, textposition="outside")
    fig_return.update_layout(
        height=310,
        xaxis_tickformat=".1%",
        xaxis_title="Weighted annual return contribution",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_return, use_container_width=True)

    st.markdown(
        '<div class="info-box">The investment is assumed to be initiated on 31 Aug 2026. Holdings (units) are fixed at their entry prices and never change; each refresh re-marks those holdings to the latest market price and adds dividends whose ex-date has passed.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Daily Market Movement</div>', unsafe_allow_html=True)

    daily_move = df_master[
        ["Name", "MarketCapCr", "CurrentPrice", "DailyPriceChange", "DailyPriceChangePct"]
    ].copy()

    daily_move["Market Cap (₹ Cr)"] = daily_move["MarketCapCr"]
    # Streamlit's printf-style "%.2f%%" prints the raw number, so convert the
    # decimal change to percentage points first (0.0125 -> 1.25).
    daily_move["DailyPriceChangePct"] = daily_move["DailyPriceChangePct"] * 100.0
    daily_move = daily_move[
        ["Name", "Market Cap (₹ Cr)", "CurrentPrice", "DailyPriceChange", "DailyPriceChangePct"]
    ]
    daily_move = daily_move.sort_values("DailyPriceChangePct", ascending=False)

    st.dataframe(
        daily_move,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Market Cap (₹ Cr)": st.column_config.NumberColumn(
                "Market Cap (₹ Cr)", format="₹%,.0f"
            ),
            "CurrentPrice": st.column_config.NumberColumn(
                "Current Price", format="₹%.2f"
            ),
            "DailyPriceChange": st.column_config.NumberColumn(
                "₹ Change", format="₹%.2f"
            ),
            "DailyPriceChangePct": st.column_config.NumberColumn(
                "Daily Change", format="%.2f%%"
            ),
        },
    )

    st.markdown(
        f'<div class="section-title">Dividend Tracker · ₹{dividend_entitlement:,.0f} earned/entitled</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-note">Dividend events with an ex-date after the 31 Aug 2026 investment date, valued on the original units. Only events whose ex-date has passed are counted in portfolio value; later ones are listed as upcoming. NSE announcement/broadcast date is shown separately when the public NSE announcements page exposes a dividend-related filing; no announcement date is invented when unavailable.</div>',
        unsafe_allow_html=True,
    )
    required_display_cols = [
        "Company", "Dividend / Share", "NSE Announcement Date",
        "NSE Announcement Subject", "Ex-Date", "Record Date",
        "Dividend Entitlement", "Status", "NSE Verified", "Source"
    ]
    dividend_view = div_hist.reindex(columns=required_display_cols).copy()

    if unparsed_dividends:
        st.warning(
            f"{unparsed_dividends} dividend event(s) have no parsable amount per share and are "
            f"NOT included in the entitlement total."
        )

    if not dividend_view.empty:
        for _date_col in ["NSE Announcement Date", "Ex-Date", "Record Date"]:
            dividend_view[_date_col] = pd.to_datetime(
                dividend_view[_date_col], errors="coerce"
            )

        st.dataframe(
            dividend_view.sort_values(
                "Ex-Date", ascending=False, na_position="last"
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Dividend / Share": st.column_config.NumberColumn(format="₹%.2f"),
                "NSE Announcement Date": st.column_config.DateColumn(
                    "NSE Announcement / Broadcast", format="DD MMM YYYY"
                ),
                "Ex-Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                "Record Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                "Dividend Entitlement": st.column_config.NumberColumn(
                    "Dividend Earned / Entitlement", format="₹%d"
                ),
            },
        )

        st.caption(
            f"Dividend entitlement included in portfolio total return: "
            f"₹{dividend_entitlement:,.0f}. NSE is the primary corporate-action source; "
            f"Yahoo Finance is used only when NSE returns no event for a selected stock."
        )
    else:
        st.info(
            "No dividend event has been recorded since the 31 Aug 2026 investment date. "
            "The official NSE corporate-action feed will be checked again on the next refresh."
        )

# ---------- TAB 2: HOLDINGS ----------
with tab_holdings:
    st.markdown('<div class="section-title">All 15 Instruments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Original holdings: entry price, fixed units, current value, expected return and allocation.</div>',
        unsafe_allow_html=True,
    )

    display_df = df_master.copy()
    display_df["Entry Price"] = display_df["EntryPrice"]
    display_df["Current Price"] = display_df["CurrentPrice"]
    display_df["Dividend Earned"] = display_df["DividendEntitlement"]
    display_df["Accrued Interest"] = display_df["AccruedInterest"]
    display_df["Total Gain"] = display_df["TotalInvestmentGain"]

    # Streamlit's printf-style "%.2f%%" prints the raw number, so every decimal
    # percentage must be converted to percentage points first (0.125 -> 12.5).
    display_df["Daily Change %"] = display_df["DailyPriceChangePct"] * 100.0
    display_df["Historical CAGR"] = display_df["Actual_CAGR"] * 100.0
    display_df["Dividend / Interest"] = display_df["ExpectedIncomeYield"] * 100.0
    display_df["Capital Gain"] = display_df["CapitalGainExpected"] * 100.0
    display_df["Expected Total Return"] = display_df["Total_Expected_Return"] * 100.0
    display_df["Portfolio Weight %"] = display_df["Portfolio_Weight"] * 100.0
    display_df["Current Weight %"] = display_df["CurrentWeight"] * 100.0

    display_df["Market Cap (₹ Cr)"] = display_df["MarketCapCr"]

    cols = [
        "Name", "Market Cap (₹ Cr)", "Type", "Sector", "CapCategory", "Entry Price",
        "Current Price", "Units", "Portfolio Weight %", "Current Weight %",
        "Allocated_Amount", "CurrentValue", "Dividend Earned", "Accrued Interest",
        "Total Gain", "Daily Change %", "Historical CAGR", "Capital Gain",
        "Dividend / Interest", "Expected Total Return", "Beta",
    ]

    total_weight = float(df_master["Portfolio_Weight"].sum())
    asset_weights = df_master.groupby("Type")["Portfolio_Weight"].sum().to_dict()

    rw1, rw2, rw3, rw4, rw5 = st.columns(5)
    rw1.metric("Total Weight", f"{total_weight:.2%}")
    rw2.metric("Equity", f"{asset_weights.get('Equity', 0.0):.2%}")
    rw3.metric("Debt", f"{asset_weights.get('Bond', 0.0):.2%}")
    rw4.metric("Gold", f"{asset_weights.get('Gold ETF', 0.0):.2%}")
    rw5.metric("Silver", f"{asset_weights.get('Silver ETF', 0.0):.2%}")

    if abs(total_weight - 1.0) > 1e-8:
        st.error("Original portfolio weights do not reconcile to 100%. Check the snapshot file.")

    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Market Cap (₹ Cr)": st.column_config.NumberColumn(
                "Market Cap (₹ Cr)", format="₹%,.0f"
            ),
            "Entry Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Units": st.column_config.NumberColumn(format="%.4f"),
            "Portfolio Weight %": st.column_config.ProgressColumn(
                "Original Weight",
                format="%.1f%%",
                min_value=0,
                max_value=max(float(display_df["Portfolio Weight %"].max()) * 1.15, 10.0),
            ),
            "Current Weight %": st.column_config.NumberColumn(format="%.1f%%"),
            "Allocated_Amount": st.column_config.NumberColumn("Invested", format="₹%d"),
            "CurrentValue": st.column_config.NumberColumn("Current Value", format="₹%d"),
            "Dividend Earned": st.column_config.NumberColumn(format="₹%d"),
            "Accrued Interest": st.column_config.NumberColumn(format="₹%d"),
            "Total Gain": st.column_config.NumberColumn(format="₹%d"),
            "Daily Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "Historical CAGR": st.column_config.NumberColumn(format="%.2f%%"),
            "Capital Gain": st.column_config.NumberColumn(format="%.2f%%"),
            "Dividend / Interest": st.column_config.NumberColumn(format="%.2f%%"),
            "Expected Total Return": st.column_config.NumberColumn(format="%.2f%%"),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.write("")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown('<div class="section-title">Original Weight by Instrument</div>', unsafe_allow_html=True)
        sorted_df = df_master.sort_values("Portfolio_Weight", ascending=True)
        fig_bar = px.bar(
            sorted_df,
            x="Portfolio_Weight",
            y="Name",
            orientation="h",
            color="Type",
            color_discrete_map={
                "Equity": GOLD,
                "Bond": TEAL,
                "Gold ETF": "#A6784A",
                "Silver ETF": "#8F99A8",
            },
        )
        fig_bar.update_layout(
            height=480,
            xaxis_tickformat=".1%",
            yaxis_title="",
            xaxis_title="Portfolio weight",
            legend_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with h2:
        st.markdown('<div class="section-title">Current Value by Asset Class</div>', unsafe_allow_html=True)
        asset_value_df = pd.DataFrame([
            {"Asset Class": "Equity", "Current Value": df_master.loc[equity_mask, "TotalCurrentValue"].sum()},
            {"Asset Class": "Bond", "Current Value": df_master.loc[bond_mask, "TotalCurrentValue"].sum()},
            {"Asset Class": "Gold", "Current Value": df_master.loc[gold_mask, "TotalCurrentValue"].sum()},
            {"Asset Class": "Silver", "Current Value": df_master.loc[silver_mask, "TotalCurrentValue"].sum()},
        ])
        fig_asset = px.bar(
            asset_value_df,
            x="Current Value",
            y="Asset Class",
            orientation="h",
            text=asset_value_df["Current Value"].map(lambda x: f"₹{x/1e5:.1f} L"),
        )
        fig_asset.update_traces(marker_color=TEAL, textposition="outside")
        fig_asset.update_layout(
            height=480,
            xaxis_title="Current value (₹)",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig_asset, use_container_width=True)

# ---------- TAB 3: RECOMMENDATION (separate from the original holdings) ----------
with tab_reco:
    st.markdown('<div class="section-title">Current Recommendation vs Original Holdings</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">The optimizer re-runs on current data (lookback {lookback_yrs}Y, Rf {rf_rate:.1%}, Rm {mkt_return:.1%}) for the {selected_persona} profile. '
        f'This is a suggestion only: it never changes your original units or the portfolio value above.</div>',
        unsafe_allow_html=True,
    )

    if reco_df is None:
        st.info(f"The current recommendation could not be computed: {reco_error}")
    else:
        cmp_df = df_master[["Ticker", "Name", "Type", "Portfolio_Weight", "CurrentWeight"]].merge(
            reco_df[["Ticker", "Portfolio_Weight"]].rename(columns={"Portfolio_Weight": "Recommended_Weight"}),
            on="Ticker", how="left", validate="one_to_one",
        )
        cmp_df["Difference"] = cmp_df["Recommended_Weight"] - cmp_df["CurrentWeight"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended portfolio expected return", f"{optimization_meta['achieved_expected_return']:.2%}")
        m2.metric("Original holdings expected return", f"{portfolio_expected_return:.2%}")
        m3.metric("Target status (from inception)", "Feasible" if optimization_meta["target_feasible"] else "Not feasible")
        st.caption(optimization_meta["solver_status"])

        cmp_show = pd.DataFrame({
            "Instrument": cmp_df["Name"],
            "Type": cmp_df["Type"],
            "Original weight %": cmp_df["Portfolio_Weight"] * 100.0,
            "Current weight %": cmp_df["CurrentWeight"] * 100.0,
            "Recommended weight %": cmp_df["Recommended_Weight"] * 100.0,
            "Recommended − current (pp)": cmp_df["Difference"] * 100.0,
        })
        st.dataframe(
            cmp_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Original weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Current weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Recommended weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Recommended − current (pp)": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        fig_reco = go.Figure()
        fig_reco.add_trace(go.Bar(
            x=cmp_df["Name"], y=cmp_df["Portfolio_Weight"],
            name="Original (locked)", marker_color=GOLD,
        ))
        fig_reco.add_trace(go.Bar(
            x=cmp_df["Name"], y=cmp_df["Recommended_Weight"],
            name="Current recommendation", marker_color=TEAL,
        ))
        fig_reco.update_layout(
            barmode="group", height=420, yaxis_tickformat=".0%",
            yaxis_title="Portfolio weight", xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_reco, use_container_width=True)

# ---------- TAB 4: RISK & RETURN ----------
with tab_risk:
    eq_df = df_master[df_master["Type"] == "Equity"].copy()

    st.markdown('<div class="section-title">Security Market Line — Equity Sleeve</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Beta is estimated from weekly returns against the Nifty 50 benchmark.</div>',
        unsafe_allow_html=True,
    )

    beta_max = max(float(eq_df["Beta"].max()) * 1.15, 1.5)
    beta_range = np.linspace(0, beta_max, 50)
    sml = rf_rate + beta_range * (mkt_return - rf_rate)

    fig_sml = go.Figure()
    fig_sml.add_trace(
        go.Scatter(
            x=beta_range,
            y=sml,
            mode="lines",
            name="Security Market Line",
            line=dict(color=TEXT_MUTED, dash="dash", width=1.5),
        )
    )
    eq_df["Position"] = np.where(
        eq_df["Actual_CAGR"] >= eq_df["CAPM_Return"],
        "Outperforming",
        "Underperforming",
    )
    fig_sml.add_trace(
        go.Scatter(
            x=eq_df["Beta"],
            y=eq_df["Actual_CAGR"],
            mode="markers+text",
            text=eq_df["Ticker"],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=eq_df["Portfolio_Weight"] * 420 + 10,
                color=np.where(
                    eq_df["Position"] == "Outperforming", POSITIVE, NEGATIVE
                ),
                line=dict(color=SURFACE, width=1),
            ),
            name="Historical CAGR",
            hovertext=eq_df["Name"],
        )
    )
    fig_sml.update_layout(
        height=440,
        xaxis_title="Beta (systematic risk vs. Nifty 50)",
        yaxis_title="Return",
        yaxis_tickformat=".0%",
        showlegend=True,
    )
    st.plotly_chart(fig_sml, use_container_width=True)
    st.caption("Bubble size = original portfolio weight. Green = historical CAGR above CAPM-implied return.")

    r1, r2 = st.columns(2)

    with r1:
        st.markdown('<div class="section-title">Beta by Stock</div>', unsafe_allow_html=True)
        beta_sorted = eq_df.sort_values("Beta")
        fig_beta = px.bar(
            beta_sorted,
            x="Beta",
            y="Ticker",
            orientation="h",
            color="Beta",
            color_continuous_scale=[TEAL, SURFACE_ALT, ROSE],
        )
        fig_beta.add_vline(x=1.0, line_dash="dot", line_color=TEXT_MUTED)
        fig_beta.update_layout(
            height=420,
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_beta, use_container_width=True)

    with r2:
        st.markdown('<div class="section-title">Equity Correlation Matrix</div>', unsafe_allow_html=True)
        eq_tickers = [t for t in eq_df["Ticker"] if t in returns_df.columns]
        if len(eq_tickers) >= 2:
            corr = returns_df[eq_tickers].corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale=[TEAL, "#FFFFFF", GOLD],
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig_corr.update_layout(height=420, coloraxis_colorbar=dict(title=""))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough overlapping return history to build the correlation matrix.")

# ---------- TAB 5: HISTORICAL TRENDS ----------
with tab_history:
    st.markdown('<div class="section-title">Normalized Price Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Selected series (dividend-adjusted closes) indexed to 100 at the start of the available lookback window.</div>',
        unsafe_allow_html=True,
    )

    _hist_start = latest_market_date - pd.DateOffset(years=int(lookback_yrs))
    hist_prices = adj_df.loc[adj_df.index >= _hist_start]

    all_names = {
        row["Ticker"]: row["Name"]
        for _, row in df_master.iterrows()
        if row["Ticker"] in hist_prices.columns
    }
    default_pick = [
        t
        for t in df_master[df_master["Type"] == "Equity"]["Ticker"].head(5)
        if t in hist_prices.columns
    ]

    picked = st.multiselect(
        "Instruments to plot",
        options=list(all_names.keys()) + (
            ["^NSEI"] if "^NSEI" in hist_prices.columns else []
        ),
        default=default_pick + (
            ["^NSEI"] if "^NSEI" in hist_prices.columns else []
        ),
        format_func=lambda t: "Nifty 50" if t == "^NSEI" else all_names.get(t, t),
    )

    if picked:
        norm_df = hist_prices[picked].dropna(how="all")
        if not norm_df.empty:
            # Index each series to 100 at ITS OWN first valid price.
            first_valid = norm_df.apply(
                lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
            )
            norm_df = norm_df / first_valid * 100
            fig_hist = go.Figure()
            for i, col in enumerate(picked):
                label = "Nifty 50" if col == "^NSEI" else all_names.get(col, col)
                if col == "^NSEI":
                    style = dict(color=TEXT_MUTED, dash="dot", width=2)
                else:
                    style = dict(
                        width=2,
                        color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)],
                    )
                fig_hist.add_trace(
                    go.Scatter(
                        x=norm_df.index,
                        y=norm_df[col],
                        mode="lines",
                        name=label,
                        line=style,
                    )
                )
            fig_hist.update_layout(
                height=460,
                yaxis_title="Indexed value (start = 100)",
                xaxis_title="",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Select at least one instrument to plot.")

    st.markdown('<div class="section-title">Gold & Silver</div>', unsafe_allow_html=True)
    fi_tickers = [
        t for t in ["GOLDBEES.NS", "SILVERBEES.NS"]
        if t in close_df.columns
    ]
    if fi_tickers:
        fig_fi = go.Figure()
        for t in fi_tickers:
            series = close_df[t].dropna()
            series = series[series.index >= _hist_start]
            if not series.empty:
                line_color = GOLD if t == "GOLDBEES.NS" else "#7A8596"
                fig_fi.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        mode="lines",
                        name=dict((i["Ticker"], i["Name"]) for i in INSTRUMENTS_MASTER).get(t, t),
                        line=dict(color=line_color, width=2),
                    )
                )
        fig_fi.update_layout(height=340, yaxis_title="Price (₹)", xaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

st.write("")
st.caption(
    "Educational/simulation dashboard. The original holdings (units) are fixed from the 31 Aug 2026 entry prices "
    "and stored in a snapshot; current value is marked to the latest available trading price. Fractional units are "
    "allowed (no cash residual). Historical CAGR, beta and CAPM metrics are model estimates. Dividends are tracked from "
    "NSE corporate actions where available, with Yahoo Finance as a fallback; payment settlement is not fabricated when "
    "the source does not expose it. The G-Sec is carried at indicative par with a modelled accrued coupon."
)
