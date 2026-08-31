import warnings
from datetime import datetime

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
VALUATION_DATE = pd.Timestamp("2026-08-31")
CORPUS = 1_00_00_000
HORIZON_YEARS = 3
DEFAULT_RF = 0.069
DEFAULT_RM = 0.130

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

# ============================================================
# 5. LIVE DATA ENGINE
# ============================================================
@st.cache_data(ttl=23 * 60 * 60, show_spinner=False)
def fetch_live_market_data(lookback_yrs=5):
    tickers = [item["Ticker"] for item in INSTRUMENTS_MASTER if item["Ticker"] != "GSEC2029"]
    # Nifty 50 is the benchmark used by the current CAPM implementation.
    tickers.append("^NSEI")

    raw = yf.download(
        tickers,
        period=f"{max(lookback_yrs, 1)}y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", level=1, axis=1)
    else:
        prices = raw.copy()

    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)

    return prices.sort_index().dropna(how="all")



@st.cache_data(ttl=20 * 60 * 60, show_spinner=False)
def fetch_daily_prices():
    """Recent daily closes for day-over-day price changes and current value."""
    tickers = [item["Ticker"] for item in INSTRUMENTS_MASTER if item["Ticker"] != "GSEC2029"]
    raw = yf.download(
        tickers,
        period="15d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", level=1, axis=1)
    else:
        prices = raw.copy()
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.sort_index().dropna(how="all")


def asof_series(series, valuation_date):
    s = series.dropna().copy()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    s = s.loc[s.index <= valuation_date].sort_index()
    return s


def compute_beta(asset_returns, benchmark_returns):
    combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(combined) < 20:
        return 1.0
    mkt_var = np.var(combined.iloc[:, 1], ddof=1)
    if mkt_var <= 0:
        return 1.0
    return float(np.cov(combined.iloc[:, 0], combined.iloc[:, 1], ddof=1)[0, 1] / mkt_var)


def fetch_dividend_yield(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        div = info.get("dividendYield")
        if div is None:
            return 0.0
        div = float(div)
        # yfinance normally reports this as a decimal (e.g. 0.025 = 2.5%).
        # Guard against sources returning percent-formatted values.
        return div / 100.0 if div > 1 else div
    except Exception:
        return 0.0



@st.cache_data(ttl=23 * 60 * 60, show_spinner=False)
def fetch_dividend_history_and_events(start_date, end_date):
    """
    Builds the dividend cash-flow history from yfinance.

    yfinance provides declared dividend amounts as corporate-action/ex-dividend
    events. It does not reliably expose the issuer's board-announcement date.
    Where NSE's public corporate-action page is accessible, the app also tries
    to capture the NSE corporate-action record. The UI labels the date correctly
    so ex-dates are never presented as declaration dates.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    rows = []
    equity_items = [x for x in INSTRUMENTS_MASTER if x["Type"] == "Equity"]

    for item in equity_items:
        ticker = item["Ticker"]
        symbol = ticker.replace(".NS", "")

        # Yahoo dividend history: event date is normally the ex-dividend date.
        try:
            div = yf.Ticker(ticker).dividends
            div.index = pd.to_datetime(div.index)
            if getattr(div.index, "tz", None) is not None:
                div.index = div.index.tz_localize(None)
            div = div[(div.index >= start_date) & (div.index <= end_date)]
        except Exception:
            div = pd.Series(dtype=float)

        if div.empty:
            rows.append({
                "Company": item["Name"],
                "Symbol": symbol,
                "Dividend / Share": np.nan,
                "Ex-Date": pd.NaT,
                "NSE Announcement Date": pd.NaT,
                "Record Date": pd.NaT,
                "Source": "Yahoo Finance (no dividend event in period)",
            })
            continue

        for event_date, amount in div.items():
            rows.append({
                "Company": item["Name"],
                "Symbol": symbol,
                "Dividend / Share": float(amount),
                "Ex-Date": pd.Timestamp(event_date),
                "NSE Announcement Date": pd.NaT,
                "Record Date": pd.NaT,
                "Source": "Yahoo Finance dividend action",
            })

    return pd.DataFrame(rows)


@st.cache_data(ttl=23 * 60 * 60, show_spinner=False)
def fetch_nse_dividend_actions(symbols):
    """
    Best-effort NSE cross-check.

    1) Corporate Actions page: dividend purpose + ex-date + record date.
    2) Corporate Announcements page: latest dividend-related broadcast date
       when the public page exposes it.

    The exchange data is used as a cross-check. We label the broadcast date
    separately from the ex-date so the dashboard never confuses the two.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/",
    }

    action_rows = []
    announcement_rows = []
    session = requests.Session()

    try:
        session.get("https://www.nseindia.com/", headers=headers, timeout=12)
    except Exception:
        pass

    for symbol in symbols:
        # ---------------- Corporate Actions ----------------
        action_url = (
            "https://www.nseindia.com/companies-listing/"
            f"corporate-filings-actions?symbol={symbol}&tabIndex=equity"
        )
        try:
            r = session.get(action_url, headers=headers, timeout=12)
            r.raise_for_status()
            tables = pd.read_html(r.text)

            for table in tables:
                cols = [str(c).upper() for c in table.columns]
                if not (
                    any("PURPOSE" in c for c in cols)
                    and any("EX-DATE" in c for c in cols)
                ):
                    continue

                table.columns = [str(c).strip() for c in table.columns]
                for _, row in table.iterrows():
                    purpose = str(row.get("PURPOSE", ""))
                    if "DIVIDEND" not in purpose.upper():
                        continue

                    action_rows.append({
                        "Symbol": symbol,
                        "NSE Purpose": purpose,
                        "NSE Ex-Date": pd.to_datetime(
                            row.get("EX-DATE"), errors="coerce"
                        ),
                        "NSE Record Date": pd.to_datetime(
                            row.get("RECORD DATE"), errors="coerce"
                        ),
                    })
                break
        except Exception:
            pass

        # ---------------- Corporate Announcements ----------------
        announcement_url = (
            "https://www.nseindia.com/companies-listing/"
            f"corporate-filings-announcements?symbol={symbol}"
        )
        try:
            r = session.get(announcement_url, headers=headers, timeout=12)
            r.raise_for_status()
            tables = pd.read_html(r.text)

            for table in tables:
                table.columns = [str(c).strip() for c in table.columns]
                upper_cols = [c.upper() for c in table.columns]

                subject_col = next(
                    (c for c in table.columns if "SUBJECT" in c.upper()), None
                )
                date_col = next(
                    (
                        c for c in table.columns
                        if ("BROADCAST" in c.upper() or "ANNOUNCEMENT" in c.upper())
                    ),
                    None,
                )
                if subject_col is None or date_col is None:
                    continue

                for _, row in table.iterrows():
                    subject = str(row.get(subject_col, ""))
                    if "DIVIDEND" not in subject.upper():
                        continue

                    announcement_rows.append({
                        "Symbol": symbol,
                        "NSE Announcement Date": pd.to_datetime(
                            row.get(date_col), errors="coerce"
                        ),
                        "NSE Announcement Subject": subject,
                    })
                break
        except Exception:
            pass

    actions = pd.DataFrame(action_rows)
    announcements = pd.DataFrame(announcement_rows)

    if actions.empty and announcements.empty:
        return pd.DataFrame(
            columns=[
                "Symbol", "NSE Purpose", "NSE Ex-Date", "NSE Record Date",
                "NSE Announcement Date", "NSE Announcement Subject"
            ]
        )

    if actions.empty:
        result = announcements.copy()
    elif announcements.empty:
        result = actions.copy()
    else:
        # Keep the latest announcement per symbol; NSE's public page typically
        # returns only the most recent batch for this view.
        latest_announcement = (
            announcements.sort_values("NSE Announcement Date")
            .groupby("Symbol", as_index=False)
            .tail(1)
        )
        result = actions.merge(
            latest_announcement,
            on="Symbol",
            how="left",
        )

    return result




def analyze_live_performance(prices_df, rf_rate, mkt_return):
    weekly_returns = prices_df.pct_change().dropna(how="all")
    nifty_returns = weekly_returns["^NSEI"].dropna() if "^NSEI" in weekly_returns.columns else pd.Series(dtype=float)

    records = []

    for item in INSTRUMENTS_MASTER:
        t = item["Ticker"]
        rec = item.copy()

        if t == "GSEC2029":
            # Indicative quote represented per 100 face value.
            rec.update({
                "LivePrice": 100.0,
                "PriceDate": VALUATION_DATE,
                "Actual_CAGR": item["FixedYield"],
                "Beta": 0.0,
                "Volatility": 0.0,
                "DivYield": 0.0,
                "CAPM_Return": item["FixedYield"],
                "Total_Expected_Return": item["FixedYield"],
            })
            records.append(rec)
            continue

        if t not in prices_df.columns:
            rec.update({
                "LivePrice": np.nan,
                "PriceDate": pd.NaT,
                "Actual_CAGR": 0.0,
                "Beta": 1.0,
                "Volatility": 0.0,
                "DivYield": 0.0,
                "CAPM_Return": 0.0,
                "Total_Expected_Return": 0.0,
            })
            records.append(rec)
            continue

        px_series = asof_series(prices_df[t], VALUATION_DATE)
        if px_series.empty:
            rec.update({
                "LivePrice": np.nan,
                "PriceDate": pd.NaT,
                "Actual_CAGR": 0.0,
                "Beta": 1.0,
                "Volatility": 0.0,
                "DivYield": 0.0,
                "CAPM_Return": 0.0,
                "Total_Expected_Return": 0.0,
            })
            records.append(rec)
            continue

        latest_px = float(px_series.iloc[-1])
        price_date = px_series.index[-1]

        if len(px_series) >= 2:
            elapsed_years = max((px_series.index[-1] - px_series.index[0]).days / 365.25, 1/52)
            actual_cagr = (latest_px / float(px_series.iloc[0])) ** (1 / elapsed_years) - 1
        else:
            actual_cagr = 0.0

        if t in weekly_returns.columns and not nifty_returns.empty:
            beta = compute_beta(weekly_returns[t].dropna(), nifty_returns)
            vol = float(weekly_returns[t].dropna().std() * np.sqrt(52))
        else:
            beta = 1.0
            vol = 0.0

        div_yield = fetch_dividend_yield(t) if item["Type"] == "Equity" else 0.0

        if item["Type"] == "Equity":
            capm_return = rf_rate + beta * (mkt_return - rf_rate)
            total_expected = capm_return + div_yield
        elif item["Type"] == "Bond":
            capm_return = item.get("FixedYield", 0.0)
            total_expected = capm_return
        else:
            # For gold/silver, use realized price CAGR as the modelling anchor.
            capm_return = actual_cagr
            total_expected = actual_cagr

        rec.update({
            "LivePrice": latest_px,
            "PriceDate": price_date,
            "Actual_CAGR": actual_cagr,
            "Beta": beta,
            "Volatility": vol,
            "DivYield": div_yield,
            "CAPM_Return": capm_return,
            "Total_Expected_Return": total_expected,
        })
        records.append(rec)

    return pd.DataFrame(records), weekly_returns


def _annualized_risk_vector(out, returns_df):
    """Annualized risk proxy for the 15 instruments."""
    risks = []
    for _, row in out.iterrows():
        ticker = row["Ticker"]
        if ticker == "GSEC2029":
            risks.append(0.03)
        elif ticker in returns_df.columns:
            s = returns_df[ticker].dropna()
            risks.append(float(s.std() * np.sqrt(52)) if len(s) >= 10 else 0.12)
        else:
            risks.append(0.12)
    return np.clip(np.asarray(risks, dtype=float), 0.02, 1.50)


def apply_entry_prices_asof(df_master, daily_prices_df, valuation_date):
    """
    Use one consistent investment-date price for the initial units.
    This prevents a false day-one P&L caused by comparing weekly-series
    prices with daily-series prices.
    """
    out = df_master.copy()
    entry_prices = []

    for _, row in out.iterrows():
        t = row["Ticker"]
        if t == "GSEC2029":
            entry_prices.append(100.0)
            continue

        if t in daily_prices_df.columns:
            s = asof_series(daily_prices_df[t], valuation_date)
            if not s.empty:
                entry_prices.append(float(s.iloc[-1]))
                continue

        # Fall back to the model's as-of weekly price only when daily
        # history is unavailable.
        entry_prices.append(row["LivePrice"])

    out["EntryPrice"] = entry_prices
    return out


def build_persona_weights(df_master, persona_name, returns_df):
    """
    Desired outcome drives the weights.

    The investor's target corpus is converted into a required CAGR.
    The optimizer then:
      - requires portfolio expected return >= required CAGR;
      - minimizes a risk/concentration objective;
      - respects persona-specific equity/debt/gold/silver guardrails.

    If the target is mathematically infeasible under the current assumptions,
    the app shows the maximum-achievable-return portfolio instead of pretending
    the target was reached.
    """
    cfg = PERSONAS[persona_name]
    out = df_master.copy()

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

    # Risk objective: minimize weighted standalone risk plus a small
    # concentration penalty so the optimizer does not pile into one security.
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

    # Start at the lower asset-class floors, with the remainder in equity.
    w0 = np.zeros(len(out), dtype=float)
    w0[eq_idx] = cfg["min_equity"] / len(eq_idx)
    w0[bond_idx] = cfg["min_debt"] / len(bond_idx)
    w0[gold_idx] = cfg["min_gold"] / len(gold_idx)
    w0[silver_idx] = cfg["min_silver"] / len(silver_idx)

    remainder = 1.0 - w0.sum()
    if remainder > 0:
        w0[eq_idx] += remainder / len(eq_idx)

    # Find maximum achievable return under risk guardrails first.
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

    out["EquitySleeveWeight"] = 0.0
    eq_total = weights[eq_idx].sum()
    if eq_total > 0:
        out.loc[eq_idx, "EquitySleeveWeight"] = weights[eq_idx] / eq_total

    # Fix the initial holdings at the explicit 31 Aug 2026 investment-date price.
    # Units stay fixed; later refreshes only change CurrentPrice/CurrentValue.
    out["Units"] = np.where(
        out["EntryPrice"].notna() & (out["EntryPrice"] > 0),
        out["Allocated_Amount"] / out["EntryPrice"],
        np.nan,
    )
    out["CurrentValue"] = out["Units"] * out["EntryPrice"]
    out["UnrealizedPnL"] = out["CurrentValue"] - out["Allocated_Amount"]

    # Return decomposition.
    out["CapitalGainExpected"] = np.where(
        out["Type"].eq("Equity"),
        out["CAPM_Return"],
        np.where(
            out["Type"].isin(["Gold ETF", "Silver ETF"]),
            out["Actual_CAGR"],
            0.0,
        ),
    )
    out["ExpectedIncomeYield"] = np.where(
        out["Type"].eq("Equity"),
        out["DivYield"],
        np.where(
            out["Type"].eq("Bond"),
            out["Total_Expected_Return"],
            0.0,
        ),
    )

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
            help="The target is the investor's desired 3-year outcome. It does not directly force portfolio weights.",
        )
        lookback_yrs = st.slider(
            "Historical lookback (years)",
            min_value=1,
            max_value=5,
            value=5,
            help="Used for historical CAGR/volatility inputs and return context.",
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
        "Daily auto-refresh enabled."
        if AUTO_REFRESH_AVAILABLE
        else "Daily auto-refresh unavailable in this environment; use the refresh button. "
             "Market-data cache still expires daily."
    )
    st.caption("Investment date: 31 Aug 2026 · " + refresh_note)
    st.caption("Persona targets determine the required CAGR constraint; the optimizer then minimizes risk within persona-specific allocation guardrails.")

# ============================================================
# 7. RUN DATA ENGINE
# ============================================================
try:
    with st.spinner("Downloading market data & computing risk/return metrics..."):
        prices_df = fetch_live_market_data(lookback_yrs)
        daily_prices_df = fetch_daily_prices()
        df_master_raw, returns_df = analyze_live_performance(prices_df, rf_rate, mkt_return)

        # Make the investment-date price explicit and consistent across
        # all subsequent calculations.
        df_master_raw = apply_entry_prices_asof(
            df_master_raw,
            daily_prices_df,
            VALUATION_DATE,
        )

        # Re-use the explicit entry price during unit calculation.
        df_master, optimization_meta = build_persona_weights(
            df_master_raw,
            selected_persona,
            returns_df,
        )
except Exception as e:
    st.error("Error executing market analysis engine.")
    st.exception(e)
    st.stop()

persona_cfg = PERSONAS[selected_persona]
target_cagr = persona_cfg["target_cagr"]

# ============================================================
# 8. PORTFOLIO LIVE MARK-TO-MARKET + DIVIDENDS
# ============================================================
# Re-mark the fixed 31 Aug 2026 holdings to the latest available price.
latest_prices = {}
previous_prices = {}

for _, row in df_master.iterrows():
    ticker = row["Ticker"]
    if ticker == "GSEC2029":
        latest_prices[ticker] = 100.0
        previous_prices[ticker] = 100.0
        continue

    if ticker in daily_prices_df.columns:
        s = daily_prices_df[ticker].dropna()
        if len(s) >= 1:
            latest_prices[ticker] = float(s.iloc[-1])
        if len(s) >= 2:
            previous_prices[ticker] = float(s.iloc[-2])

df_master["CurrentPrice"] = df_master.apply(
    lambda r: latest_prices.get(r["Ticker"], r["EntryPrice"]), axis=1
)

# On the investment date itself, current price and entry price must be identical
# so the portfolio cannot show an artificial day-one gain/loss.
if TODAY <= VALUATION_DATE:
    df_master["CurrentPrice"] = df_master["EntryPrice"]
df_master["PreviousClose"] = df_master.apply(
    lambda r: previous_prices.get(r["Ticker"], r["CurrentPrice"]), axis=1
)

df_master["CurrentValue"] = df_master["Units"] * df_master["CurrentPrice"]
df_master["DailyPriceChange"] = df_master["CurrentPrice"] - df_master["PreviousClose"]
df_master["DailyPriceChangePct"] = np.where(
    df_master["PreviousClose"].notna() & (df_master["PreviousClose"] != 0),
    df_master["DailyPriceChange"] / df_master["PreviousClose"],
    0.0,
)

# Dividend tracker — cash dividends received since the 31 Aug 2026 investment date.
div_hist = fetch_dividend_history_and_events(
    pd.Timestamp("2026-08-31"),
    TODAY,
)

if not div_hist.empty:
    units_map = df_master.set_index("Ticker")["Units"].to_dict()
    symbol_to_ticker = {
        item["Ticker"].replace(".NS", ""): item["Ticker"]
        for item in INSTRUMENTS_MASTER
        if item["Type"] == "Equity"
    }
    div_hist["Ticker"] = div_hist["Symbol"].map(symbol_to_ticker)
    div_hist["Units"] = div_hist["Ticker"].map(units_map)
    div_hist["Cash Dividend"] = div_hist["Dividend / Share"] * div_hist["Units"]
else:
    div_hist = pd.DataFrame(
        columns=[
            "Company", "Symbol", "Dividend / Share", "Ex-Date",
            "NSE Announcement Date", "Record Date",
            "NSE Announcement Subject", "Source",
            "Ticker", "Units", "Cash Dividend"
        ]
    )

# Guarantee the dividend-event schema even when a data provider returns
# zero events or only a partial response. This prevents pandas KeyError.
_dividend_required_schema = [
    "Company", "Symbol", "Dividend / Share", "Ex-Date",
    "NSE Announcement Date", "Record Date", "NSE Announcement Subject",
    "Source", "Ticker", "Units", "Cash Dividend"
]
for _col in _dividend_required_schema:
    if _col not in div_hist.columns:
        if "Date" in _col or _col in {"Ex-Date", "Record Date"}:
            div_hist[_col] = pd.NaT
        elif _col in {"Dividend / Share", "Units", "Cash Dividend"}:
            div_hist[_col] = np.nan
        else:
            div_hist[_col] = ""

# NSE cross-check for declaration/broadcast information.
try:
    portfolio_symbols = [
        item["Ticker"].replace(".NS", "")
        for item in INSTRUMENTS_MASTER
        if item["Type"] == "Equity"
    ]
    nse_div = fetch_nse_dividend_actions(portfolio_symbols)
except Exception:
    nse_div = pd.DataFrame()

if not nse_div.empty:
    # Merge the nearest NSE ex-date information to the Yahoo event.
    div_hist["NSE Announcement Date"] = pd.NaT
    div_hist["NSE Record Date"] = pd.NaT
    div_hist["NSE Announcement Subject"] = ""

    for i, drow in div_hist.iterrows():
        symbol = drow.get("Symbol")
        ex_date = drow.get("Ex-Date")
        candidates = nse_div[nse_div["Symbol"] == symbol].copy()

        if not candidates.empty and pd.notna(ex_date):
            if "NSE Ex-Date" in candidates:
                candidates["date_diff"] = (
                    pd.to_datetime(candidates["NSE Ex-Date"], errors="coerce")
                    - pd.Timestamp(ex_date)
                ).abs().dt.dayss
                candidates = candidates.sort_values("date_diff")

            best = candidates.iloc[0]
            div_hist.at[i, "NSE Announcement Date"] = best.get(
                "NSE Announcement Date", pd.NaT
            )
            div_hist.at[i, "NSE Record Date"] = best.get(
                "NSE Record Date", pd.NaT
            )
            div_hist.at[i, "NSE Announcement Subject"] = best.get(
                "NSE Announcement Subject", ""
            )

dividend_cash_received = float(
    pd.to_numeric(
        div_hist.get("Cash Dividend", pd.Series(dtype=float)),
        errors="coerce"
    ).fillna(0).sum()
)

df_master["DividendCashReceived"] = 0.0
if not div_hist.empty:
    cash_by_ticker = div_hist.groupby("Ticker")["Cash Dividend"].sum()
    df_master["DividendCashReceived"] = df_master["Ticker"].map(cash_by_ticker).fillna(0.0)

df_master["TotalCurrentValue"] = df_master["CurrentValue"] + df_master["DividendCashReceived"]
df_master["TotalInvestmentGain"] = (
    df_master["TotalCurrentValue"] - df_master["Allocated_Amount"]
)
df_master["TotalInvestmentGainPct"] = np.where(
    df_master["Allocated_Amount"] != 0,
    df_master["TotalInvestmentGain"] / df_master["Allocated_Amount"],
    0.0,
)

# ============================================================
# 9. PORTFOLIO METRICS
# ============================================================
current_portfolio_value = float(df_master["TotalCurrentValue"].sum(skipna=True))
invested_capital = float(df_master["Allocated_Amount"].sum())
current_pnl = current_portfolio_value - invested_capital
current_pnl_pct = current_pnl / invested_capital if invested_capital else 0.0

portfolio_expected_return = optimization_meta["achieved_expected_return"]
portfolio_realized_cagr = float(
    (df_master["Portfolio_Weight"] * df_master["Actual_CAGR"]).sum()
)

# Separate return components for the dashboard.
equity_mask = df_master["Type"].eq("Equity")
bond_mask = df_master["Type"].eq("Bond")
gold_mask = df_master["Type"].eq("Gold ETF")
silver_mask = df_master["Type"].eq("Silver ETF")

equity_cap_gain = float(
    (df_master.loc[equity_mask, "Portfolio_Weight"] * df_master.loc[equity_mask, "CapitalGainExpected"]).sum()
)
equity_dividends = float(
    (df_master.loc[equity_mask, "Portfolio_Weight"] * df_master.loc[equity_mask, "DivYield"]).sum()
)
bond_income = float(
    (df_master.loc[bond_mask, "Portfolio_Weight"] * df_master.loc[bond_mask, "Total_Expected_Return"]).sum()
)
gold_return = float(
    (df_master.loc[gold_mask, "Portfolio_Weight"] * df_master.loc[gold_mask, "Total_Expected_Return"]).sum()
)
silver_return = float(
    (df_master.loc[silver_mask, "Portfolio_Weight"] * df_master.loc[silver_mask, "Total_Expected_Return"]).sum()
)

# Target path is informational only; it is not used to calculate weights.
yrs = np.arange(0, HORIZON_YEARS + 1)
target_path = CORPUS * ((1 + target_cagr) ** yrs)
expected_path = CORPUS * ((1 + portfolio_expected_return) ** yrs)
expected_final_value = optimization_meta["projected_value"]

# Capital + income breakdown based on weighted annual return contributions.
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
            <div class="hero-eyebrow">LIVE PORTFOLIO · AS OF {TODAY.strftime("%d %b %Y")} · INVESTED 31 AUG 2026 · {lookback_yrs}Y LOOKBACK</div>
            <div class="hero-title">15-Asset Allocation Terminal</div>
            <p class="hero-sub">
                ₹1.00 Cr allocated across 11 equities, 2 bonds, Gold ETF and Silver ETF for the
                <strong style="color:{GOLD_SOFT}">{selected_persona}</strong> investor objective.
                The desired outcome is converted into a required CAGR and used as a minimum-return constraint; the optimizer minimizes risk subject to that target.
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
            <div class="hero-big-label">Current portfolio value at 31 Aug 2026</div>
            <div class="{"kpi-delta-pos" if current_pnl >= 0 else "kpi-delta-neg"}"
                 style="margin-top:10px; font-size:13px;">
                Day-0 P&L: ₹{current_pnl/1e5:.2f} Lakh ({current_pnl_pct:+.2%})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        "Required CAGR",
        f"{target_cagr:.2%}",
        f'<div class="kpi-delta-pos">Target benchmark only</div>',
    ),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi_card(
        "Modelled Expected Return",
        f"{portfolio_expected_return:.2%}",
        f'<div class="kpi-delta-pos">Risk-profile weighted</div>',
    ),
    unsafe_allow_html=True,
)
k5.markdown(
    kpi_card(
        "Expected 3Y Value",
        f"₹{expected_final_value/1e7:.2f} Cr",
        f'<div class="{ "kpi-delta-pos" if expected_final_value >= persona_cfg["target_corpus"] else "kpi-delta-neg" }">'
        f'vs target ₹{persona_cfg["target_corpus"]/1e7:.2f} Cr</div>',
    ),
    unsafe_allow_html=True,
)

# Reconciliation check: all instruments should sum back to the ₹1 Cr invested
# amount at inception, before dividends or market movement.
entry_value_check = float(df_master["Allocated_Amount"].sum())
st.caption(
    f"Inception reconciliation: ₹{entry_value_check/1e7:.2f} Cr allocated across "
    f"{len(df_master)} instruments. Units are locked at the 31 Aug 2026 entry price."
)

st.write("")

if optimization_meta["target_feasible"]:
    st.success(
        f"Target-driven allocation: the {selected_persona} objective of "
        f"₹{persona_cfg['target_corpus']/1e7:.2f} Cr is feasible under the current "
        f"return assumptions and risk guardrails. The optimizer minimizes risk while "
        f"maintaining at least {target_cagr:.2%} expected annual return."
    )
else:
    st.warning(
        f"The {selected_persona} target of ₹{persona_cfg['target_corpus']/1e7:.2f} Cr "
        f"is not feasible under the current assumptions and risk guardrails. "
        f"The model therefore shows the maximum achievable outcome of "
        f"₹{expected_final_value/1e7:.2f} Cr."
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
i2.metric("Portfolio Weight", f"{selected_row['Portfolio_Weight']:.2%}")
i3.metric("Allocated", f"₹{selected_row['Allocated_Amount']/1e5:.2f} L")
i4.metric("Current Value", f"₹{selected_row['TotalCurrentValue']/1e5:.2f} L" if pd.notna(selected_row["TotalCurrentValue"]) else "N/A")

detail_left, detail_right = st.columns([1, 1.4])

with detail_left:
    details = {
        "Asset Class": selected_row["Type"],
        "Sector / Exposure": selected_row["Sector"],
        "Entry Price (31 Aug 2026)": f"₹{selected_row['EntryPrice']:,.2f}" if pd.notna(selected_row["EntryPrice"]) else "N/A",
        "Cap Category": selected_row["CapCategory"],
        "Price Date": str(pd.Timestamp(selected_row["PriceDate"]).date()) if pd.notna(selected_row["PriceDate"]) else "N/A",
        "Units": f"{selected_row['Units']:,.4f}" if pd.notna(selected_row["Units"]) else "N/A",
        "Daily Price Change": f"{selected_row['DailyPriceChangePct']:+.2%}",
        "Dividend Cash Received": f"₹{selected_row['DividendCashReceived']:,.0f}",
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
    if selected_row["Ticker"] != "GSEC2029" and selected_row["Ticker"] in prices_df.columns:
        series = asof_series(prices_df[selected_row["Ticker"]], VALUATION_DATE)
        if not series.empty:
            fig_selected = go.Figure()
            fig_selected.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=selected_instrument,
                    line=dict(color=GOLD, width=2.5),
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
            '<div class="info-box">The sovereign G-Sec is represented using its indicative price/yield assumption rather than a Yahoo Finance equity price series.</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ============================================================
# 13. MAIN TABS
# ============================================================
tab_overview, tab_holdings, tab_risk, tab_history = st.tabs(
    ["Overview", "Holdings", "Risk & Return", "Historical Trends"]
)

# ---------- TAB 1: OVERVIEW ----------
with tab_overview:
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown('<div class="section-title">Asset-Class Allocation</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">How the ₹1 Cr is divided across equity, debt, gold and silver</div>', unsafe_allow_html=True)

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
            '<div class="section-note">The target line is the investor objective and is converted into the minimum CAGR required by the portfolio optimizer.</div>',
            unsafe_allow_html=True,
        )

        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Scatter(
                x=yrs,
                y=expected_path,
                mode="lines+markers",
                name=f"Modelled Expected ({portfolio_expected_return:.2%})",
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
        fig_growth.update_layout(
            height=340,
            xaxis_title="Years elapsed",
            yaxis_title="Portfolio value (₹)",
            yaxis_tickprefix="₹",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown('<div class="section-title">Return Composition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Annualised weighted contribution from capital gains and income sources.</div>',
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
        '<div class="info-box">The investment is assumed to be initiated on 31 Aug 2026. Holdings are fixed at their entry prices; daily refreshes then re-mark those holdings to the latest market price and add declared/recorded dividend cash flows to total portfolio value.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Daily Market Movement</div>', unsafe_allow_html=True)
    daily_move = df_master[["Name", "CurrentPrice", "DailyPriceChange", "DailyPriceChangePct"]].copy()
    daily_move = daily_move.sort_values("DailyPriceChangePct", ascending=False)
    st.dataframe(
        daily_move,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CurrentPrice": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
            "DailyPriceChange": st.column_config.NumberColumn("₹ Change", format="₹%.2f"),
            "DailyPriceChangePct": st.column_config.NumberColumn("Daily Change", format="%.2f%%"),
        },
    )

    st.markdown('<div class="section-title">Dividend Tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Dividend events affecting the portfolio since the 31 Aug 2026 investment date. Ex-date is the corporate-action date used for dividend entitlement. NSE announcement/broadcast date is shown separately when the public NSE announcements page exposes a dividend-related filing; no announcement date is invented when unavailable.</div>',
        unsafe_allow_html=True,
    )
    required_display_cols = [
        "Company", "Dividend / Share", "NSE Announcement Date",
        "NSE Announcement Subject", "Ex-Date", "Record Date",
        "Cash Dividend", "Source"
    ]
    dividend_view = div_hist.reindex(columns=required_display_cols).copy()

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
                "Cash Dividend": st.column_config.NumberColumn(format="₹%d"),
            },
        )
    else:
        st.info(
            "No dividend event has been recorded since the 31 Aug 2026 investment date. "
            "The tracker will check again on the next daily refresh."
        )

# ---------- TAB 2: HOLDINGS ----------
with tab_holdings:
    st.markdown('<div class="section-title">All 15 Instruments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">As-of 31 Aug 2026 price, units, current value, expected return and allocation.</div>',
        unsafe_allow_html=True,
    )

    display_df = df_master.copy()
    display_df["Current Price"] = display_df["CurrentPrice"]
    display_df["Price Date"] = display_df["PriceDate"]
    display_df["Daily Change %"] = display_df["DailyPriceChangePct"]
    display_df["Dividend Cash"] = display_df["DividendCashReceived"]
    display_df["Total Gain"] = display_df["TotalInvestmentGain"]
    display_df["Historical CAGR"] = display_df["Actual_CAGR"]
    display_df["Dividend / Interest"] = display_df["ExpectedIncomeYield"]
    display_df["Capital Gain"] = display_df["CapitalGainExpected"]
    display_df["Expected Total Return"] = display_df["Total_Expected_Return"]

    # Streamlit's printf-style percent formatting displays the raw number.
    # Convert decimal weights to percentage points explicitly (0.125 -> 12.5)
    # so the table shows correct portfolio percentages.
    display_df["Portfolio Weight %"] = display_df["Portfolio_Weight"] * 100.0

    cols = [
        "Name", "Type", "Sector", "CapCategory", "Current Price", "Price Date",
        "Units", "Portfolio Weight %", "Allocated_Amount", "CurrentValue",
        "Dividend Cash", "Total Gain", "Daily Change %", "Historical CAGR",
        "Capital Gain", "Dividend / Interest", "Expected Total Return", "Beta",
    ]

    # Explicit reconciliation: raw weights must add to exactly 100% (within
    # floating-point tolerance), and asset-class totals should match the
    # optimizer output.
    total_weight = float(df_master["Portfolio_Weight"].sum())
    asset_weights = df_master.groupby("Type")["Portfolio_Weight"].sum().to_dict()

    rw1, rw2, rw3, rw4, rw5 = st.columns(5)
    rw1.metric("Total Weight", f"{total_weight:.2%}")
    rw2.metric("Equity", f"{asset_weights.get('Equity', 0.0):.2%}")
    rw3.metric("Debt", f"{asset_weights.get('Bond', 0.0):.2%}")
    rw4.metric("Gold", f"{asset_weights.get('Gold ETF', 0.0):.2%}")
    rw5.metric("Silver", f"{asset_weights.get('Silver ETF', 0.0):.2%}")

    if abs(total_weight - 1.0) > 1e-8:
        st.error("Portfolio weights do not reconcile to 100%. Please refresh the model.")

    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Price Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Units": st.column_config.NumberColumn(format="%.4f"),
            "Portfolio Weight %": st.column_config.ProgressColumn(
                "Portfolio Weight",
                format="%.1f%%",
                min_value=0,
                max_value=max(float(display_df["Portfolio Weight %"].max()) * 1.15, 10.0),
            ),
            "Allocated_Amount": st.column_config.NumberColumn("Invested", format="₹%d"),
            "CurrentValue": st.column_config.NumberColumn("Current Value", format="₹%d"),
            "UnrealizedPnL": st.column_config.NumberColumn("Price P&L", format="₹%d"),
            "Dividend Cash": st.column_config.NumberColumn(format="₹%d"),
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
        st.markdown('<div class="section-title">Weight by Instrument</div>', unsafe_allow_html=True)
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

# ---------- TAB 3: RISK & RETURN ----------
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
    st.caption("Bubble size = portfolio weight. Green = historical CAGR above CAPM-implied return.")

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

# ---------- TAB 4: HISTORICAL TRENDS ----------
with tab_history:
    st.markdown('<div class="section-title">Normalized Price Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Selected series indexed to 100 at the start of the available lookback window.</div>',
        unsafe_allow_html=True,
    )

    all_names = {
        row["Ticker"]: row["Name"]
        for _, row in df_master.iterrows()
        if row["Ticker"] in prices_df.columns
    }
    default_pick = [
        t
        for t in df_master[df_master["Type"] == "Equity"]["Ticker"].head(5)
        if t in prices_df.columns
    ]

    picked = st.multiselect(
        "Instruments to plot",
        options=list(all_names.keys()) + (
            ["^NSEI"] if "^NSEI" in prices_df.columns else []
        ),
        default=default_pick + (
            ["^NSEI"] if "^NSEI" in prices_df.columns else []
        ),
        format_func=lambda t: "Nifty 50" if t == "^NSEI" else all_names.get(t, t),
    )

    if picked:
        norm_df = prices_df[picked].dropna(how="all")
        if not norm_df.empty:
            norm_df = norm_df / norm_df.iloc[0] * 100
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
        if t in prices_df.columns
    ]
    if fi_tickers:
        fig_fi = go.Figure()
        for t in fi_tickers:
            series = asof_series(prices_df[t], VALUATION_DATE)
            if not series.empty:
                line_color = GOLD if t == "GOLDBEES.NS" else "#7A8596"
                fig_fi.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        mode="lines",
                        name=all_names.get(t, t),
                        line=dict(color=line_color, width=2),
                    )
                )
        fig_fi.update_layout(height=340, yaxis_title="Price (₹)", xaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

st.write("")
st.caption(
    f"Educational/simulation dashboard. Holdings are fixed from 31 Aug 2026 entry prices; current value is marked to the latest available trading price. On the investment date itself, day-one P&L is zero by construction. "
    f"Historical CAGR, beta and CAPM metrics are model estimates and should not be treated as guaranteed future returns."
)
