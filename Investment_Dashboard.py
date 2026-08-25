from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Portfolio & Investment Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Investment & Portfolio Dashboard")

# ============================================================================
# 1. PORTFOLIO CONFIGURATION
# ============================================================================

EQUITY_HOLDINGS = [
    # name, ticker (NSE), sector, invested_amount (Rs)
    ("Maruti Suzuki India", "MARUTI.NS", "Automobile", 900000),
    ("Tata Consultancy Services", "TCS.NS", "Information Technology", 900000),
    ("Indian Hotels Company", "INDHOTEL.NS", "Services & Logistics", 900000),
    ("Amara Raja Energy & Mobility", "ARE&M.NS", "Automobile", 600000),
    ("Housing & Urban Dev Corp", "HUDCO.NS", "Financial Services", 600000),
    ("Emami", "EMAMILTD.NS", "FMCG", 600000),
    ("Blue Star", "BLUESTARCO.NS", "Consumer Durables", 600000),
    ("IRCTC", "IRCTC.NS", "Services & Logistics", 600000),
    ("UTI Asset Management", "UTIAMC.NS", "Financial Services", 400000),
    ("Newgen Software", "NEWGEN.NS", "Information Technology", 400000),
    ("Heritage Foods", "HERITGFOOD.NS", "FMCG", 500000),
]

GOLD_HOLDINGS = [
    ("Nippon India Gold ETF (Gold BeES)", "GOLDBEES.NS", "Gold", 1000000),
]

DEBT_HOLDINGS = [
    # name, annual_ytm (decimal), invested_amount (Rs)
    ("AAA PSU Bond / Bharat Bond ETF", 0.078, 800000),
    ("AA/AA+ Corporate NCD", 0.098, 700000),
    ("RBI G-Sec / SDL (3-yr)", 0.073, 500000),
]

BENCHMARK_TICKER = "^NSEI"  # Nifty 50 index

# Sidebar Controls
st.sidebar.header("Dashboard Controls")
start_date = st.sidebar.date_input(
    "Start Date", value=date.today() - timedelta(days=365)
)

# ============================================================================
# 2. DATA FETCHING & PROCESSING
# ============================================================================


@st.cache_data
def fetch_price_history(tickers, start_d):
    """Download daily adjusted close price history for a list of tickers."""
    end = date.today() + timedelta(days=1)
    data = yf.download(tickers, start=start_d, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers
    return close.dropna(how="all")


def get_entry_and_current_price(price_series, start_d):
    """Entry price = first available close on/after start_date. Current = last available close."""
    series = price_series.dropna()
    if series.empty:
        return None, None
    entry_idx = series.index[series.index >= pd.Timestamp(start_d)]
    entry_price = (
        series[entry_idx[0]] if len(entry_idx) > 0 else series.iloc[0]
    )
    current_price = series.iloc[-1]
    return float(entry_price), float(current_price)


def accrued_debt_value(invested, annual_ytm, start_d):
    """Simple compounding accrual: value = invested * (1 + ytm)^(days/365)."""
    days_elapsed = (date.today() - start_d).days
    days_elapsed = max(days_elapsed, 0)
    return invested * (1 + annual_ytm) ** (days_elapsed / 365)


all_market_tickers = [t for _, t, _, _ in EQUITY_HOLDINGS] + [
    t for _, t, _, _ in GOLD_HOLDINGS
]
all_tickers_incl_benchmark = all_market_tickers + [BENCHMARK_TICKER]

with st.spinner("Fetching live and historical prices..."):
    price_history = fetch_price_history(
        all_tickers_incl_benchmark, start_date
    )

rows = []

# Equity
for name, ticker, sector, invested in EQUITY_HOLDINGS:
    if ticker not in price_history.columns:
        st.warning(f"No price data found for {ticker}")
        continue
    entry_price, current_price = get_entry_and_current_price(
        price_history[ticker], start_date
    )
    if entry_price is None:
        continue
    qty = invested / entry_price
    current_value = qty * current_price
    rows.append({
        "Instrument": name,
        "Sleeve": "Equity",
        "Sector": sector,
        "Invested (₹)": invested,
        "Entry Price": entry_price,
        "Current Price": current_price,
        "Quantity": qty,
        "Current Value (₹)": current_value,
    })

# Gold
for name, ticker, sector, invested in GOLD_HOLDINGS:
    if ticker not in price_history.columns:
        st.warning(f"No price data found for {ticker}")
        continue
    entry_price, current_price = get_entry_and_current_price(
        price_history[ticker], start_date
    )
    if entry_price is None:
        continue
    qty = invested / entry_price
    current_value = qty * current_price
    rows.append({
        "Instrument": name,
        "Sleeve": "Gold",
        "Sector": sector,
        "Invested (₹)": invested,
        "Entry Price": entry_price,
        "Current Price": current_price,
        "Quantity": qty,
        "Current Value (₹)": current_value,
    })

# Debt
for name, ytm, invested in DEBT_HOLDINGS:
    current_value = accrued_debt_value(invested, ytm, start_date)
    rows.append({
        "Instrument": name,
        "Sleeve": "Debt",
        "Sector": "Fixed Income",
        "Invested (₹)": invested,
        "Entry Price": np.nan,
        "Current Price": np.nan,
        "Quantity": np.nan,
        "Current Value (₹)": current_value,
    })

holdings = pd.DataFrame(rows)
holdings["Absolute Return (₹)"] = (
    holdings["Current Value (₹)"] - holdings["Invested (₹)"]
)
holdings["Return (%)"] = (
    holdings["Absolute Return (₹)"] / holdings["Invested (₹)"]
) * 100

total_invested = holdings["Invested (₹)"].sum()
total_current = holdings["Current Value (₹)"].sum()
total_abs_return = total_current - total_invested
total_pct_return = (total_abs_return / total_invested) * 100
days_held = max((date.today() - start_date).days, 1)
annualized_return = (
    (total_current / total_invested) ** (365 / days_held) - 1
) * 100

nifty_series = price_history[BENCHMARK_TICKER].dropna()
nifty_entry, nifty_current = get_entry_and_current_price(
    nifty_series, start_date
)
nifty_pct_return = (
    ((nifty_current - nifty_entry) / nifty_entry) * 100
    if nifty_entry
    else np.nan
)

# ============================================================================
# 3. DASHBOARD VISUALS & METRICS
# ============================================================================

st.header("Executive Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Invested", f"₹{total_invested:,.0f}")
m2.metric(
    "Current Value",
    f"₹{total_current:,.0f}",
    delta=f"{total_pct_return:+.2f}%",
)
m3.metric("Annualized Return", f"{annualized_return:.2f}%")
m4.metric(
    "vs Nifty 50",
    f"{total_pct_return - nifty_pct_return:+.2f}%",
    delta=f"Nifty: {nifty_pct_return:+.2f}%",
)

st.markdown("---")

# Portfolio vs Nifty Comparison Plot
st.subheader("Performance vs Nifty 50 Index")
market_price_df = price_history[all_market_tickers].dropna(how="all").ffill()
portfolio_daily_value = pd.Series(0.0, index=market_price_df.index)

for name, ticker, sector, invested in EQUITY_HOLDINGS + GOLD_HOLDINGS:
    if ticker not in market_price_df.columns:
        continue
    series = market_price_df[ticker]
    entry_idx = series.index[series.index >= pd.Timestamp(start_date)]
    if len(entry_idx) == 0:
        continue
    entry_price = series[entry_idx[0]]
    qty = invested / entry_price
    portfolio_daily_value = portfolio_daily_value.add(
        series * qty, fill_value=0
    )

for name, ytm, invested in DEBT_HOLDINGS:
    debt_series = pd.Series(
        [
            invested
            * (1 + ytm) ** (max((d.date() - start_date).days, 0) / 365)
            for d in market_price_df.index
        ],
        index=market_price_df.index,
    )
    portfolio_daily_value = portfolio_daily_value.add(
        debt_series, fill_value=0
    )

portfolio_indexed = (
    portfolio_daily_value / portfolio_daily_value.iloc[0]
) * 100
nifty_indexed = (
    nifty_series.reindex(portfolio_daily_value.index).ffill()
    / nifty_series.iloc[0]
) * 100

fig_compare = go.Figure()
fig_compare.add_trace(
    go.Scatter(
        x=portfolio_indexed.index,
        y=portfolio_indexed,
        name="Portfolio",
        line=dict(color="#2E86AB", width=2.5),
    )
)
fig_compare.add_trace(
    go.Scatter(
        x=nifty_indexed.index,
        y=nifty_indexed,
        name="Nifty 50",
        line=dict(color="#A23B72", width=2, dash="dash"),
    )
)
fig_compare.update_layout(
    height=420,
    xaxis_title="Date",
    yaxis_title="Indexed Value (Start = 100)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_compare, use_container_width=True)

# Allocation Pie Charts
st.subheader("Portfolio Allocations")
c1, c2 = st.columns(2)

with c1:
    sleeve_alloc = (
        holdings.groupby("Sleeve")["Current Value (₹)"].sum().reset_index()
    )
    fig_sleeve = px.pie(
        sleeve_alloc,
        names="Sleeve",
        values="Current Value (₹)",
        title="By Asset Class (Current Value)",
        hole=0.45,
        color_discrete_sequence=["#2E86AB", "#F18F01", "#C73E1D"],
    )
    st.plotly_chart(fig_sleeve, use_container_width=True)

with c2:
    sector_alloc = (
        holdings.groupby("Sector")["Current Value (₹)"].sum().reset_index()
    )
    fig_sector = px.pie(
        sector_alloc,
        names="Sector",
        values="Current Value (₹)",
        title="By Sector (Current Value)",
        hole=0.45,
    )
    st.plotly_chart(fig_sector, use_container_width=True)

# Return (%) by Instrument Bar Chart
st.subheader("Return (%) by Instrument")
display_df = holdings.copy()
fig_bar = px.bar(
    display_df.sort_values("Return (%)"),
    x="Return (%)",
    y="Instrument",
    orientation="h",
    color="Sleeve",
    color_discrete_sequence=["#2E86AB", "#F18F01", "#C73E1D"],
)
fig_bar.update_layout(height=500)
st.plotly_chart(fig_bar, use_container_width=True)

# Detailed Holdings Table
st.subheader("Holdings Details")
for col in ["Invested (₹)", "Current Value (₹)", "Absolute Return (₹)"]:
    display_df[col] = display_df[col].round(0)
display_df["Entry Price"] = display_df["Entry Price"].round(2)
display_df["Current Price"] = display_df["Current Price"].round(2)
display_df["Quantity"] = display_df["Quantity"].round(2)
display_df["Return (%)"] = display_df["Return (%)"].round(2)


def color_returns(val):
    if pd.isna(val):
        return ""
    color = "#1a7f37" if val >= 0 else "#c92a2a"
    return f"color: {color}; font-weight: 600"


styled = display_df.sort_values("Return (%)", ascending=False).style.map(
    color_returns, subset=["Absolute Return (₹)", "Return (%)"]
).format({
    "Invested (₹)": "₹{:,.0f}",
    "Current Value (₹)": "₹{:,.0f}",
    "Absolute Return (₹)": "₹{:,.0f}",
    "Return (%)": "{:+.2f}%",
    "Entry Price": "{:.2f}",
    "Current Price": "{:.2f}",
    "Quantity": "{:.2f}",
})

st.dataframe(styled, use_container_width=True)
