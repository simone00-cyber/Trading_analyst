"""
CYCLICAL GLOBAL MACRO TERMINAL

Dashboard Streamlit in stile terminale professionale.

Pagine:
1. Global Overview
2. Global Macro
3. Market Regime
4. Security Report
5. Methodology

Sorgente dati di mercato: Yahoo Finance tramite yfinance.
La metodologia ciclica del Security Report usa esclusivamente le componenti
formalizzate nei paper forniti: KEY, XTL, Composite Momentum e lettura
multi-timeframe.

Avvio:
    python -m streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
import yfinance as yf

from caruso_analysis import (
    RESAMPLE_RULES,
    TimeframeResult,
    calculate_composite_momentum,
    download_prices,
    download_prices_raw,
    resample_ohlc,
    strategy_from_matrix,
    summarize_timeframe,
)

import requests
url = "https://api.kpler.com/v2/maritime/ais-latest"

headers = {
    "Authorization": "Basic YOUR_KEY",
    "Accept": "application/json"
}

response = requests.get(
    url,
    headers=headers,
    timeout=30
)

print(response.status_code)

data = response.json()

print(type(data))
print(data)
# =============================================================================
# CONFIGURAZIONE GRAFICA
# =============================================================================

st.set_page_config(
    page_title="Cyclical Global Macro Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#050505"
PANEL = "#0e0e0e"
PANEL_2 = "#151515"
ORANGE = "#ff9f00"
GREEN = "#00d26a"
RED = "#ff3b3b"
TEXT = "#f2f2f2"
MUTED = "#9a9a9a"
GRID = "#2a2a2a"
BLUE = "#4da3ff"
CYAN = "#3ee6e0"
PURPLE = "#b58cff"

CUSTOM_CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: Consolas, "Courier New", monospace;
    }}

    .stApp {{ background: {BG}; color: {TEXT}; }}

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }}

    .stApp > header {{ display: none !important; }}

    .block-container {{
        padding-top: 0.25rem !important;
        padding-bottom: 3rem;
        max-width: 100%;
    }}

    section[data-testid="stSidebar"] {{
        background: #080808;
        border-right: 1px solid #272727;
    }}

    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

    h1, h2, h3 {{
        color: {ORANGE} !important;
        letter-spacing: 0.02em;
    }}

    .top-terminal-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {ORANGE};
        color: #000;
        padding: 0.42rem 0.70rem;
        font-weight: 900;
        border-bottom: 2px solid #000;
        margin-bottom: 0.18rem;
    }}

    .ticker-strip {{
        display: flex;
        gap: 1.10rem;
        flex-wrap: wrap;
        background: #0a0a0a;
        border-top: 1px solid #292929;
        border-bottom: 1px solid #292929;
        padding: 0.42rem 0.70rem;
        margin-bottom: 0.70rem;
        font-size: 0.84rem;
    }}

    .terminal-header {{
        background: {ORANGE};
        color: #000;
        padding: 0.55rem 0.8rem;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.7rem;
        border-radius: 2px;
    }}

    .terminal-subheader {{
        color: {ORANGE};
        border-bottom: 1px solid {ORANGE};
        padding-bottom: 0.25rem;
        margin: 0.9rem 0 0.55rem 0;
        font-size: 0.95rem;
        font-weight: 700;
    }}

    .panel {{
        border: 1px solid #333;
        background: {PANEL};
        padding: 0.8rem;
    }}

    .report-box {{
        border: 1px solid #333;
        border-left: 4px solid {ORANGE};
        padding: 1rem 1.1rem;
        background: {PANEL};
        line-height: 1.6;
        color: {TEXT};
    }}

    .signal-box {{
        border: 1px solid #3a3a3a;
        border-left: 5px solid {ORANGE};
        padding: 0.9rem 1rem;
        background: {PANEL_2};
        margin-bottom: 0.8rem;
    }}

    .small-note {{ color: {MUTED}; font-size: 0.82rem; }}

    .regime-badge {{
        padding: 0.55rem 0.8rem;
        font-size: 1.35rem;
        font-weight: 900;
        text-align: center;
        border: 1px solid #444;
        background: {PANEL};
    }}

    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid #303030;
        border-radius: 2px;
        padding: 0.7rem;
    }}

    div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; }}

    button[kind="primary"] {{
        background: {ORANGE} !important;
        color: #000 !important;
        border: 1px solid {ORANGE} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: #080808;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {PANEL};
        border: 1px solid #292929;
        color: {TEXT};
        border-radius: 0;
    }}

    .stTabs [aria-selected="true"] {{
        background: {ORANGE} !important;
        color: #000 !important;
    }}

    div[data-testid="stDataFrame"] {{ border: 1px solid #303030; }}
    hr {{ border-color: #2b2b2b; }}
    code {{ color: {ORANGE}; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# UNIVERSO DATI
# =============================================================================

EQUITY_INDICES: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "RUSSELL 2000": "^RUT",
    "VIX": "^VIX",
    "EURO STOXX 50": "^STOXX50E",
    "FTSE MIB": "FTSEMIB.MI",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "FTSE 100": "^FTSE",
    "SMI": "^SSMI",
    "NIKKEI 225": "^N225",
    "HANG SENG": "^HSI",
    "SHANGHAI": "000001.SS",
    "SENSEX": "^BSESN",
    "KOSPI": "^KS11",
}

FX_UNIVERSE: Dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "USD/CHF": "CHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CNH": "CNH=X",
}

COMMODITY_UNIVERSE: Dict[str, str] = {
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F",
    "NATURAL GAS": "NG=F",
}

CRYPTO_UNIVERSE: Dict[str, str] = {
    "BITCOIN": "BTC-USD",
    "ETHEREUM": "ETH-USD",
}

CREDIT_UNIVERSE: Dict[str, str] = {
    "HIGH YIELD": "HYG",
    "INVESTMENT GRADE": "LQD",
    "EM BONDS": "EMB",
    "US TREASURY 20Y+": "TLT",
}

BOND_PRICE_PROXIES: Dict[str, str] = {
    "US 2Y FUTURE": "ZT=F",
    "US 5Y FUTURE": "ZF=F",
    "US 10Y FUTURE": "ZN=F",
    "US LONG BOND": "ZB=F",
    "BTP 10Y ETF": "BTP10.MI",
    "GERMANY GOVT BOND ETF": "IS0L.DE",
}

# I primi simboli sono preferiti; gli altri sono fallback.
RATE_CANDIDATES: Dict[str, List[str]] = {
    "US 13W": ["^IRX"],
    "US 2Y": ["^AXTWO", "^USTTWO", "TMUBMUSD02Y"],
    "US 5Y": ["^FVX"],
    "US 10Y": ["^TNX"],
    "US 30Y": ["^TYX"],
}

TIMEFRAME_LABELS = {
    "YEARLY": "Annuale",
    "QUARTERLY": "Trimestrale",
    "MONTHLY": "Mensile",
    "WEEKLY": "Settimanale",
}


# =============================================================================
# MODELLI DATI
# =============================================================================

@dataclass
class MarketRow:
    name: str
    ticker: str
    last: float
    change_1d: float
    change_1w: float
    change_1m: float
    change_3m: float
    date: pd.Timestamp


@dataclass
class RegimeSignal:
    name: str
    value: Optional[float]
    bullish: Optional[bool]
    weight: float
    explanation: str


# =============================================================================
# DOWNLOAD ROBUSTO YAHOO FINANCE
# =============================================================================

def _flatten_download(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Estrae il Close da un download yfinance mono o multi ticker."""
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.sort_index().dropna(how="all")


@st.cache_data(ttl=600, show_spinner=False)
def download_close_batch(
    tickers_tuple: Tuple[str, ...],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download multi-ticker con fallback ticker-per-ticker.

    Il fallback evita che un singolo simbolo non disponibile blocchi l'intera
    sezione Global Macro.
    """
    tickers = list(dict.fromkeys(tickers_tuple))
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
            timeout=15,
        )
        close = _flatten_download(raw, tickers)
    except Exception:
        close = pd.DataFrame()

    missing = [ticker for ticker in tickers if ticker not in close.columns or close[ticker].dropna().empty]

    for ticker in missing:
        series = None
        for attempt in range(2):
            try:
                raw_single = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=12,
                )
                extracted = _flatten_download(raw_single, [ticker])
                if ticker in extracted.columns and not extracted[ticker].dropna().empty:
                    series = extracted[ticker]
                    break
            except Exception:
                pass

            if attempt == 0:
                sleep(0.35)

        if series is not None:
            close = close.join(series.rename(ticker), how="outer") if not close.empty else series.to_frame(ticker)

    return close.sort_index().dropna(how="all")


@st.cache_data(ttl=600, show_spinner=False)
def resolve_rate_series(period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Trova il primo ticker Yahoo funzionante per ogni scadenza Treasury."""
    resolved: Dict[str, pd.Series] = {}
    symbols: Dict[str, str] = {}

    all_candidates = tuple(
        dict.fromkeys(
            ticker
            for candidates in RATE_CANDIDATES.values()
            for ticker in candidates
        )
    )
    downloaded = download_close_batch(all_candidates, period=period)

    for label, candidates in RATE_CANDIDATES.items():
        for ticker in candidates:
            if ticker in downloaded.columns:
                series = downloaded[ticker].dropna()
                if len(series) >= 2:
                    resolved[label] = series
                    symbols[label] = ticker
                    break

    if not resolved:
        return pd.DataFrame(), {}

    return pd.DataFrame(resolved).sort_index(), symbols


# =============================================================================
# CALCOLI GENERALI
# =============================================================================

def safe_pct_change(series: pd.Series, sessions: int) -> float:
    clean = series.dropna()
    if len(clean) <= sessions:
        return np.nan
    return (float(clean.iloc[-1]) / float(clean.iloc[-1 - sessions]) - 1.0) * 100.0


def build_market_table(close: pd.DataFrame, universe: Dict[str, str]) -> pd.DataFrame:
    rows: List[dict] = []

    for name, ticker in universe.items():
        if ticker not in close.columns:
            continue

        series = close[ticker].dropna()
        if len(series) < 2:
            continue

        rows.append(
            {
                "Strumento": name,
                "Ticker": ticker,
                "Ultimo": float(series.iloc[-1]),
                "1D %": safe_pct_change(series, 1),
                "1W %": safe_pct_change(series, 5),
                "1M %": safe_pct_change(series, 21),
                "3M %": safe_pct_change(series, 63),
                "Data": series.index[-1],
            }
        )

    return pd.DataFrame(rows)


def latest_change_bp(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    return (float(clean.iloc[-1]) - float(clean.iloc[-2])) * 100.0


def normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.ffill().dropna(how="all")
    if clean.empty:
        return clean

    result = pd.DataFrame(index=clean.index)
    for column in clean.columns:
        series = clean[column].dropna()
        if series.empty or float(series.iloc[0]) == 0:
            continue
        result[column] = clean[column] / float(series.iloc[0]) * 100.0
    return result


def ratio_series(close: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    if numerator not in close.columns or denominator not in close.columns:
        return pd.Series(dtype=float)
    ratio = close[numerator] / close[denominator]
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


# =============================================================================
# LAYOUT PLOTLY
# =============================================================================

def apply_terminal_layout(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Consolas, Courier New, monospace"),
        margin=dict(l=25, r=25, t=55, b=25),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#444")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#444")
    return fig


def create_line_chart(
    frame: pd.DataFrame,
    title: str,
    y_title: str = "",
    height: int = 480,
) -> go.Figure:
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=str(column),
                line=dict(width=1.8),
            )
        )
    fig.update_layout(title=title, yaxis_title=y_title)
    return apply_terminal_layout(fig, height)


def create_bar_chart(table: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    data = table.dropna(subset=[value_column]).sort_values(value_column)
    colors = [GREEN if value >= 0 else RED for value in data[value_column]]

    fig = go.Figure(
        go.Bar(
            x=data[value_column],
            y=data["Strumento"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.2f}%" for value in data[value_column]],
            textposition="outside",
        )
    )
    fig.update_layout(title=title, xaxis_title=value_column)
    return apply_terminal_layout(fig, 500)


def create_yield_curve_chart(rates: pd.DataFrame) -> go.Figure:
    order = ["US 13W", "US 2Y", "US 5Y", "US 10Y", "US 30Y"]
    available = [label for label in order if label in rates.columns and not rates[label].dropna().empty]

    current = [float(rates[label].dropna().iloc[-1]) for label in available]

    previous = []
    for label in available:
        series = rates[label].dropna()
        previous.append(float(series.iloc[-6]) if len(series) >= 6 else np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=available,
            y=current,
            mode="lines+markers",
            name="Current",
            line=dict(color=ORANGE, width=3),
            marker=dict(size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=available,
            y=previous,
            mode="lines+markers",
            name="5 sessions ago",
            line=dict(color=BLUE, width=1.8, dash="dash"),
        )
    )
    fig.update_layout(title="US TREASURY YIELD CURVE", yaxis_title="Yield %")
    return apply_terminal_layout(fig, 440)
def create_shipping_risk_gauge(value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "SHIPPING RISK INDEX"},
            gauge={
                "axis": {"range": [0,100]},
                "bar": {"color": ORANGE},

                "steps": [
                    {"range":[0,25],"color":"#0d3f1f"},
                    {"range":[25,50],"color":"#274d15"},
                    {"range":[50,75],"color":"#5b4700"},
                    {"range":[75,100],"color":"#5f1111"}
                ],
            }
        )
    )

    return apply_terminal_layout(fig, 320)
def create_hormuz_map(ships):

    color_map = {
        "Crude Tanker": ORANGE,
        "LNG": CYAN,
        "Container": BLUE,
        "Bulk": PURPLE,
    }

    fig = px.scatter_mapbox(
        ships,
        lat="lat",
        lon="lon",
        color="Type",
        hover_name="Ship",
        color_discrete_map=color_map,
        zoom=6,
        height=550,
    )

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
def build_shipping_comment(latest, avg30):

    delta = ((latest/avg30)-1)*100

    risk = "LOW"

    if delta < -5:
        risk = "ELEVATED"

    if delta < -10:
        risk = "HIGH"

    return (
        f"Hormuz traffic is {delta:+.1f}% versus "
        f"the 30-day average. "

        f"Current flow regime suggests "
        f"{risk.lower()} logistics risk. "

        f"Persistent weakness in crude tanker "
        f"traffic historically coincides with "
        f"tighter energy market conditions and "
        f"higher Brent sensitivity."
    )
# =============================================================================
# HEADER E TICKER STRIP
# =============================================================================

def fmt_change(value: float) -> str:
    if pd.isna(value):
        return "N/D"
    return f"{value:+.2f}%"


def render_top_bar() -> None:
    now = datetime.now(ZoneInfo("Europe/Rome"))
    st.markdown(
        f"<div class='top-terminal-bar'>"
        f"<span>CYCLICAL GLOBAL MACRO TERMINAL</span>"
        f"<span>{now.strftime('%A %d %B %Y // %H:%M CET')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    strip_universe = {
        "S&P": "^GSPC",
        "NASDAQ": "^IXIC",
        "DAX": "^GDAXI",
        "MIB": "FTSEMIB.MI",
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "WTI": "CL=F",
        "BTC": "BTC-USD",
    }

    close = download_close_batch(tuple(strip_universe.values()), period="1mo")
    items = []

    for name, ticker in strip_universe.items():
        if ticker not in close.columns or close[ticker].dropna().empty:
            items.append(f"<span>{name} <b style='color:{MUTED}'>N/D</b></span>")
            continue

        series = close[ticker].dropna()
        last = float(series.iloc[-1])
        change = safe_pct_change(series, 1)
        color = GREEN if change >= 0 else RED
        items.append(
            f"<span>{name} {last:,.2f} "
            f"<b style='color:{color}'>{fmt_change(change)}</b></span>"
        )

    st.markdown(
        f"<div class='ticker-strip'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# COMPONENTI TABELLA
# =============================================================================

def style_market_table(table: pd.DataFrame):
    display = table.copy()
    if "Data" in display.columns:
        display["Data"] = pd.to_datetime(display["Data"]).dt.strftime("%d/%m/%Y")

    formatters = {
        "Ultimo": "{:,.4f}",
        "1D %": "{:+.2f}%",
        "1W %": "{:+.2f}%",
        "1M %": "{:+.2f}%",
        "3M %": "{:+.2f}%",
    }

    existing_formatters = {key: value for key, value in formatters.items() if key in display.columns}
    styled = display.style.format(existing_formatters, na_rep="N/D")

    change_columns = [column for column in ["1D %", "1W %", "1M %", "3M %"] if column in display.columns]
    if change_columns:
        styled = styled.map(
            lambda value: (
                f"color:{GREEN};font-weight:700"
                if isinstance(value, (int, float, np.floating)) and value >= 0
                else f"color:{RED};font-weight:700"
            ),
            subset=change_columns,
        )

    return styled.set_properties(
        **{
            "background-color": PANEL,
            "color": TEXT,
            "border-color": "#333",
        }
    )


# =============================================================================
# GLOBAL OVERVIEW
# =============================================================================

def render_global_overview() -> None:
    st.markdown(
        "<div class='terminal-header'>GLOBAL OVERVIEW // EQUITY INDICES</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Aggiornamento indici globali..."):
        close = download_close_batch(tuple(EQUITY_INDICES.values()), period="6mo")

    table = build_market_table(close, EQUITY_INDICES)
    if table.empty:
        st.error("Yahoo Finance non ha restituito dati per gli indici.")
        return

    card_names = ["S&P 500", "NASDAQ", "FTSE MIB", "DAX", "NIKKEI 225", "VIX","KOSPI"]
    indexed = table.set_index("Strumento")
    cols = st.columns(7)
    for col, name in zip(cols, card_names):
        if name not in indexed.index:
            col.metric(name, "N/D")
            continue
        row = indexed.loc[name]
        col.metric(name, f"{row['Ultimo']:,.2f}", f"{row['1D %']:+.2f}%")

    left, right = st.columns([2.1, 1])

    with left:
        st.markdown("<div class='terminal-subheader'>RELATIVE PERFORMANCE</div>", unsafe_allow_html=True)
        reverse = {ticker: name for name, ticker in EQUITY_INDICES.items()}
        renamed = close.rename(columns=reverse)
        defaults = [name for name in ["S&P 500", "NASDAQ", "EURO STOXX 50", "FTSE MIB", "DAX", "NIKKEI 225","KOSPI"] if name in renamed.columns]
        selected = st.multiselect(
            "Indici",
            options=list(renamed.columns),
            default=defaults,
            label_visibility="collapsed",
            key="overview_indices",
        )
        if selected:
            chart = normalized_frame(renamed[selected])
            st.plotly_chart(
                create_line_chart(chart, "GLOBAL EQUITY // BASE 100", "Base 100", 510),
                use_container_width=True,
            )

    with right:
        st.markdown("<div class='terminal-subheader'>1D PERFORMANCE</div>", unsafe_allow_html=True)
        st.plotly_chart(
            create_bar_chart(table, "1D %", "LEADERS / LAGGARDS"),
            use_container_width=True,
        )

    st.markdown("<div class='terminal-subheader'>WORLD INDEX TABLE</div>", unsafe_allow_html=True)
    st.dataframe(style_market_table(table), use_container_width=True, hide_index=True, height=570)


# =============================================================================
# GLOBAL MACRO
# =============================================================================

def build_macro_comment(
    rates: pd.DataFrame,
    fx_table: pd.DataFrame,
    commodity_table: pd.DataFrame,
    credit_close: pd.DataFrame,
) -> str:
    sentences: List[str] = []

    if "US 10Y" in rates.columns and len(rates["US 10Y"].dropna()) >= 2:
        s = rates["US 10Y"].dropna()
        change_bp = latest_change_bp(s)
        direction = "in aumento" if change_bp > 0 else "in calo"
        sentences.append(
            f"Il Treasury decennale è {direction} di {abs(change_bp):.1f} punti base nell'ultima seduta, "
            f"con rendimento a {float(s.iloc[-1]):.2f}%."
        )

    dxy = fx_table.loc[fx_table["Strumento"] == "DXY"] if not fx_table.empty else pd.DataFrame()
    if not dxy.empty and pd.notna(dxy.iloc[0]["1M %"]):
        value = float(dxy.iloc[0]["1M %"])
        sentences.append(
            f"Il dollaro mostra una variazione mensile del {value:+.2f}%, "
            + ("segnalando condizioni finanziarie più restrittive." if value > 0 else "riducendo parzialmente la pressione sulle condizioni finanziarie globali.")
        )

    oil = commodity_table.loc[commodity_table["Strumento"] == "WTI"] if not commodity_table.empty else pd.DataFrame()
    copper = commodity_table.loc[commodity_table["Strumento"] == "COPPER"] if not commodity_table.empty else pd.DataFrame()
    if not oil.empty and pd.notna(oil.iloc[0]["1M %"]):
        sentences.append(f"Il WTI registra una performance mensile del {float(oil.iloc[0]['1M %']):+.2f}%.")
    if not copper.empty and pd.notna(copper.iloc[0]["1M %"]):
        sentences.append(f"Il rame, indicatore ciclico industriale, varia del {float(copper.iloc[0]['1M %']):+.2f}% su un mese.")

    hy_ratio = ratio_series(credit_close, "HYG", "LQD")
    if len(hy_ratio) > 21:
        change = safe_pct_change(hy_ratio, 21)
        sentences.append(
            "Il rapporto High Yield/Investment Grade è "
            + (f"in miglioramento del {change:+.2f}% su un mese, coerente con propensione al rischio." if change > 0 else f"in deterioramento del {change:+.2f}% su un mese, coerente con maggiore prudenza sul credito.")
        )

    if not sentences:
        return "Il quadro macro non è determinabile perché alcune serie Yahoo Finance non sono disponibili."

    return " ".join(sentences)


def render_rates_section() -> None:
    st.markdown("<div class='terminal-subheader'>RATES // US TREASURY YIELDS</div>", unsafe_allow_html=True)

    rates, symbols = resolve_rate_series(period="2y")
    if rates.empty:
        st.warning("Le serie dei rendimenti Treasury non sono disponibili su Yahoo Finance in questo momento.")
        return

    ordered = [label for label in ["US 13W", "US 2Y", "US 5Y", "US 10Y", "US 30Y"] if label in rates.columns]
    cols = st.columns(len(ordered))
    for col, label in zip(cols, ordered):
        series = rates[label].dropna()
        col.metric(
            label,
            f"{float(series.iloc[-1]):.3f}%",
            f"{latest_change_bp(series):+.1f} bp",
        )

    left, right = st.columns([1, 1.5])
    with left:
        st.plotly_chart(create_yield_curve_chart(rates), use_container_width=True)

    with right:
        st.plotly_chart(
            create_line_chart(rates[ordered], "TREASURY YIELDS // HISTORY", "Yield %", 440),
            use_container_width=True,
        )

    spreads = pd.DataFrame(index=rates.index)
    if "US 10Y" in rates.columns and "US 2Y" in rates.columns:
        spreads["10Y-2Y"] = (rates["US 10Y"] - rates["US 2Y"]) * 100.0
    if "US 10Y" in rates.columns and "US 13W" in rates.columns:
        spreads["10Y-13W"] = (rates["US 10Y"] - rates["US 13W"]) * 100.0
    if "US 30Y" in rates.columns and "US 5Y" in rates.columns:
        spreads["30Y-5Y"] = (rates["US 30Y"] - rates["US 5Y"]) * 100.0

    if not spreads.empty:
        st.plotly_chart(
            create_line_chart(spreads, "US CURVE SPREADS", "Basis points", 390),
            use_container_width=True,
        )

    symbol_text = ", ".join(f"{label}: {ticker}" for label, ticker in symbols.items())
    st.markdown(
        f"<div class='small-note'>Ticker Yahoo risolti: {symbol_text}. "
        "I valori sono rendimenti percentuali quando il ticker Yahoo rappresenta un indice di rendimento.</div>",
        unsafe_allow_html=True,
    )


def render_bond_proxies() -> None:
    st.markdown("<div class='terminal-subheader'>SOVEREIGN BOND PRICE PROXIES</div>", unsafe_allow_html=True)
    close = download_close_batch(tuple(BOND_PRICE_PROXIES.values()), period="1y")
    table = build_market_table(close, BOND_PRICE_PROXIES)

    if table.empty:
        st.warning("Proxy obbligazionari non disponibili.")
        return

    reverse = {ticker: name for name, ticker in BOND_PRICE_PROXIES.items()}
    renamed = close.rename(columns=reverse)
    selected = [name for name in BOND_PRICE_PROXIES if name in renamed.columns]
    normalized = normalized_frame(renamed[selected])

    left, right = st.columns([1.7, 1])
    with left:
        st.plotly_chart(
            create_line_chart(normalized, "BOND PRICES / FUTURES // BASE 100", "Base 100", 450),
            use_container_width=True,
        )
    with right:
        st.dataframe(style_market_table(table), use_container_width=True, hide_index=True, height=450)

    st.warning(
        "BTP10.MI e IS0L.DE sono proxy di prezzo tramite ETF, non rendimenti benchmark. "
        "Per questo il terminale non calcola un falso spread BTP-Bund partendo da questi prezzi. "
        "Un vero spread richiede rendimento BTP 10Y meno rendimento Bund 10Y."
    )


def render_global_macro() -> None:
    st.markdown("<div class='terminal-header'>GLOBAL MACRO // RATES, FX, COMMODITIES, CREDIT</div>", unsafe_allow_html=True)

    render_rates_section()
    render_bond_proxies()

    macro_tickers = tuple(
        list(FX_UNIVERSE.values())
        + list(COMMODITY_UNIVERSE.values())
        + list(CRYPTO_UNIVERSE.values())
        + list(CREDIT_UNIVERSE.values())
    )

    with st.spinner("Aggiornamento macro assets..."):
        close = download_close_batch(macro_tickers, period="1y")

    fx_table = build_market_table(close, FX_UNIVERSE)
    commodity_table = build_market_table(close, COMMODITY_UNIVERSE)
    crypto_table = build_market_table(close, CRYPTO_UNIVERSE)
    credit_table = build_market_table(close, CREDIT_UNIVERSE)

    tabs = st.tabs(["FX", "COMMODITIES", "CRYPTO", "CREDIT"])

    with tabs[0]:
        if fx_table.empty:
            st.warning("Dati FX non disponibili.")
        else:
            left, right = st.columns([1.6, 1])
            with left:
                reverse = {ticker: name for name, ticker in FX_UNIVERSE.items()}
                data = normalized_frame(close.rename(columns=reverse)[[name for name in reverse.values() if name in close.rename(columns=reverse).columns]])
                st.plotly_chart(create_line_chart(data, "FX PERFORMANCE // BASE 100", "Base 100", 470), use_container_width=True)
            with right:
                st.dataframe(style_market_table(fx_table), use_container_width=True, hide_index=True, height=470)

    with tabs[1]:
        if commodity_table.empty:
            st.warning("Dati commodity non disponibili.")
        else:
            left, right = st.columns([1.6, 1])
            with left:
                reverse = {ticker: name for name, ticker in COMMODITY_UNIVERSE.items()}
                renamed = close.rename(columns=reverse)
                columns = [name for name in COMMODITY_UNIVERSE if name in renamed.columns]
                st.plotly_chart(create_line_chart(normalized_frame(renamed[columns]), "COMMODITIES // BASE 100", "Base 100", 470), use_container_width=True)
            with right:
                st.dataframe(style_market_table(commodity_table), use_container_width=True, hide_index=True, height=470)

    with tabs[2]:
        if crypto_table.empty:
            st.warning("Dati crypto non disponibili.")
        else:
            reverse = {ticker: name for name, ticker in CRYPTO_UNIVERSE.items()}
            renamed = close.rename(columns=reverse)
            columns = [name for name in CRYPTO_UNIVERSE if name in renamed.columns]
            st.plotly_chart(create_line_chart(normalized_frame(renamed[columns]), "CRYPTO // BASE 100", "Base 100", 470), use_container_width=True)
            st.dataframe(style_market_table(crypto_table), use_container_width=True, hide_index=True)

    with tabs[3]:
        if credit_table.empty:
            st.warning("Dati credit proxy non disponibili.")
        else:
            left, right = st.columns([1.5, 1])
            with left:
                hy_ig = ratio_series(close, "HYG", "LQD")
                risk_treasury = ratio_series(close, "HYG", "TLT")
                ratios = pd.DataFrame({"HYG/LQD": hy_ig, "HYG/TLT": risk_treasury}).dropna(how="all")
                st.plotly_chart(create_line_chart(normalized_frame(ratios), "CREDIT RISK RATIOS // BASE 100", "Base 100", 470), use_container_width=True)
            with right:
                st.dataframe(style_market_table(credit_table), use_container_width=True, hide_index=True, height=470)

    rates, _ = resolve_rate_series(period="6mo")
    comment = build_macro_comment(rates, fx_table, commodity_table, close)
    st.markdown("<div class='terminal-subheader'>MACRO STRATEGIST COMMENT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-box'>{comment}</div>", unsafe_allow_html=True)
def render_shipping():

    st.markdown(
        "<div class='terminal-header'>GLOBAL SHIPPING & ENERGY FLOWS</div>",
        unsafe_allow_html=True
    )

    traffic, ships = get_shipping_data()

    latest = float(traffic["Hormuz"].iloc[-1])
    avg30 = float(traffic["Hormuz"].tail(30).mean())

    crude = 54
    lng = 18

    risk_index = min(
        100,
        max(
            10,
            int(50 + (avg30-latest)*2)
        )
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "HORMUZ TRANSITS",
        f"{latest:.0f}",
        f"{((latest/avg30)-1)*100:+.1f}%"
    )

    c2.metric(
        "CRUDE TANKERS",
        crude
    )

    c3.metric(
        "LNG CARRIERS",
        lng
    )

    c4.metric(
        "RISK INDEX",
        risk_index
    )

    left,right = st.columns([2,1])

    with left:

        st.plotly_chart(
            create_hormuz_map(ships),
            use_container_width=True
        )

    with right:

        st.plotly_chart(
            create_shipping_risk_gauge(risk_index),
            use_container_width=True
        )

    st.markdown(
        "<div class='terminal-subheader'>HORMUZ TRAFFIC TREND</div>",
        unsafe_allow_html=True
    )

    trend = go.Figure()

    trend.add_trace(
        go.Scatter(
            x=traffic["Date"],
            y=traffic["Hormuz"],
            name="Traffic",
            line=dict(color=ORANGE,width=2)
        )
    )

    trend.add_trace(
        go.Scatter(
            x=traffic["Date"],
            y=traffic["Hormuz"].rolling(30).mean(),
            name="30D Average",
            line=dict(color=BLUE,width=2)
        )
    )

    st.plotly_chart(
        apply_terminal_layout(
            trend,
            420
        ),
        use_container_width=True
    )

    comment = build_shipping_comment(
        latest,
        avg30
    )

    st.markdown(
        "<div class='terminal-subheader'>SHIPPING STRATEGIST COMMENT</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='report-box'>{comment}</div>",
        unsafe_allow_html=True
    )
# =============================================================================
# SHIPPING & ENERGY FLOWS
# =============================================================================

@st.cache_data(ttl=3600)
def get_shipping_data():

    np.random.seed(42)

    dates = pd.date_range(
        end=pd.Timestamp.today(),
        periods=180,
        freq="D"
    )

    traffic = pd.DataFrame({
        "Date": dates,
        "Hormuz": np.random.normal(115, 8, len(dates)).cumsum()/20,
    })

    traffic["Hormuz"] = (
        120
        + np.sin(np.arange(len(dates))/15)*10
        + np.random.normal(0, 4, len(dates))
    )

    ships = pd.DataFrame({
        "Ship": [
            "VLCC Alpha",
            "LNG Falcon",
            "Box Asia",
            "Bulk Star",
            "VLCC Titan",
            "LNG Horizon",
        ],

        "Type": [
            "Crude Tanker",
            "LNG",
            "Container",
            "Bulk",
            "Crude Tanker",
            "LNG",
        ],

        "lat": [
            26.40,
            26.55,
            26.20,
            26.75,
            26.10,
            26.35,
        ],

        "lon": [
            56.10,
            56.30,
            56.45,
            56.00,
            56.60,
            56.20,
        ]
    })

    return traffic, ships
# =============================================================================
# MARKET REGIME V3 - STRUCTURAL / TACTICAL / DAILY
# =============================================================================

REGIME_UNIVERSE = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "ACWI": "ACWI",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "HYG": "HYG",
    "LQD": "LQD",
    "TLT": "TLT",
    "COPPER": "HG=F",
    "GOLD": "GC=F",
    "US_13W": "^IRX",
    "US_10Y": "^TNX",
}


@dataclass
class RegimePillar:
    name: str
    score: float
    state: str
    details: str


@dataclass
class RegimeLayer:
    key: str
    title: str
    horizon: str
    diagnosis: str
    score: float
    previous_diagnosis: str
    previous_score: float
    pillars: List[RegimePillar]

    @property
    def transition(self) -> str:
        if self.diagnosis == self.previous_diagnosis:
            return f"{self.previous_diagnosis} → {self.diagnosis}"
        return f"{self.previous_diagnosis} → {self.diagnosis}"


def _series(close: pd.DataFrame, ticker: str) -> pd.Series:
    if ticker not in close.columns:
        return pd.Series(dtype=float)
    return close[ticker].dropna().astype(float)


def _last(close: pd.DataFrame, ticker: str) -> Optional[float]:
    series = _series(close, ticker)
    return None if series.empty else float(series.iloc[-1])


def _return(close: pd.DataFrame, ticker: str, sessions: int) -> Optional[float]:
    series = _series(close, ticker)
    if len(series) <= sessions:
        return None
    return float((series.iloc[-1] / series.iloc[-1 - sessions] - 1.0) * 100.0)


def _ratio_return(
    close: pd.DataFrame,
    numerator: str,
    denominator: str,
    sessions: int,
) -> Optional[float]:
    ratio = ratio_series(close, numerator, denominator).dropna()
    if len(ratio) <= sessions:
        return None
    return float((ratio.iloc[-1] / ratio.iloc[-1 - sessions] - 1.0) * 100.0)


def _yield_change_bps(close: pd.DataFrame, ticker: str, sessions: int) -> Optional[float]:
    """Per ^TNX e ^IRX, una variazione di 0,10 equivale circa a 1 bp."""
    series = _series(close, ticker)
    if len(series) <= sessions:
        return None
    return float((series.iloc[-1] - series.iloc[-1 - sessions]) * 10.0)


def _moving_average_distance(close: pd.DataFrame, ticker: str, window: int) -> Optional[float]:
    series = _series(close, ticker)
    if len(series) < window:
        return None
    average = float(series.rolling(window).mean().iloc[-1])
    if average == 0:
        return None
    return float((series.iloc[-1] / average - 1.0) * 100.0)


def _safe_mean(values: Iterable[Optional[float]]) -> float:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    return float(np.mean(clean)) if clean else 0.0


def _clip_score(value: float) -> float:
    return float(np.clip(value, -2.0, 2.0))


def _state_5(score: float) -> str:
    if score >= 1.20:
        return "STRONGLY POSITIVE"
    if score >= 0.35:
        return "POSITIVE"
    if score <= -1.20:
        return "STRONGLY NEGATIVE"
    if score <= -0.35:
        return "NEGATIVE"
    return "NEUTRAL"


def _regime_color(label: str) -> str:
    upper = label.upper()
    if any(word in upper for word in ["CONSTRUCTIVE", "IMPROVING", "RISK-ON", "POSITIVE"]):
        return GREEN
    if any(word in upper for word in ["DEFENSIVE", "DETERIORATING", "RISK-OFF", "NEGATIVE"]):
        return RED
    return ORANGE


# -----------------------------------------------------------------------------
# 1) STRUCTURAL BACKDROP
# -----------------------------------------------------------------------------

def _strategic_equity(close: pd.DataFrame) -> RegimePillar:
    spy_6m = _return(close, "SPY", 126)
    qqq_6m = _return(close, "QQQ", 126)
    acwi_6m = _return(close, "ACWI", 126)
    spy_ma200 = _moving_average_distance(close, "SPY", 200)
    acwi_ma200 = _moving_average_distance(close, "ACWI", 200)

    score = 0.0
    score += 0.55 if (spy_6m or 0) > 5 else -0.55 if (spy_6m or 0) < -5 else 0.0
    score += 0.45 if (qqq_6m or 0) > 5 else -0.45 if (qqq_6m or 0) < -5 else 0.0
    score += 0.40 if (acwi_6m or 0) > 3 else -0.40 if (acwi_6m or 0) < -3 else 0.0
    score += 0.35 if (spy_ma200 or 0) > 0 else -0.35
    score += 0.25 if (acwi_ma200 or 0) > 0 else -0.25
    score = _clip_score(score)

    details = (
        f"SPY 6M {spy_6m:+.2f}% | QQQ 6M {qqq_6m:+.2f}% | "
        f"ACWI 6M {acwi_6m:+.2f}% | SPY vs MM200 {spy_ma200:+.2f}%"
        if None not in (spy_6m, qqq_6m, acwi_6m, spy_ma200)
        else "Dati equity strutturali parziali"
    )
    return RegimePillar("EQUITY", score, _state_5(score), details)


def _strategic_volatility(close: pd.DataFrame) -> RegimePillar:
    vix = _series(close, "^VIX")
    if vix.empty:
        return RegimePillar("VOLATILITY", 0.0, "NEUTRAL", "VIX non disponibile")

    level = float(vix.iloc[-1])
    ma63 = float(vix.rolling(63).mean().iloc[-1]) if len(vix) >= 63 else level
    percentile = float((vix.tail(252) <= level).mean() * 100.0)

    score = 0.0
    score += 0.80 if level < 17 else 0.25 if level < 22 else -0.75 if level < 30 else -1.50
    score += 0.55 if level < ma63 else -0.55
    score += 0.35 if percentile < 40 else -0.35 if percentile > 70 else 0.0
    score = _clip_score(score)

    details = f"VIX {level:.2f} | MM63 {ma63:.2f} | percentile 1Y {percentile:.0f}"
    return RegimePillar("VOLATILITY", score, _state_5(score), details)


def _strategic_credit(close: pd.DataFrame) -> RegimePillar:
    ratio_6m = _ratio_return(close, "HYG", "LQD", 126)
    ratio_ma200 = None
    ratio = ratio_series(close, "HYG", "LQD").dropna()
    if len(ratio) >= 200:
        ratio_ma200 = float((ratio.iloc[-1] / ratio.rolling(200).mean().iloc[-1] - 1.0) * 100.0)
    hyg_6m = _return(close, "HYG", 126)

    score = 0.0
    score += 0.85 if (ratio_6m or 0) > 1.5 else -0.85 if (ratio_6m or 0) < -1.5 else 0.0
    score += 0.65 if (ratio_ma200 or 0) > 0 else -0.65
    score += 0.35 if (hyg_6m or 0) > 0 else -0.35
    score = _clip_score(score)

    details = (
        f"HYG/LQD 6M {ratio_6m:+.2f}% | vs MM200 {ratio_ma200:+.2f}% | HYG 6M {hyg_6m:+.2f}%"
        if None not in (ratio_6m, ratio_ma200, hyg_6m)
        else "Dati credito strutturali parziali"
    )
    return RegimePillar("CREDIT", score, _state_5(score), details)


def _strategic_rates(close: pd.DataFrame) -> RegimePillar:
    ten = _last(close, "^TNX")
    bill = _last(close, "^IRX")
    move_3m = _yield_change_bps(close, "^TNX", 63)
    curve = (ten - bill) * 10.0 if ten is not None and bill is not None else None

    score = 0.0
    if move_3m is not None:
        score += 0.70 if move_3m <= -20 else 0.25 if move_3m < 10 else -0.60 if move_3m < 30 else -1.10
    if curve is not None:
        score += 0.45 if curve > 25 else -0.65 if curve < -50 else -0.20 if curve < 0 else 0.15
    if ten is not None:
        score += 0.25 if ten < 35 else -0.25 if ten > 45 else 0.0
    score = _clip_score(score)

    details = (
        f"US10Y {ten / 10:.2f}% | 3M {move_3m:+.1f} bp | 10Y-13W {curve:+.1f} bp"
        if None not in (ten, move_3m, curve)
        else "Dati tassi strutturali parziali"
    )
    return RegimePillar("RATES", score, _state_5(score), details)


def _strategic_macro(close: pd.DataFrame) -> RegimePillar:
    copper_gold_6m = _ratio_return(close, "HG=F", "GC=F", 126)
    dxy_6m = _return(close, "DX-Y.NYB", 126)

    score = 0.0
    score += 0.90 if (copper_gold_6m or 0) > 5 else -0.90 if (copper_gold_6m or 0) < -5 else 0.0
    score += 0.55 if (dxy_6m or 0) < -3 else -0.55 if (dxy_6m or 0) > 3 else 0.0
    score = _clip_score(score)

    details = (
        f"Copper/Gold 6M {copper_gold_6m:+.2f}% | DXY 6M {dxy_6m:+.2f}%"
        if None not in (copper_gold_6m, dxy_6m)
        else "Dati macro strutturali parziali"
    )
    return RegimePillar("MACRO", score, _state_5(score), details)


def _strategic_diagnosis(score: float, pillars: List[RegimePillar]) -> str:
    rates = next(p.score for p in pillars if p.name == "RATES")
    credit = next(p.score for p in pillars if p.name == "CREDIT")
    volatility = next(p.score for p in pillars if p.name == "VOLATILITY")

    if score >= 0.60:
        if rates <= -0.50:
            return "CONSTRUCTIVE — RATES-CONSTRAINED"
        return "CONSTRUCTIVE"
    if score <= -0.60:
        if credit <= -0.60 or volatility <= -0.80:
            return "DEFENSIVE — FINANCIAL STRESS"
        return "DEFENSIVE"
    if rates <= -0.70 and credit >= 0:
        return "NEUTRAL — RATES HEADWIND"
    return "NEUTRAL / TRANSITION"


def compute_strategic_layer(close: pd.DataFrame) -> Tuple[float, str, List[RegimePillar]]:
    pillars = [
        _strategic_equity(close),
        _strategic_volatility(close),
        _strategic_credit(close),
        _strategic_rates(close),
        _strategic_macro(close),
    ]
    weights = {"EQUITY": 1.35, "VOLATILITY": 1.00, "CREDIT": 1.30, "RATES": 0.90, "MACRO": 0.75}
    score = sum(p.score * weights[p.name] for p in pillars) / sum(weights.values())
    return score, _strategic_diagnosis(score, pillars), pillars


# -----------------------------------------------------------------------------
# 2) TACTICAL DIRECTION
# -----------------------------------------------------------------------------

def _momentum_acceleration(close: pd.DataFrame, ticker: str, recent: int = 21) -> Optional[float]:
    series = _series(close, ticker)
    if len(series) <= recent * 2:
        return None
    current = (series.iloc[-1] / series.iloc[-1 - recent] - 1.0) * 100.0
    previous = (series.iloc[-1 - recent] / series.iloc[-1 - recent * 2] - 1.0) * 100.0
    return float(current - previous)


def _tactical_equity(close: pd.DataFrame) -> RegimePillar:
    spy_1m = _return(close, "SPY", 21)
    qqq_1m = _return(close, "QQQ", 21)
    acwi_1m = _return(close, "ACWI", 21)
    acceleration = _momentum_acceleration(close, "SPY", 21)
    breadth = _safe_mean([spy_1m, qqq_1m, acwi_1m])

    score = 0.0
    score += 0.75 if breadth > 2 else -0.75 if breadth < -2 else breadth / 4.0
    score += 0.65 if (acceleration or 0) > 1.5 else -0.65 if (acceleration or 0) < -1.5 else 0.0
    score += 0.30 if all((v or 0) > 0 for v in [spy_1m, qqq_1m, acwi_1m]) else -0.30 if all((v or 0) < 0 for v in [spy_1m, qqq_1m, acwi_1m]) else 0.0
    score = _clip_score(score)

    details = (
        f"SPY 1M {spy_1m:+.2f}% | QQQ {qqq_1m:+.2f}% | ACWI {acwi_1m:+.2f}% | accelerazione {acceleration:+.2f} pp"
        if None not in (spy_1m, qqq_1m, acwi_1m, acceleration)
        else "Dati equity tattici parziali"
    )
    return RegimePillar("EQUITY", score, _state_5(score), details)


def _tactical_volatility(close: pd.DataFrame) -> RegimePillar:
    level = _last(close, "^VIX")
    one_month = _return(close, "^VIX", 21)
    one_week = _return(close, "^VIX", 5)

    score = 0.0
    if level is not None:
        score += 0.50 if level < 18 else -0.50 if level > 23 else 0.0
    if one_month is not None:
        score += 0.75 if one_month < -10 else -0.75 if one_month > 15 else -0.20 if one_month > 0 else 0.20
    if one_week is not None:
        score += 0.50 if one_week < -8 else -0.50 if one_week > 10 else 0.0
    score = _clip_score(score)

    details = (
        f"VIX {level:.2f} | 1M {one_month:+.2f}% | 1W {one_week:+.2f}%"
        if None not in (level, one_month, one_week)
        else "Dati volatilità tattici parziali"
    )
    return RegimePillar("VOLATILITY", score, _state_5(score), details)


def _tactical_credit(close: pd.DataFrame) -> RegimePillar:
    ratio_1m = _ratio_return(close, "HYG", "LQD", 21)
    ratio_1w = _ratio_return(close, "HYG", "LQD", 5)
    hyg_1m = _return(close, "HYG", 21)

    score = 0.0
    score += 0.90 if (ratio_1m or 0) > 0.7 else -0.90 if (ratio_1m or 0) < -0.7 else 0.0
    score += 0.55 if (ratio_1w or 0) > 0.25 else -0.55 if (ratio_1w or 0) < -0.25 else 0.0
    score += 0.30 if (hyg_1m or 0) > 0 else -0.30
    score = _clip_score(score)

    details = (
        f"HYG/LQD 1M {ratio_1m:+.2f}% | 1W {ratio_1w:+.2f}% | HYG 1M {hyg_1m:+.2f}%"
        if None not in (ratio_1m, ratio_1w, hyg_1m)
        else "Dati credito tattici parziali"
    )
    return RegimePillar("CREDIT", score, _state_5(score), details)


def _tactical_rates(close: pd.DataFrame) -> RegimePillar:
    move_1m = _yield_change_bps(close, "^TNX", 21)
    move_1w = _yield_change_bps(close, "^TNX", 5)

    score = 0.0
    if move_1m is not None:
        score += 0.80 if move_1m <= -15 else -0.80 if move_1m >= 20 else -0.25 if move_1m > 5 else 0.25 if move_1m < -5 else 0.0
    if move_1w is not None:
        score += 0.45 if move_1w <= -8 else -0.45 if move_1w >= 10 else 0.0
    score = _clip_score(score)

    details = (
        f"US10Y 1M {move_1m:+.1f} bp | 1W {move_1w:+.1f} bp"
        if None not in (move_1m, move_1w)
        else "Dati tassi tattici parziali"
    )
    return RegimePillar("RATES", score, _state_5(score), details)


def _tactical_macro(close: pd.DataFrame) -> RegimePillar:
    copper_gold_1m = _ratio_return(close, "HG=F", "GC=F", 21)
    dxy_1m = _return(close, "DX-Y.NYB", 21)

    score = 0.0
    score += 0.90 if (copper_gold_1m or 0) > 2 else -0.90 if (copper_gold_1m or 0) < -2 else 0.0
    score += 0.55 if (dxy_1m or 0) < -1.5 else -0.55 if (dxy_1m or 0) > 1.5 else 0.0
    score = _clip_score(score)

    details = (
        f"Copper/Gold 1M {copper_gold_1m:+.2f}% | DXY 1M {dxy_1m:+.2f}%"
        if None not in (copper_gold_1m, dxy_1m)
        else "Dati macro tattici parziali"
    )
    return RegimePillar("MACRO", score, _state_5(score), details)


def _tactical_diagnosis(score: float, previous_score: float) -> str:
    delta = score - previous_score
    if delta >= 0.35:
        return "IMPROVING"
    if delta <= -0.35:
        return "DETERIORATING"
    if score >= 0.60:
        return "STABLE — POSITIVE"
    if score <= -0.60:
        return "STABLE — NEGATIVE"
    return "STABLE / MIXED"


def compute_tactical_score(close: pd.DataFrame) -> Tuple[float, List[RegimePillar]]:
    pillars = [
        _tactical_equity(close),
        _tactical_volatility(close),
        _tactical_credit(close),
        _tactical_rates(close),
        _tactical_macro(close),
    ]
    weights = {"EQUITY": 1.30, "VOLATILITY": 1.10, "CREDIT": 1.35, "RATES": 0.85, "MACRO": 0.75}
    score = sum(p.score * weights[p.name] for p in pillars) / sum(weights.values())
    return score, pillars


# -----------------------------------------------------------------------------
# 3) TODAY'S TONE
# -----------------------------------------------------------------------------

def _daily_pillar(name: str, score: float, details: str) -> RegimePillar:
    score = _clip_score(score)
    return RegimePillar(name, score, _state_5(score), details)


def compute_daily_layer(close: pd.DataFrame) -> Tuple[float, str, List[RegimePillar]]:
    spy = _return(close, "SPY", 1)
    qqq = _return(close, "QQQ", 1)
    acwi = _return(close, "ACWI", 1)
    vix = _return(close, "^VIX", 1)
    credit = _ratio_return(close, "HYG", "LQD", 1)
    ten = _yield_change_bps(close, "^TNX", 1)
    dxy = _return(close, "DX-Y.NYB", 1)
    copper_gold = _ratio_return(close, "HG=F", "GC=F", 1)

    equity_score = _clip_score(_safe_mean([
        2 if (spy or 0) > 0.7 else 1 if (spy or 0) > 0 else -2 if (spy or 0) < -0.7 else -1,
        2 if (qqq or 0) > 0.9 else 1 if (qqq or 0) > 0 else -2 if (qqq or 0) < -0.9 else -1,
        2 if (acwi or 0) > 0.5 else 1 if (acwi or 0) > 0 else -2 if (acwi or 0) < -0.5 else -1,
    ]))
    vol_score = 2 if (vix or 0) < -8 else 1 if (vix or 0) < 0 else -2 if (vix or 0) > 10 else -1
    credit_score = 2 if (credit or 0) > 0.25 else 1 if (credit or 0) > 0 else -2 if (credit or 0) < -0.25 else -1
    rates_score = 1 if (ten or 0) < -5 else -1 if (ten or 0) > 5 else 0
    macro_score = _clip_score(_safe_mean([
        1 if (copper_gold or 0) > 0 else -1,
        1 if (dxy or 0) < 0 else -1,
    ]))

    pillars = [
        _daily_pillar("EQUITY", equity_score, f"SPY {spy:+.2f}% | QQQ {qqq:+.2f}% | ACWI {acwi:+.2f}%" if None not in (spy, qqq, acwi) else "Dati equity giornalieri parziali"),
        _daily_pillar("VOLATILITY", vol_score, f"VIX {vix:+.2f}%" if vix is not None else "VIX non disponibile"),
        _daily_pillar("CREDIT", credit_score, f"HYG/LQD {credit:+.2f}%" if credit is not None else "Credito non disponibile"),
        _daily_pillar("RATES", rates_score, f"US10Y {ten:+.1f} bp" if ten is not None else "Tassi non disponibili"),
        _daily_pillar("MACRO", macro_score, f"Copper/Gold {copper_gold:+.2f}% | DXY {dxy:+.2f}%" if None not in (copper_gold, dxy) else "Macro giornaliero parziale"),
    ]

    weights = {"EQUITY": 1.40, "VOLATILITY": 1.25, "CREDIT": 1.25, "RATES": 0.65, "MACRO": 0.70}
    score = sum(p.score * weights[p.name] for p in pillars) / sum(weights.values())

    if score >= 0.45:
        diagnosis = "RISK-ON"
    elif score <= -0.45:
        diagnosis = "RISK-OFF"
    else:
        diagnosis = "MIXED"
    return score, diagnosis, pillars


# -----------------------------------------------------------------------------
# COSTRUZIONE E RENDERING
# -----------------------------------------------------------------------------

def _slice_before(close: pd.DataFrame, sessions: int) -> pd.DataFrame:
    if sessions <= 0 or len(close) <= sessions:
        return close
    return close.iloc[:-sessions].copy()


def build_market_regime(close: pd.DataFrame) -> Dict[str, RegimeLayer]:
    strategic_score, strategic_diag, strategic_pillars = compute_strategic_layer(close)
    strategic_prev_close = _slice_before(close, 21)
    strategic_prev_score, strategic_prev_diag, _ = compute_strategic_layer(strategic_prev_close)

    tactical_score, tactical_pillars = compute_tactical_score(close)
    tactical_prev_close = _slice_before(close, 5)
    tactical_prev_score, _ = compute_tactical_score(tactical_prev_close)
    tactical_diag = _tactical_diagnosis(tactical_score, tactical_prev_score)
    tactical_prev_prev_close = _slice_before(close, 10)
    tactical_prev_prev_score, _ = compute_tactical_score(tactical_prev_prev_close)
    tactical_prev_diag = _tactical_diagnosis(tactical_prev_score, tactical_prev_prev_score)

    daily_score, daily_diag, daily_pillars = compute_daily_layer(close)
    daily_prev_close = _slice_before(close, 1)
    daily_prev_score, daily_prev_diag, _ = compute_daily_layer(daily_prev_close)

    return {
        "STRATEGIC": RegimeLayer(
            key="STRATEGIC",
            title="STRUCTURAL BACKDROP",
            horizon="3-6 MESI",
            diagnosis=strategic_diag,
            score=strategic_score,
            previous_diagnosis=strategic_prev_diag,
            previous_score=strategic_prev_score,
            pillars=strategic_pillars,
        ),
        "TACTICAL": RegimeLayer(
            key="TACTICAL",
            title="TACTICAL DIRECTION",
            horizon="1-4 SETTIMANE",
            diagnosis=tactical_diag,
            score=tactical_score,
            previous_diagnosis=tactical_prev_diag,
            previous_score=tactical_prev_score,
            pillars=tactical_pillars,
        ),
        "DAILY": RegimeLayer(
            key="DAILY",
            title="TODAY'S TONE",
            horizon="ULTIMA SEDUTA",
            diagnosis=daily_diag,
            score=daily_score,
            previous_diagnosis=daily_prev_diag,
            previous_score=daily_prev_score,
            pillars=daily_pillars,
        ),
    }


def create_regime_radar(layer: RegimeLayer) -> go.Figure:
    labels = [pillar.name for pillar in layer.pillars]
    values = [pillar.score for pillar in layer.pillars]
    labels += [labels[0]]
    values += [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line=dict(color=ORANGE, width=2),
            fillcolor="rgba(255,159,0,0.18)",
        )
    )
    fig.update_layout(
        height=430,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Consolas, Courier New, monospace"),
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(range=[-2, 2], gridcolor=GRID, tickfont=dict(color=MUTED)),
            angularaxis=dict(gridcolor=GRID),
        ),
        margin=dict(l=30, r=30, t=55, b=30),
        showlegend=False,
        title=f"{layer.title} // PILLAR MAP",
    )
    return fig


def render_regime_card(layer: RegimeLayer) -> None:
    color = _regime_color(layer.diagnosis)
    delta = layer.score - layer.previous_score
    st.markdown(
        f"<div class='regime-badge' style='color:{color};border-color:{color}'>"
        f"<span style='font-size:0.74rem;color:{MUTED}'>{layer.title} // {layer.horizon}</span><br>"
        f"{layer.diagnosis}<br>"
        f"<span style='font-size:0.86rem;color:{TEXT}'>SCORE {layer.score:+.2f} | Δ {delta:+.2f}</span><br>"
        f"<span style='font-size:0.72rem;color:{MUTED}'>{layer.previous_diagnosis} → {layer.diagnosis}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _driver_lines(layer: RegimeLayer) -> Tuple[List[str], List[str]]:
    positive = sorted(
        [pillar for pillar in layer.pillars if pillar.score >= 0.35],
        key=lambda pillar: pillar.score,
        reverse=True,
    )
    negative = sorted(
        [pillar for pillar in layer.pillars if pillar.score <= -0.35],
        key=lambda pillar: pillar.score,
    )
    return (
        [f"{pillar.name}: {pillar.state} ({pillar.score:+.2f})" for pillar in positive],
        [f"{pillar.name}: {pillar.state} ({pillar.score:+.2f})" for pillar in negative],
    )


def build_regime_comment(results: Dict[str, RegimeLayer]) -> str:
    strategic = results["STRATEGIC"]
    tactical = results["TACTICAL"]
    daily = results["DAILY"]

    tactical_positive, tactical_negative = _driver_lines(tactical)
    strongest = max(tactical.pillars, key=lambda pillar: pillar.score)
    weakest = min(tactical.pillars, key=lambda pillar: pillar.score)

    parts = [
        f"Il contesto strutturale è **{strategic.diagnosis.lower()}**, "
        f"la direzione tattica è **{tactical.diagnosis.lower()}** e il tono dell'ultima seduta è "
        f"**{daily.diagnosis.lower()}**."
    ]

    if strategic.score > 0.35 and tactical.diagnosis == "DETERIORATING":
        parts.append(
            "Il deterioramento tattico si sviluppa all'interno di un quadro di fondo ancora costruttivo: "
            "la lettura è coerente con una correzione o con una perdita di partecipazione, non ancora con una "
            "rottura strutturale confermata."
        )
    elif strategic.score < -0.35 and tactical.diagnosis == "IMPROVING":
        parts.append(
            "Il miglioramento tattico avviene all'interno di un quadro strutturale difensivo: il recupero può "
            "rappresentare un rally di reazione finché credito, volatilità e trend globale non confermano il cambio di regime."
        )
    elif np.sign(strategic.score) == np.sign(tactical.score):
        parts.append(
            "Il segnale tattico è coerente con il contesto di fondo, aumentando la robustezza della lettura cross-asset."
        )
    else:
        parts.append(
            "I diversi orizzonti non sono pienamente allineati; il mercato si trova in una fase di transizione in cui "
            "è opportuno distinguere il trend di fondo dal movimento operativo di breve periodo."
        )

    parts.append(
        f"Il principale sostegno tattico proviene da **{strongest.name.lower()}** ({strongest.state.lower()}), "
        f"mentre il freno più rilevante è **{weakest.name.lower()}** ({weakest.state.lower()})."
    )

    if tactical_negative:
        parts.append("Fattori di rischio attivi: " + "; ".join(tactical_negative[:3]) + ".")
    if tactical_positive:
        parts.append("Fattori di supporto: " + "; ".join(tactical_positive[:3]) + ".")

    parts.append(
        "La diagnosi descrive l'ambiente cross-asset e la sua direzione di cambiamento; non costituisce, isolatamente, "
        "un segnale operativo sul singolo strumento."
    )
    return " ".join(parts)


def render_market_regime() -> None:
    st.markdown(
        "<div class='terminal-header'>MARKET REGIME // STRUCTURAL, TACTICAL & DAILY DIAGNOSIS</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Calcolo del regime strutturale, della direzione tattica e del tono giornaliero..."):
        close = download_close_batch(tuple(REGIME_UNIVERSE.values()), period="2y")

    if close.empty:
        st.error("Dati insufficienti per calcolare il Market Regime.")
        return

    results = build_market_regime(close)

    cards = st.columns(3)
    for column, key in zip(cards, ("STRATEGIC", "TACTICAL", "DAILY")):
        with column:
            render_regime_card(results[key])

    st.markdown("<div class='terminal-subheader'>REGIME TRANSITION</div>", unsafe_allow_html=True)
    transition_rows = [
        {
            "ORIZZONTE": layer.title,
            "PRECEDENTE": layer.previous_diagnosis,
            "ATTUALE": layer.diagnosis,
            "SCORE PRECEDENTE": layer.previous_score,
            "SCORE ATTUALE": layer.score,
            "DELTA": layer.score - layer.previous_score,
        }
        for layer in results.values()
    ]
    transition_table = pd.DataFrame(transition_rows)
    st.dataframe(
        transition_table.style.format(
            {
                "SCORE PRECEDENTE": "{:+.2f}",
                "SCORE ATTUALE": "{:+.2f}",
                "DELTA": "{:+.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='terminal-subheader'>REGIME INTERPRETATION</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='report-box'>{build_regime_comment(results)}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.95])
    with left:
        selected_layer = st.selectbox(
            "Pillar map",
            options=["STRATEGIC", "TACTICAL", "DAILY"],
            format_func=lambda key: results[key].title,
        )
        st.plotly_chart(create_regime_radar(results[selected_layer]), use_container_width=True)

    with right:
        st.markdown("<div class='terminal-subheader'>PILLAR DIAGNOSTICS</div>", unsafe_allow_html=True)
        rows = []
        for key, layer in results.items():
            for pillar in layer.pillars:
                rows.append(
                    {
                        "ORIZZONTE": layer.title,
                        "PILASTRO": pillar.name,
                        "STATO": pillar.state,
                        "SCORE": pillar.score,
                        "DETTAGLIO": pillar.details,
                    }
                )
        table = pd.DataFrame(rows)
        styled = (
            table.style
            .format({"SCORE": "{:+.2f}"})
            .map(
                lambda value: (
                    f"color:{GREEN};font-weight:700"
                    if "POSITIVE" in str(value)
                    else f"color:{RED};font-weight:700"
                    if "NEGATIVE" in str(value)
                    else f"color:{ORANGE};font-weight:700"
                ),
                subset=["STATO"],
            )
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=470)

    st.markdown("<div class='terminal-subheader'>MODEL LOGIC</div>", unsafe_allow_html=True)
    st.markdown(
        """
- **Structural Backdrop:** trend a 3-6 mesi, posizione rispetto alla MM200, credito strutturale, VIX, curva e tassi.
- **Tactical Direction:** accelerazione/decelerazione a 1-4 settimane e confronto con la rilevazione di una settimana prima.
- **Today's Tone:** movimento dell'ultima seduta su equity, VIX, credito, tassi, dollaro e Copper/Gold.
- Le tre sezioni hanno **classificazioni diverse**: `CONSTRUCTIVE/DEFENSIVE`, `IMPROVING/DETERIORATING`, `RISK-ON/RISK-OFF`.
- La tabella **Regime Transition** mostra esplicitamente il passaggio dallo stato precedente a quello attuale.
- Questo modello macro è separato dalla metodologia documentale del Composite Momentum.
        """
    )



# =============================================================================
# SECURITY REPORT: METODOLOGIA CICLICA DOCUMENTALE
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_analysis(
    ticker: str,
    period: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, TimeframeResult], Dict[str, str]]:
    daily = download_prices(ticker, period)
    daily_raw = download_prices_raw(ticker, period)
    frames: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, TimeframeResult] = {}
    errors: Dict[str, str] = {}

    for timeframe, rule in RESAMPLE_RULES.items():
        try:
            ohlc = resample_ohlc(daily, rule)
            calculated = calculate_composite_momentum(ohlc)
            frames[timeframe] = calculated
            summaries[timeframe] = summarize_timeframe(timeframe, calculated)
        except Exception as error:
            errors[timeframe] = str(error)

    return daily,daily_raw, frames, summaries, errors


def direction_word(direction: str) -> str:
    return {"UP": "crescente", "DOWN": "decrescente", "FLAT": "laterale"}.get(direction, direction.lower())


def describe_timeframe(label: str, result: TimeframeResult) -> str:
    movement = result.composite - result.previous_composite
    magnitude = abs(movement)

    if magnitude >= 20:
        pace = "con un'accelerazione marcata"
    elif magnitude >= 8:
        pace = "con una variazione significativa"
    elif magnitude >= 2:
        pace = "con una variazione moderata"
    else:
        pace = "con una variazione contenuta"

    if result.composite >= 50:
        zone = "Il momentum è in area di eccesso positivo; l'eccesso non costituisce da solo un segnale di vendita."
    elif result.composite <= -50:
        zone = "Il momentum è in area di eccesso negativo; l'ipervenduto non equivale automaticamente a un acquisto."
    elif result.composite > 0:
        zone = "Il Composite Momentum resta nella metà positiva della scala."
    else:
        zone = "Il Composite Momentum resta nella metà negativa della scala."

    return (
        f"Sul timeframe **{label.lower()}**, il Composite Momentum quota **{result.composite:.1f}**, "
        f"rispetto a **{result.previous_composite:.1f}** della rilevazione precedente. "
        f"La pendenza è **{direction_word(result.direction)}**, {pace}. {zone} "
        f"La lettura del flesso è **{result.turn.lower()}**."
    )


def build_security_report(
    ticker: str,
    summaries: Dict[str, TimeframeResult],
) -> Tuple[str, str, int, str]:
    parts: List[str] = []
    yearly = summaries.get("YEARLY")
    quarterly = summaries.get("QUARTERLY")
    monthly = summaries.get("MONTHLY")
    weekly = summaries.get("WEEKLY")

    if yearly:
        structural = "costruttivo" if yearly.direction == "UP" else "difensivo" if yearly.direction == "DOWN" else "neutrale"
        parts.append(
            f"Il quadro strutturale di **{ticker.upper()}** appare **{structural}**. "
            f"Il momentum annuale è {direction_word(yearly.direction)} e si colloca a **{yearly.composite:.1f}**. "
            "Questa indicazione definisce il contesto di fondo, ma non rappresenta autonomamente un timing operativo."
        )

    for key in ("QUARTERLY", "MONTHLY", "WEEKLY"):
        if key in summaries:
            parts.append(describe_timeframe(TIMEFRAME_LABELS[key], summaries[key]))

    action = "QUADRO INCOMPLETO"
    rating = 0
    matrix_note = "Non sono disponibili tutti i timeframe richiesti dalla matrice."

    if quarterly and monthly and weekly:
        action, rating, matrix_note = strategy_from_matrix(
            quarterly.direction,
            monthly.direction,
            weekly.turn,
        )

        if action == "BUY":
            parts.append("La combinazione multi-timeframe identifica una **finestra tattica rialzista**.")
        elif action == "SELL SHORT":
            parts.append("La combinazione multi-timeframe identifica una **finestra tattica ribassista**.")
        elif action == "TAKE PROFIT":
            parts.append("La matrice indica una fase di **presa di profitto o riduzione del rischio**.")
        else:
            parts.append("Nell'ultima barra settimanale non emerge una nuova giuntura operativa.")

    return "\n\n".join(parts), action, rating, matrix_note


def create_price_chart(daily: pd.DataFrame, ticker: str, years: int) -> go.Figure:
    cutoff = daily.index.max() - pd.DateOffset(years=years)
    frame = daily.loc[daily.index >= cutoff]

    fig = go.Figure(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="Prezzo",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
        )
    )
    fig.update_layout(title=f"{ticker.upper()} // PRICE ACTION", xaxis_rangeslider_visible=False, yaxis_title="Prezzo")
    return apply_terminal_layout(fig, 520)


def create_composite_chart(frame: pd.DataFrame, ticker: str, timeframe: str) -> go.Figure:
    clean = frame.dropna(subset=["Composite"]).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=clean.index, y=clean["Close"], name="Chiusura", line=dict(width=1.4, color=BLUE)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=clean.index, y=clean["Composite"], name="Composite Momentum", line=dict(width=2.2, color=ORANGE)),
        secondary_y=True,
    )

    for level, dash, color in [(80, "dot", RED), (50, "dash", ORANGE), (0, "solid", MUTED), (-50, "dash", ORANGE), (-80, "dot", GREEN)]:
        fig.add_hline(y=level, line_width=1, line_dash=dash, line_color=color, opacity=0.7, secondary_y=True)

    fig.update_yaxes(title_text="Prezzo", secondary_y=False)
    fig.update_yaxes(title_text="Composite Momentum", range=[-105, 105], secondary_y=True)
    fig.update_layout(title=f"{ticker.upper()} // COMPOSITE MOMENTUM // {TIMEFRAME_LABELS[timeframe].upper()}")
    return apply_terminal_layout(fig, 520)


def render_summary_table(summaries: Dict[str, TimeframeResult]) -> None:
    rows = []
    for key in ("YEARLY", "QUARTERLY", "MONTHLY", "WEEKLY"):
        if key not in summaries:
            continue
        item = summaries[key]
        rows.append(
            {
                "TIMEFRAME": TIMEFRAME_LABELS[key].upper(),
                "DATA": item.date.strftime("%d/%m/%Y"),
                "COMPOSITE": round(item.composite, 2),
                "PRECEDENTE": round(item.previous_composite, 2),
                "DIREZIONE": item.direction,
                "QUADRANTE": item.quadrant,
                "ZONA": item.position,
                "FLESSO": item.turn,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_security_report() -> None:
    st.markdown("<div class='terminal-header'>SECURITY REPORT // CYCLICAL ANALYSIS</div>", unsafe_allow_html=True)

    controls = st.columns([1.2, 1, 1, 1])
    ticker = controls[0].text_input("Ticker Yahoo Finance", value="ENI.MI").strip().upper()
    period = controls[1].selectbox("Storico", ["max", "20y", "15y", "10y"], index=0)
    chart_years = controls[2].slider("Anni grafico", 1, 15, 5)
    controls[3].markdown("<br>", unsafe_allow_html=True)
    controls[3].button("GENERATE REPORT", type="primary", use_container_width=True)

    if not ticker:
        st.info("Inserisci un ticker.")
        return

    try:
        with st.spinner(f"Analisi di {ticker} in corso..."):
            daily,daily_raw, frames, summaries, errors = load_analysis(ticker, period)
    except Exception as error:
        st.error(f"Impossibile completare l'analisi: {error}")
        return

    if not summaries:
        st.error("Nessun timeframe dispone di dati sufficienti.")
        return

    report, action, rating, matrix_note = build_security_report(ticker, summaries)
    latest_close = float(daily["Close"].iloc[-1])
    previous_close = float(daily["Close"].iloc[-2])
    daily_change = (latest_close / previous_close - 1.0) * 100.0
    weekly = summaries.get("WEEKLY")
    monthly = summaries.get("MONTHLY")

    cols = st.columns(4)
    cols[0].metric("LAST PRICE", f"{latest_close:,.2f}", f"{daily_change:+.2f}%")
    cols[1].metric("WEEKLY CM", f"{weekly.composite:.1f}" if weekly else "N/D", weekly.direction if weekly else None)
    cols[2].metric("MONTHLY CM", f"{monthly.composite:.1f}" if monthly else "N/D", monthly.direction if monthly else None)
    cols[3].metric("MATRIX SIGNAL", action, "●" * rating if rating else "N/D")

    signal_color = GREEN if action == "BUY" else RED if action == "SELL SHORT" else ORANGE
    st.markdown("<div class='terminal-subheader'>TACTICAL SIGNAL</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='signal-box' style='border-left-color:{signal_color}'>"
        f"<b style='color:{signal_color};font-size:1.15rem'>{action}</b><br>"
        f"Reward/Risk documentale: <b>{'●' * rating if rating else 'N/D'}</b><br>"
        f"<span class='small-note'>{matrix_note}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='terminal-subheader'>ANALYST COMMENT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-box'>{report.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='terminal-subheader'>QUANTITATIVE FRAMEWORK</div>", unsafe_allow_html=True)
    render_summary_table(summaries)

    tabs = st.tabs(["PRICE", "WEEKLY CM", "MONTHLY CM", "QUARTERLY CM"])
    with tabs[0]:
        st.plotly_chart(create_price_chart(daily_raw, ticker, chart_years), use_container_width=True)
    with tabs[1]:
        if "WEEKLY" in frames:
            st.plotly_chart(create_composite_chart(frames["WEEKLY"], ticker, "WEEKLY"), use_container_width=True)
    with tabs[2]:
        if "MONTHLY" in frames:
            st.plotly_chart(create_composite_chart(frames["MONTHLY"], ticker, "MONTHLY"), use_container_width=True)
    with tabs[3]:
        if "QUARTERLY" in frames:
            st.plotly_chart(create_composite_chart(frames["QUARTERLY"], ticker, "QUARTERLY"), use_container_width=True)

    if errors:
        with st.expander("TIMEFRAME NON DISPONIBILI"):
            for timeframe, message in errors.items():
                st.write(f"**{TIMEFRAME_LABELS[timeframe]}:** {message}")


# =============================================================================
# METHODOLOGY
# =============================================================================

def render_methodology() -> None:
    st.markdown("<div class='terminal-header'>METHODOLOGY // DATA AND MODEL BOUNDARIES</div>", unsafe_allow_html=True)
    st.markdown(
        """
### Security Report

La sezione sul singolo titolo usa esclusivamente la parte formalizzata nei paper:

- KEY
- XTL
- Composite Momentum
- livelli 0, ±50 e ±80
- direzione annuale, trimestrale, mensile e settimanale
- flessi settimanali
- matrice operativa multi-timeframe

Non vengono replicate formule proprietarie che non compaiono integralmente nei documenti.

### Global Macro

Global Macro e Market Regime sono moduli aggiuntivi della dashboard. Utilizzano dati Yahoo Finance e non fanno parte della tecnica documentale di Francesco Caruso.

### Bond data

- `^TNX`, `^FVX`, `^IRX`, `^TYX` e gli eventuali ticker 2Y disponibili sono trattati come indici di rendimento.
- futures Treasury ed ETF BTP/Bund sono strumenti di **prezzo**, non rendimenti.
- il terminale non calcola lo spread BTP-Bund da ETF, perché sarebbe metodologicamente errato.

### Uso

Il progetto ha finalità didattiche e di ricerca. I dati possono essere ritardati, incompleti o temporaneamente indisponibili.
        """
    )


# =============================================================================
# NAVIGAZIONE
# =============================================================================

render_top_bar()

with st.sidebar:
    st.markdown("<div class='terminal-header'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio(
        "Page",
        [
            "GLOBAL OVERVIEW",
            "GLOBAL MACRO",
            "GLOBAL SHIPPING",
            "MARKET REGIME",
            "SECURITY REPORT",
            "METHODOLOGY",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("CLEAR DATA CACHE", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

   

if page == "GLOBAL OVERVIEW":
    render_global_overview()

elif page == "GLOBAL MACRO":
    render_global_macro()

elif page == "GLOBAL SHIPPING":
    render_shipping()

elif page == "MARKET REGIME":
    render_market_regime()

elif page == "SECURITY REPORT":
    render_security_report()

else:
    render_methodology()
