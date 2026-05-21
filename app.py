import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


MAX_SECURITIES = 30
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN"]


@dataclass
class SearchResult:
    label: str
    display_name: str
    symbol: str
    exchange: str
    quote_type: str
    region: str


def normalize_quote_type(quote_type: str) -> str:
    mapping = {
        "EQUITY": "Equity",
        "ETF": "ETF",
        "MUTUALFUND": "Mutual Fund",
        "INDEX": "Index",
        "CRYPTOCURRENCY": "Crypto",
    }
    return mapping.get(quote_type.upper(), quote_type.title() if quote_type else "Unknown type")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_search_results(query: str) -> List[SearchResult]:
    query = query.strip()
    if len(query) < 2:
        return []

    try:
        raw_results = yf.Search(query, max_results=10).quotes
    except Exception:
        return []

    results: List[SearchResult] = []
    seen = set()

    for item in raw_results:
        symbol = str(item.get("symbol", "")).strip().upper()
        short_name = str(item.get("shortname") or item.get("longname") or symbol).strip()
        exchange = str(item.get("exchange", "")).strip()
        quote_type = str(item.get("quoteType", "")).strip()
        region = str(item.get("region", "")).strip()

        if not symbol or symbol in seen:
            continue

        seen.add(symbol)
        normalized_quote_type = normalize_quote_type(quote_type)
        results.append(
            SearchResult(
                label=f"{short_name} ({symbol}) · {exchange or 'Unknown exchange'} · {normalized_quote_type}",
                display_name=short_name,
                symbol=symbol,
                exchange=exchange or "Unknown exchange",
                quote_type=normalized_quote_type,
                region=region or "Unknown region",
            )
        )

    return results


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_price_history(
    symbols: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    download_args = {
        "tickers": list(symbols),
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
        "threads": True,
    }

    download_args["start"] = start_date
    download_args["end"] = end_date + timedelta(days=1)

    history = yf.download(**download_args)

    if history.empty:
        return pd.DataFrame()
