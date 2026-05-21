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

    if isinstance(history.columns, pd.MultiIndex):
        if "Close" in history.columns.get_level_values(0):
            prices = history["Close"].copy()
        else:
            prices = history.xs("Close", axis=1, level=-1, drop_level=False).copy()
    else:
        prices = history.to_frame(name=symbols[0])

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    prices = prices.dropna(how="all")
    prices.columns = [str(col) for col in prices.columns]
    return prices


def build_seeded_matrix(symbols: List[str], market_corr: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols, dtype=float)

    for row_symbol in symbols:
        for col_symbol in symbols:
            if row_symbol == col_symbol:
                matrix.loc[row_symbol, col_symbol] = 1.0
            elif row_symbol in market_corr.index and col_symbol in market_corr.columns:
                value = float(market_corr.loc[row_symbol, col_symbol])
                matrix.loc[row_symbol, col_symbol] = 0.0 if math.isnan(value) else value

    return matrix


def sanitize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    cleaned = matrix.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cleaned = cleaned.clip(-1.0, 1.0)

    for idx in cleaned.index:
        cleaned.loc[idx, idx] = 1.0

    symmetric = (cleaned + cleaned.T) / 2
    for idx in symmetric.index:
        symmetric.loc[idx, idx] = 1.0

    return symmetric.round(4)


def rebuild_matrix_for_selection(symbols: List[str], base_matrix: pd.DataFrame, fallback_matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = fallback_matrix.copy()

    if base_matrix is None or base_matrix.empty:
        return sanitize_matrix(matrix)

    for row_symbol in symbols:
        for col_symbol in symbols:
            if row_symbol in base_matrix.index and col_symbol in base_matrix.columns:
                matrix.loc[row_symbol, col_symbol] = base_matrix.loc[row_symbol, col_symbol]

    return sanitize_matrix(matrix)


def style_correlation_matrix(matrix: pd.DataFrame):
    return (
        matrix.style.format("{:.2f}")
        .background_gradient(cmap="RdYlGn", vmin=-1, vmax=1, axis=None)
        .set_properties(**{"text-align": "center"})
    )


def add_symbol(symbol: str) -> None:
    current = st.session_state.selected_symbols
    symbol = symbol.upper()

    if symbol in current:
        st.info(f"{symbol} est deja dans la liste.")
        return

    if len(current) >= MAX_SECURITIES:
        st.warning(f"Vous pouvez selectionner jusqu'a {MAX_SECURITIES} titres.")
        return

    current.append(symbol)
    st.session_state.selected_symbols = current
    st.session_state.last_added_symbol = symbol


def remove_symbol(symbol: str) -> None:
    st.session_state.selected_symbols = [item for item in st.session_state.selected_symbols if item != symbol]
    st.session_state.last_removed_symbol = symbol


def init_state() -> None:
    if "selected_symbols" not in st.session_state:
        st.session_state.selected_symbols = DEFAULT_SYMBOLS.copy()
    if "matrix_editor" not in st.session_state:
        st.session_state.matrix_editor = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "last_search_query" not in st.session_state:
        st.session_state.last_search_query = ""
    if "last_added_symbol" not in st.session_state:
        st.session_state.last_added_symbol = None
    if "last_removed_symbol" not in st.session_state:
        st.session_state.last_removed_symbol = None
def render_search_panel() -> None:
    st.subheader("Recherche de securities")
    query = st.text_input(
        "Chercher et ajouter un titre",
        placeholder="Exemple: App, Tesla, Total, SPY...",
        help="Commencez a taper pour obtenir des suggestions Yahoo Finance.",
    )

    if len(query.strip()) < 2:
        st.session_state.search_results = []
        st.caption("Tapez au moins 2 caracteres pour afficher les suggestions.")
        return

    if query != st.session_state.last_search_query:
        st.session_state.search_results = fetch_search_results(query)
        st.session_state.last_search_query = query

    results = st.session_state.search_results
    if not results:
        st.caption("Aucun resultat trouve.")
        return

    st.caption("Suggestions affichees automatiquement pendant la saisie.")
    suggestion_box = st.container(border=True)
    with suggestion_box:
        for result in results:
            info_col, action_col = st.columns([6, 1])
            with info_col:
                st.markdown(
                    f"**{result.display_name}** (`{result.symbol}`)  \n"
                    f"{result.exchange} · {result.quote_type} · {result.region}"
                )
            with action_col:
                if st.button("Ajouter", key=f"add_{result.symbol}", use_container_width=True):
                    add_symbol(result.symbol)
                    st.rerun()


def render_selected_symbols() -> None:
    st.subheader(f"Selection ({len(st.session_state.selected_symbols)}/{MAX_SECURITIES})")

    if not st.session_state.selected_symbols:
        st.caption("Aucun titre selectionne pour le moment.")
        return

    st.caption("Titres actuellement inclus dans la matrice.")
    cols = st.columns(3)
    for index, symbol in enumerate(st.session_state.selected_symbols):
        with cols[index % 3]:
            st.markdown(f"`{symbol}`")
            if st.button("Retirer", key=f"remove_{symbol}"):
                remove_symbol(symbol)
                st.rerun()


def render_matrix_editor() -> None:
    symbols = st.session_state.selected_symbols
    if len(symbols) < 2:
        st.info("Ajoutez au moins 2 titres pour construire une matrice de correlation.")
        return

    st.subheader("Plage de temps")
    st.caption("Choisissez les dates precises utilisees pour calculer la matrice de correlation.")
    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(days=365)
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input("Date de debut", value=default_start_date)
    with date_col2:
        end_date = st.date_input("Date de fin", value=default_end_date)

    if start_date >= end_date:
        st.warning("La date de debut doit etre anterieure a la date de fin.")
        return

    with st.spinner("Chargement des donnees de marche..."):
        prices = fetch_price_history(
            tuple(symbols),
            start_date=start_date,
            end_date=end_date,
        )

    if prices.empty:
        st.warning("Impossible de charger les prix via Yahoo Finance pour la selection actuelle.")
        seeded_matrix = pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)
    else:
        returns = prices.pct_change().dropna(how="all")
        market_corr = returns.corr().fillna(0.0)
        seeded_matrix = build_seeded_matrix(symbols, market_corr)

    if st.session_state.matrix_editor is None:
        st.session_state.matrix_editor = sanitize_matrix(seeded_matrix)
    elif list(st.session_state.matrix_editor.index) != symbols:
        st.session_state.matrix_editor = rebuild_matrix_for_selection(
            symbols,
            st.session_state.matrix_editor,
            seeded_matrix,
        )

    st.subheader("Matrice de correlation editable")
    st.caption("Les valeurs sont forcees entre -1 et 1. La diagonale reste toujours a 1.")

    edited = st.data_editor(
        st.session_state.matrix_editor,
        use_container_width=True,
        num_rows="fixed",
        key="corr_editor_widget",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Symetriser et valider", type="primary"):
            st.session_state.matrix_editor = sanitize_matrix(edited)
            st.success("La matrice a ete nettoyee et symetrisee.")
            st.rerun()
    with c2:
        if st.button("Reinitialiser depuis le marche"):
            st.session_state.matrix_editor = seeded_matrix
            st.rerun()

    final_matrix = sanitize_matrix(edited)
    st.subheader("Matrice finale")
    st.caption("Heatmap: rouge = correlation negative, blanc = neutre, vert = correlation positive.")
    st.dataframe(style_correlation_matrix(final_matrix), use_container_width=True)

    csv_data = final_matrix.to_csv().encode("utf-8")
    st.download_button(
        "Telecharger en CSV",
        data=csv_data,
        file_name="correlation_matrix.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Correlation Matrix Editor", layout="wide")
    init_state()
    st.markdown(
        """
        <style>
        div[data-testid="stTextInput"] {
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Correlation Matrix Editor")
    st.write(
        "Selectionnez jusqu'a 30 securities via Yahoo Finance, puis modifiez librement la matrice de correlation."
    )

    top_col1, top_col2, top_col3 = st.columns(3)
    top_col1.metric("Titres selectionnes", len(st.session_state.selected_symbols))
    top_col2.metric("Maximum", MAX_SECURITIES)
    top_col3.metric("Places restantes", MAX_SECURITIES - len(st.session_state.selected_symbols))

    left_col, right_col = st.columns([1.1, 1])
    with left_col:
        render_search_panel()
    with right_col:
        render_selected_symbols()

    st.divider()
    render_matrix_editor()


if __name__ == "__main__":
    main()
