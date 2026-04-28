from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import json
import os
import re
from pathlib import Path
import urllib.error
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
import requests

API_KEY = "y0axaPDDB3bSmyfPBpLg45tj4ZJdMjgW"
OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"

# User-intent taxonomy (primary + optional secondary). Do not add categories outside this set.
ALLOWED_ANALYSIS_INTENTS = frozenset(
    {"valuation", "risk", "growth", "comparison", "full_analysis"}
)

# Annual EPS, yearly average price (historical P/E), price history API range — max year; TTM uses last 4 quarters separately.
MAX_HISTORICAL_YEAR = 2025

app = FastAPI()

# Allow the simple static chat UI to call this API from a browser.
# (Browsers often send an OPTIONS preflight for JSON POSTs.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory conversation memory.
# Keyed by a per-client session id; stores ticker context + short chat history.
MAX_CONVERSATION_TURNS = 5
MAX_CONVERSATION_MESSAGES = MAX_CONVERSATION_TURNS * 2
session_memory: dict[str, dict[str, object]] = {}

# Cache resolved company-name -> ticker to reduce repeated API calls.
ticker_name_cache: dict[str, str] = {}


def _new_conversation_state() -> dict[str, object]:
    """Create a reusable per-session state payload."""
    return {
        "current_ticker": None,
        "messages": [],
    }


def _coerce_conversation_state(state: object) -> dict[str, object]:
    """
    Normalize legacy/partial state shapes into the current structure:
    {current_ticker: str|None, messages: list[{role, content, ticker?}]}
    """
    if not isinstance(state, dict):
        return _new_conversation_state()

    current_ticker = state.get("current_ticker")
    if not isinstance(current_ticker, str) or not current_ticker.strip():
        # Backwards compatibility with older schema that used `last_ticker`.
        legacy = state.get("last_ticker")
        current_ticker = legacy.strip().upper() if isinstance(legacy, str) and legacy.strip() else None
    else:
        current_ticker = current_ticker.strip().upper()

    messages_raw = state.get("messages")
    messages: list[dict[str, str]] = []
    if isinstance(messages_raw, list):
        for item in messages_raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            msg: dict[str, str] = {"role": role, "content": content.strip()}
            ticker = item.get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                msg["ticker"] = ticker.strip().upper()
            messages.append(msg)

    if len(messages) > MAX_CONVERSATION_MESSAGES:
        messages = messages[-MAX_CONVERSATION_MESSAGES:]

    return {"current_ticker": current_ticker, "messages": messages}


def get_conversation_state(session_key: str) -> dict[str, object]:
    """Return normalized, reusable state for a session."""
    key = str(session_key or "").strip() or "anonymous"
    state = _coerce_conversation_state(session_memory.get(key))
    session_memory[key] = state
    return state


def get_reusable_conversation_state(session_key: str) -> dict[str, object]:
    """
    Return a copy-safe state structure suitable for reuse across requests
    or persistence in another store.
    """
    state = get_conversation_state(session_key)
    current_ticker = state.get("current_ticker")
    messages = state.get("messages", [])
    safe_messages = list(messages) if isinstance(messages, list) else []
    return {
        "session_id": str(session_key or "").strip() or "anonymous",
        "current_ticker": current_ticker if isinstance(current_ticker, str) else None,
        "messages": safe_messages[-MAX_CONVERSATION_MESSAGES:],
    }


def add_conversation_message(
    state: dict[str, object],
    role: str,
    content: str,
    ticker: Optional[str] = None,
) -> dict[str, object]:
    """Append a message and keep only the most recent messages."""
    if role not in {"user", "assistant"}:
        return state
    text = str(content or "").strip()
    if not text:
        return state

    messages = state.setdefault("messages", [])
    if not isinstance(messages, list):
        messages = []
        state["messages"] = messages

    item: dict[str, str] = {"role": role, "content": text}
    if isinstance(ticker, str) and ticker.strip():
        item["ticker"] = ticker.strip().upper()
    messages.append(item)

    if len(messages) > MAX_CONVERSATION_MESSAGES:
        state["messages"] = messages[-MAX_CONVERSATION_MESSAGES:]
    return state


def resolve_tickers_for_turn(
    state: dict[str, object],
    extracted_tickers: list[str],
) -> list[str]:
    """
    Resolve tickers for the current turn:
    - If user mentions ticker(s), use them and update current_ticker.
    - Otherwise fall back to previously discussed ticker.
    """
    clean = [str(t).strip().upper() for t in extracted_tickers if str(t).strip()]
    if clean:
        state["current_ticker"] = clean[-1]
        return clean

    current_ticker = state.get("current_ticker")
    if isinstance(current_ticker, str) and current_ticker.strip():
        return [current_ticker.strip().upper()]
    return []


def resolve_query_context(
    extracted_tickers: list[str],
    session_state: dict[str, object],
) -> dict[str, object]:
    """
    Resolve ticker context for the current query turn.
    Rules:
    - Use extracted tickers if present.
    - Otherwise fall back to previously stored ticker.
    - Preserve support for multi-ticker queries.
    - Always return a structured result with a clarification message when unresolved.
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for item in extracted_tickers:
        t = str(item or "").strip().upper()
        if not t:
            continue
        if not re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,2})?", t):
            continue
        if t in seen:
            continue
        seen.add(t)
        normalized.append(t)

    if normalized:
        session_state["current_ticker"] = normalized[-1]
        return {
            "has_ticker": True,
            "tickers": normalized,
            "current_ticker": normalized[-1],
            "is_multi_ticker": len(normalized) > 1,
            "clarification_message": None,
        }

    stored = session_state.get("current_ticker")
    if isinstance(stored, str) and stored.strip():
        ticker = stored.strip().upper()
        session_state["current_ticker"] = ticker
        return {
            "has_ticker": True,
            "tickers": [ticker],
            "current_ticker": ticker,
            "is_multi_ticker": False,
            "clarification_message": None,
        }

    return {
        "has_ticker": False,
        "tickers": [],
        "current_ticker": None,
        "is_multi_ticker": False,
        "clarification_message": (
            "I couldn't identify a stock ticker in your message. "
            "Please mention at least one ticker (for example: AAPL, MSFT, or NVDA vs AMD)."
        ),
    }


def _fetch_json(url, timeout, default):
    try:
        return requests.get(url, timeout=timeout).json()
    except Exception:
        return default              # 100% coverage for this function


def fetch_news(ticker: str) -> list[str]:
    """Recent headline strings for a symbol via Financial Modeling Prep stock_news."""
    sym = (ticker or "").strip().upper()
    if not sym:
        return []
    url = (
        f"https://financialmodelingprep.com/api/v3/stock_news"
        f"?tickers={sym}&limit=20&apikey={API_KEY}"
    )
    payload = _fetch_json(url, timeout=20, default=[])
    if not isinstance(payload, list):
        return []
    out: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            out.append(title.strip())
    return out


def format_news_context(
    headlines: list[str], max_chars: int = 1200, min_headlines: int = 3, max_headlines: int = 5
) -> str:
    """
    Convert headline strings into a compact prompt-safe context block.
    Prioritizes the most recent/relevant 3-5 headlines (when available) and
    truncates cleanly to max_chars.
    """
    if not isinstance(headlines, list) or max_chars <= 0:
        return ""
    if min_headlines < 1:
        min_headlines = 1
    if max_headlines < min_headlines:
        max_headlines = min_headlines

    cleaned: list[str] = []
    seen: set[str] = set()
    for h in headlines:
        if not isinstance(h, str):
            continue
        text = " ".join(h.split()).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)

    if not cleaned:
        return ""

    # API returns newest first; keep only top N to avoid overwhelming the LLM.
    selected = cleaned[:max_headlines]

    prefix = "Recent headlines:\n"
    lines: list[str] = []
    total_len = len(prefix)

    for item in selected:
        line = f"- {item}"
        add_len = len(line) + (1 if lines else 0)  # newline before subsequent lines
        if total_len + add_len > max_chars:
            break
        lines.append(line)
        total_len += add_len

    if not lines:
        first = selected[0]
        room = max_chars - len(prefix) - 5  # keep space for bullet + ellipsis
        if room <= 0:
            return ""
        snippet = first[:room].rstrip()
        if len(snippet) < len(first):
            snippet += "..."
        return f"{prefix}- {snippet}"

    # If character limit clipped too aggressively and we still have available
    # selected headlines, force-add up to min_headlines as shortened bullets.
    idx = len(lines)
    while len(lines) < min_headlines and idx < len(selected):
        base = f"- {selected[idx]}"
        remaining = max_chars - total_len - (1 if lines else 0)
        if remaining <= 6:
            break
        if len(base) > remaining:
            base = base[: remaining - 3].rstrip() + "..."
        lines.append(base)
        total_len += len(base) + (1 if len(lines) > 1 else 0)
        idx += 1

    result = prefix + "\n".join(lines)
    if len(lines) < len(selected) and len(result) + 3 <= max_chars:
        result += "..."
    return result


def _build_price_data(historical_prices):
    yearly_prices = {}
    for item in historical_prices:
        try:
            year = item["date"][:4]
            if int(year) > MAX_HISTORICAL_YEAR:
                continue
            price = float(item["close"])
            yearly_prices.setdefault(year, []).append(price)
        except Exception:
            continue

    y_max = str(MAX_HISTORICAL_YEAR)
    return {
        year: float(np.mean(prices))
        for year, prices in yearly_prices.items()
        if "2017" <= year <= y_max
    }


def _compute_eps_ttm(income_quarter_payload):
    if not isinstance(income_quarter_payload, list) or len(income_quarter_payload) < 4:
        return None

    eps_list = []
    for q in income_quarter_payload:
        try:
            eps_list.append(float(q.get("eps")))
        except Exception:
            continue

    return sum(eps_list) if eps_list else None


# ----------------------------
# Data Agent Layer
# ----------------------------
def data_agent(symbol):
    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}"
    income_annual_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={API_KEY}"
    income_latest_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=1&apikey={API_KEY}"
    balance_latest_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={API_KEY}"
    cash_latest_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?limit=1&apikey={API_KEY}"
    dividends_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{symbol}?apikey={API_KEY}"
    to_date = f"{MAX_HISTORICAL_YEAR}-12-31"
    price_history_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&from=2017-01-01&to={to_date}&apikey={API_KEY}"
    income_quarter_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&limit=4&apikey={API_KEY}"

    profile_payload = _fetch_json(profile_url, timeout=20, default=[])
    if not isinstance(profile_payload, list) or not profile_payload:
        return {}

    income_annual_payload = _fetch_json(income_annual_url, timeout=30, default=[])
    income_latest_payload = _fetch_json(income_latest_url, timeout=20, default=[])
    balance_latest_payload = _fetch_json(balance_latest_url, timeout=20, default=[])
    cash_latest_payload = _fetch_json(cash_latest_url, timeout=20, default=[])
    dividends_payload = _fetch_json(dividends_url, timeout=30, default={})
    price_history_payload = _fetch_json(price_history_url, timeout=45, default={})
    income_quarter_payload = _fetch_json(income_quarter_url, timeout=20, default=[])

    historical_prices = price_history_payload.get("historical", []) if isinstance(price_history_payload, dict) else []

    return {
        "symbol": symbol,
        "profile": profile_payload[0] if isinstance(profile_payload, list) and profile_payload else {},
        "income_annual": income_annual_payload if isinstance(income_annual_payload, list) else [],
        "income_latest": income_latest_payload[0] if isinstance(income_latest_payload, list) and income_latest_payload else {},
        "balance_latest": balance_latest_payload[0] if isinstance(balance_latest_payload, list) and balance_latest_payload else {},
        "cash_latest": cash_latest_payload[0] if isinstance(cash_latest_payload, list) and cash_latest_payload else {},
        "dividends": dividends_payload.get("historical", []) if isinstance(dividends_payload, dict) else [],
        "price_history": historical_prices,
        "price_data": _build_price_data(historical_prices),
        "eps_ttm": _compute_eps_ttm(income_quarter_payload),
    }


# ----------------------------
# Valuation Agent Layer Helpers
# ----------------------------
def build_ratios(raw_data):
    try:
        profile = raw_data["profile"]
        income = raw_data["income_latest"]
        balance = raw_data["balance_latest"]
        cash = raw_data["cash_latest"]

        price = float(profile["price"])
        revenue = float(income["revenue"])
        net_income = float(income["netIncome"])
        shares = float(income["weightedAverageShsOut"])
        equity = float(balance["totalStockholdersEquity"])
        fcf = float(cash["freeCashFlow"])
        liabilities = float(balance["totalLiabilities"])
        gross_profit = float(income.get("grossProfit", 0))

        if equity == 0 or revenue == 0 or shares == 0:
            return None

        return {
            "psr": (price * shares) / revenue,
            "pbr": price / (equity / shares),
            "roe": (net_income / equity) * 100,
            "fcf_ratio": (fcf / revenue) * 100,
            "d/e": liabilities / equity,
            "gross_margin": (gross_profit / revenue) * 100 if revenue else 0.0,
        }
    except Exception:
        return None


def fetch_dividend_yield(raw_data, price):
    divs = raw_data.get("dividends", [])
    if not divs or not price:
        return 0.0

    div_by_year = {}
    for entry in divs:
        try:
            year = int(entry["date"][:4])
            if 2020 <= year <= MAX_HISTORICAL_YEAR:
                div_by_year[year] = div_by_year.get(year, 0.0) + float(entry["dividend"])
        except Exception:
            continue

    annual_24 = div_by_year.get(MAX_HISTORICAL_YEAR, 0.0)
    return (annual_24 / price) * 100 if price else 0.0


def calculate_annualized_return(raw_data, current_price):
    try:
        eps_data = raw_data.get("income_annual", [])
        divs = raw_data.get("dividends", [])
        current_price = float(raw_data.get("profile", {}).get("price"))

        eps_pairs = []
        for item in eps_data:
            try:
                if "date" in item and "eps" in item:
                    y = int(item["date"][:4])
                    if y > MAX_HISTORICAL_YEAR:
                        continue
                    eps_val = float(item["eps"])
                    eps_pairs.append((y, eps_val))
            except Exception:
                continue

        if len(eps_pairs) < 2:
            return 0.0

        eps_pairs_sorted = sorted(eps_pairs, key=lambda x: x[0])
        eps_pairs = eps_pairs_sorted[-8:] if len(eps_pairs_sorted) > 8 else eps_pairs_sorted

        if len(eps_pairs) < 2:
            return 0.0

        eps_years = {y: eps_val for y, eps_val in eps_pairs}
        yrs = sorted(eps_years)
        eps_vals = [eps_years[y] for y in yrs]

        oldest, latest = eps_vals[0], eps_vals[-1]
        if oldest <= 0:
            return 0.0

        cagr = (latest / oldest) ** (1 / (len(eps_vals) - 1)) - 1

        eps1 = latest * (1 + cagr)
        eps2 = eps1 * (1 + cagr)
        eps3 = eps2 * (1 + cagr)

        pe_now = current_price / latest
        projected = eps3 * pe_now

        dps = {}
        for entry in divs:
            try:
                y = int(entry["date"][:4])
                dps[y] = dps.get(y, 0) + float(entry["dividend"])
            except Exception:
                continue

        overlap = [y for y in yrs if y in dps]
        if overlap:
            ratios = [dps[y] / eps_years[y] for y in overlap[-3:] if eps_years[y] > 0]
            payout = sum(ratios) / len(ratios) if ratios else 0
            div1, div2, div3 = eps1 * payout, eps2 * payout, eps3 * payout
        else:
            div1 = div2 = div3 = 0

        gain = (projected - current_price) + div1 + div2 + div3
        ann = (pow(1 + gain / current_price, 1 / 3) - 1) * 100
        return float(ann)
    except Exception:
        return 0.0


def _extract_eps_pairs(eps_data):
    eps_pairs = []
    for item in eps_data:
        try:
            y = int(item["date"][:4])
            if y > MAX_HISTORICAL_YEAR:
                continue
            eps_val = float(item["eps"])
            eps_pairs.append((y, eps_val))
        except Exception:
            continue

    if len(eps_pairs) < 2:
        return []

    eps_pairs = sorted(eps_pairs, key=lambda x: x[0])
    return eps_pairs[-8:] if len(eps_pairs) > 8 else eps_pairs


def _filter_pe_outliers(pe_values):
    if not pe_values:
        return []
    if len(pe_values) < 4:
        return pe_values

    q1 = np.percentile(pe_values, 25)
    q3 = np.percentile(pe_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    return [pe for pe in pe_values if lower_bound <= pe <= upper_bound]


# ----------------------------
# Valuation Agent Layer
# ----------------------------
def valuation_agent(raw_data):
    eps_data = raw_data.get("income_annual", [])
    price_data = raw_data.get("price_data")
    current = raw_data.get("profile", {}).get("price")
    ratios = build_ratios(raw_data)
    symbol = raw_data.get("symbol", "")

    if not eps_data or not price_data or not current or not ratios:
        return None

    eps_pairs = _extract_eps_pairs(eps_data)
    if len(eps_pairs) < 2:
        return None

    _, vals = zip(*eps_pairs)
    oldest = vals[0]
    latest = vals[-1]

    if oldest <= 0:
        return None

    yrs_apart = len(vals) - 1
    eps_cagr = (latest / oldest) ** (1 / yrs_apart) - 1

    if eps_cagr < 0.08:
        return None

    pe_values = []
    for y, eps in eps_pairs:
        y = str(y)
        if y in price_data and eps != 0:
            pe_values.append(price_data[y] / eps)

    if not pe_values:
        return None

    pe_values = _filter_pe_outliers(pe_values)
    if not pe_values:
        return None

    avg_pe = float(np.mean(pe_values))

    eps_ttm = raw_data.get("eps_ttm")
    if eps_ttm is None or eps_ttm <= 0:
        try:
            eps_ttm = float(raw_data.get("profile", {}).get("eps"))
        except Exception:
            return None

    if eps_ttm is None or eps_ttm <= 0:
        return None

    intrinsic = avg_pe * eps_ttm
    underval = (intrinsic - current) / current * 100
    annual_ret = calculate_annualized_return(raw_data, current)
    div_yield = fetch_dividend_yield(raw_data, current)

    return {
        "symbol": symbol,
        "valuation": {
            "intrinsic_value": intrinsic,
            "current_price": current,
            "undervaluation_percent": underval,
        },
        "fundamentals": {
            "roe_percent": ratios["roe"],
            "gross_margin_percent": ratios["gross_margin"],
            "price_to_sales_ratio": ratios["psr"],
            "price_to_book_ratio": ratios["pbr"],
            "fcf_ratio_percent": ratios["fcf_ratio"],
            "debt_to_equity_percent": ratios["d/e"] * 100,
            "dividend_stock": div_yield > 0,
        },
        "growth": {
            "eps_growth_rate_percent": eps_cagr * 100,
            "peg_ratio": avg_pe / (eps_cagr * 100),
        },
        "returns": {
            "annualized_return_percent": annual_ret,
            "dividend_yield_percent": div_yield,
        },
    }


def calculate_intrinsic_value(symbol):
    raw_data = data_agent(symbol)
    return valuation_agent(raw_data)


def build_plot_data(raw_data):
    income_annual = raw_data.get("income_annual", [])
    price_data = raw_data.get("price_data", {})

    eps_by_year = {}
    for item in income_annual:
        try:
            year = int(str(item.get("date", ""))[:4])
            eps = float(item.get("eps"))
            if 2017 <= year <= MAX_HISTORICAL_YEAR:
                eps_by_year[year] = eps
        except Exception:
            continue

    valid_points = []
    for year in range(2017, MAX_HISTORICAL_YEAR + 1):
        avg_price = price_data.get(str(year))
        eps = eps_by_year.get(year)
        if avg_price is None or eps is None or eps <= 0:
            continue
        valid_points.append({"year": year, "avg_price": float(avg_price), "eps": float(eps)})

    if not valid_points:
        return []

    pe_values = [point["avg_price"] / point["eps"] for point in valid_points]
    pe_values = _filter_pe_outliers(pe_values)
    if not pe_values:
        return []

    avg_historical_pe = float(np.mean(pe_values))

    plot_data = []
    for point in valid_points:
        plot_data.append(
            {
                "year": point["year"],
                "avg_price": point["avg_price"],
                "eps": point["eps"],
                "intrinsic_value": avg_historical_pe * point["eps"],
            }
        )

    return sorted(plot_data, key=lambda x: x["year"])


# ----------------------------
# OpenAI Layer
# ----------------------------
def call_openai(prompt: str, model: Optional[str] = None) -> str:
    """
    Send a prompt to the OpenAI Responses API and return assistant text.
    Uses OPENAI_API_KEY from the environment (same pattern as daily screener script).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Export it in your environment to use /chat.",
        )

    use_model = model or OPENAI_MODEL_DEFAULT
    req_body = {
        "model": use_model,
        "input": prompt,
        # Avoid server-side storage path (OpenAI incidents have targeted store: true).
        "store": False,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(req_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ai_payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.code} {err_body}") from e
    except OSError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}") from e

    ai_text = ai_payload.get("output_text")
    if not ai_text:
        out = ai_payload.get("output", [])
        parts = []
        for item in out:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        ai_text = "\n".join(parts).strip()

    if not ai_text:
        raise HTTPException(status_code=502, detail="OpenAI response did not contain output text.")

    return ai_text


def _map_legacy_intent_to_primary_secondary(intent: str) -> tuple[str, list[str]]:
    """Map older extraction intents onto the fixed five-category taxonomy."""
    base = (intent or "").strip().lower()
    if base == "valuation":
        return ("valuation", [])
    if base == "risk":
        return ("risk", [])
    if base == "financials":
        return ("growth", [])
    if base in ("recommendation", "general", "news"):
        return ("full_analysis", [])
    return ("full_analysis", [])


def _heuristic_primary_secondary(query: str) -> tuple[str, list[str]]:
    """Fallback classification when the model is unavailable."""
    q = (query or "").lower()
    secondary: list[str] = []

    if re.search(r"\b(vs|versus|compare|comparison|better than|which one)\b", q):
        return ("comparison", [])

    risk_hit = any(k in q for k in ("risk", "downside", "volatility", "threat", "concern", "danger"))
    val_hit = any(
        k in q
        for k in ("value", "valuation", "intrinsic", "undervalued", "overvalued", "cheap", "expensive")
    )
    growth_hit = any(
        k in q
        for k in (
            "growth",
            "growing",
            "revenue",
            "eps",
            "cagr",
            "earnings growth",
            "top line",
            "long-term",
        )
    )
    full_hit = any(
        k in q
        for k in (
            "recommend",
            "buy",
            "sell",
            "hold",
            "should i",
            "good investment",
            "worth buying",
            "analyze",
            "analysis",
            "overview",
            "full picture",
        )
    )

    if risk_hit and val_hit and not growth_hit:
        return ("risk", ["valuation"])
    if risk_hit and growth_hit:
        return ("risk", ["growth"])
    if val_hit and growth_hit:
        return ("valuation", ["growth"])
    if risk_hit:
        return ("risk", [])
    if val_hit:
        return ("valuation", [])
    if growth_hit:
        return ("growth", [])
    if full_hit:
        extra: list[str] = []
        if val_hit:
            extra.append("valuation")
        if growth_hit:
            extra.append("growth")
        return ("full_analysis", extra[:2])

    return ("full_analysis", secondary)


def _wants_news_only_route(query: str) -> bool:
    """Headlines-only path: user wants news flow, not a fundamentals/valuation run."""
    ql = (query or "").lower().strip()
    if not ql:
        return False
    if not any(
        k in ql
        for k in (
            "news",
            "headline",
            "headlines",
            "what's going on",
            "what is going on",
            "happening with",
            "latest on",
        )
    ):
        return False
    if any(
        k in ql
        for k in (
            "valuation",
            "intrinsic",
            "undervalued",
            "overvalued",
            "p/e",
            "peg",
            "fundamental",
            "margin",
            "debt to equity",
            "balance sheet",
        )
    ):
        return False
    return True


def _wants_financials_only_route(query: str) -> bool:
    """Structured metrics path without prioritizing news narrative."""
    ql = (query or "").lower().strip()
    if not ql:
        return False
    fin = any(
        k in ql
        for k in (
            "financial",
            "financials",
            "income statement",
            "balance sheet",
            "cash flow",
            "margin",
            "roe",
            "revenue",
            "earnings per share",
            "eps trend",
        )
    )
    if not fin:
        return False
    if _wants_news_only_route(query):
        return False
    if any(k in ql for k in ("news", "headline", "headlines")):
        return False
    return True


def _heuristic_tickers(query: str) -> list[str]:
    """
    Best-effort extraction of explicit tickers from user text.
    - Accepts `$AAPL` / `AAPL` (ALL CAPS) style tickers.
    - Does NOT treat regular company names (e.g. "Apple") as tickers.
    """
    q = (query or "").strip()
    if not q:
        return []

    out: list[str] = []

    def add(tok: str) -> None:
        t = (tok or "").strip().upper()
        if not t:
            return
        if t not in out:
            out.append(t)

    # Common explicit patterns: `$AAPL`
    for m in re.finditer(r"\$([A-Za-z]{1,5}([.-][A-Za-z]{1,2})?)\b", q):
        add(m.group(1))

    # ALL CAPS tokens that look like tickers.
    # Only match when the token was ALL CAPS in the original string (not title case).
    blocked = {
        "WHAT",
        "WHICH",
        "STOCK",
        "STOCKS",
        "ABOUT",
        "SHOULD",
        "TODAY",
        "PRICE",
        "RISK",
        "BUY",
        "SELL",
        "HOLD",
        "VS",
        "AND",
        "OR",
        "THE",
        "A",
        "AN",
        "AI",
        "BEST",
        "NEXT",
        "COMPARE",
        "HEDGE",
        "FUNDS",
        "LIKE",
    }

    # Support dot/dash tickers like BRK.B / RDS-A
    for m in re.finditer(r"\b([A-Z]{1,4}(?:[.-][A-Z]{1,2})?)\b", q):
        tok = m.group(1).strip().upper()
        if tok and tok not in blocked:
            add(tok)

    return out


def _heuristic_ticker(query: str) -> Optional[str]:
    tickers = _heuristic_tickers(query)
    return tickers[0] if tickers else None


def _resolve_ticker_from_company_name(query: str) -> Optional[str]:
    """
    Resolve a company/stock name (e.g. "Apple") into a US ticker using FMP search.
    Returns None if nothing is confidently matched.
    """
    text = (query or "").strip()
    if not text:
        return None

    # Use the front of the query as a candidate company name to search.
    head = re.split(r"[?.!]", text, maxsplit=1)[0].strip()

    # Light cleanup: drop common intent words.
    head = re.sub(
        r"\b(stock|company|shares|price|valuation|analyze|analysis|about|tell|me|what|is|the|next|earnings|news)\b",
        " ",
        head,
        flags=re.IGNORECASE,
    ).strip()

    if not head:
        return None

    words = head.split()
    # Keep it short; company name search generally works better with short queries.
    candidate = " ".join(words[:6]).strip()
    if not candidate:
        return None

    cache_key = candidate.lower()
    if cache_key in ticker_name_cache:
        return ticker_name_cache[cache_key]

    search_url = (
        "https://financialmodelingprep.com/api/v3/search"
        f"?query={urllib.parse.quote(candidate)}&limit=5&apikey={API_KEY}"
    )
    payload = _fetch_json(search_url, timeout=20, default=[])

    ticker: Optional[str] = None
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            sym = item.get("symbol") or item.get("ticker")
            if not isinstance(sym, str):
                continue
            sym = sym.strip().upper()
            # Keep it conservative: tickers are typically 1-5 letters (+ optional . or - suffix).
            if re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,2})?", sym):
                ticker = sym
                break

    if ticker:
        ticker_name_cache[cache_key] = ticker
    return ticker


def _parse_query_intent_structured(text: str) -> dict:
    """Parse model JSON for ticker/intent; return normalized fallback-safe payload."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    ticker_raw = data.get("ticker")
    ticker = ticker_raw.strip().upper() if isinstance(ticker_raw, str) and ticker_raw.strip() else None
    intent_raw = str(data.get("intent", "general")).strip().lower()

    allowed = {"valuation", "risk", "recommendation", "general", "news", "financials"}
    intent = intent_raw if intent_raw in allowed else "general"

    return {"ticker": ticker, "intent": intent}


def _normalize_secondary_list(raw: object, primary: str) -> list[str]:
    out: list[str] = []
    if raw is None:
        return out
    if isinstance(raw, str):
        if raw.strip().upper() == "NONE" or not raw.strip():
            return out
        raw = [raw]
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        t = item.strip().lower()
        if t in ALLOWED_ANALYSIS_INTENTS and t != primary and t not in out:
            out.append(t)
    return out


def _finalize_primary_and_secondary(
    primary: str,
    secondary: list[str],
    query: str,
    tickers: list[str],
) -> tuple[str, list[str]]:
    """Clamp to allowed categories; comparison overrides when multiple names or compare-language appears."""
    q = (query or "").lower()
    p = (primary or "").strip().lower()
    if p not in ALLOWED_ANALYSIS_INTENTS:
        p = "full_analysis"

    sec = _normalize_secondary_list(secondary, p)

    comparison_signals = len(tickers) > 1 or bool(
        re.search(r"\b(vs|versus|compare|comparison|better than|which one)\b", q)
    )
    if comparison_signals:
        if p != "comparison" and p in ALLOWED_ANALYSIS_INTENTS - {"comparison"}:
            if p not in sec:
                sec.insert(0, p)
        p = "comparison"
        sec = [x for x in sec if x != "comparison"]

    return p, sec


def _parse_query_tickers_intent_structured(text: str) -> dict:
    """Parse model JSON for tickers/intent; return normalized fallback-safe payload."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    tickers_out: list[str] = []
    tickers_raw = data.get("tickers", data.get("ticker"))

    if isinstance(tickers_raw, list):
        for item in tickers_raw:
            if isinstance(item, str):
                t = item.strip().upper()
                if t and re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,2})?", t):
                    if t not in tickers_out:
                        tickers_out.append(t)
    elif isinstance(tickers_raw, str):
        t = tickers_raw.strip().upper()
        if t and re.fullmatch(r"[A-Z]{1,5}([.-][A-Z]{1,2})?", t):
            tickers_out.append(t)

    primary_raw = data.get("primary")
    if isinstance(primary_raw, str) and primary_raw.strip().lower() in ALLOWED_ANALYSIS_INTENTS:
        primary = primary_raw.strip().lower()
    else:
        intent_raw = str(data.get("intent", "general")).strip().lower()
        allowed_legacy = {"valuation", "risk", "recommendation", "general", "news", "financials"}
        legacy = intent_raw if intent_raw in allowed_legacy else "general"
        primary, _sec_from_legacy = _map_legacy_intent_to_primary_secondary(legacy)

    secondary = _normalize_secondary_list(data.get("secondary"), primary)

    return {"tickers": tickers_out, "primary_intent": primary, "secondary_intents": secondary}


def extract_tickers_and_intent(query: str, model: Optional[str] = None) -> dict:
    """
    Extract tickers plus primary/secondary intent from natural language.
    Returns {"tickers", "primary_intent", "secondary_intents", "intent"} (intent mirrors primary_intent).
    """
    text = (query or "").strip()
    if not text:
        return {
            "tickers": [],
            "primary_intent": "full_analysis",
            "secondary_intents": [],
            "intent": "full_analysis",
        }

    def _fallback_name_resolution() -> list[str]:
        segments = re.split(
            r"\bvs\b|,|&|\band\b|\bor\b|\blike\b|\/",
            text,
            flags=re.IGNORECASE,
        )
        tickers: list[str] = []
        for seg in segments:
            t = _resolve_ticker_from_company_name(seg)
            if t and t not in tickers:
                tickers.append(t)
        return tickers

    if not os.environ.get("OPENAI_API_KEY"):
        tickers = _heuristic_tickers(text)
        if not tickers:
            tickers = _fallback_name_resolution()
        hp, hs = _heuristic_primary_secondary(text)
        hp, hs = _finalize_primary_and_secondary(hp, hs, text, tickers)
        return {
            "tickers": tickers,
            "primary_intent": hp,
            "secondary_intents": hs,
            "intent": hp,
        }

    prompt = (
        "Extract structured fields from the user query.\n"
        "Return ONE JSON object only with exactly these keys:\n"
        '  "tickers": array of strings (US-style tickers uppercase) or empty array\n'
        '  "primary": one of "valuation" | "risk" | "growth" | "comparison" | "full_analysis"\n'
        '  "secondary": array of zero or more of those same strings (must not include "primary"; omit duplicates)\n'
        "Rules:\n"
        "- If the user mentions multiple tickers (e.g. 'NVDA vs AMD', 'AAPL, MSFT and GOOGL'), include them all.\n"
        "- If the user mentions company/stock names, resolve them to the primary US ticker when possible.\n"
        "- tickers must be unique, uppercase, and 1-5 letters (optionally with '.' or '-' suffix).\n"
        "- If no tickers can be resolved, return an empty array.\n"
        '- "valuation": fair value, cheap/expensive, intrinsic value, undervalued/overvalued.\n'
        '- "risk": downside, threats, volatility, balance sheet risk, macro risk to the name.\n'
        '- "growth": revenue/EPS trajectory, reinvestment, long-term expansion (not a side-by-side compare).\n'
        '- "comparison": compare two or more names, or ask which is better / vs / relative to peers.\n'
        '- "full_analysis": broad or buy/hold/sell style asks when no narrower category fits best.\n'
        "- If a second theme clearly applies, add it to secondary (e.g. long-term buy with valuation angle → "
        'full_analysis primary, secondary ["valuation","growth"]).\n'
        "- If there is no secondary theme, use an empty array.\n"
        "- No extra keys or prose.\n\n"
        f"User query: {text}"
    )

    try:
        raw_text = call_openai(prompt, model=model)
        parsed = _parse_query_tickers_intent_structured(raw_text)
    except Exception:
        parsed = {
            "tickers": [],
            "primary_intent": "full_analysis",
            "secondary_intents": [],
        }

    tickers = parsed.get("tickers") or []
    if not tickers:
        tickers = _heuristic_tickers(text) or _fallback_name_resolution()

    primary = str(parsed.get("primary_intent", "full_analysis")).strip().lower()
    if primary not in ALLOWED_ANALYSIS_INTENTS:
        primary = "full_analysis"
    secondary = parsed.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    secondary = _normalize_secondary_list(secondary, primary)
    if primary == "full_analysis" and not secondary:
        hp, hs = _heuristic_primary_secondary(text)
        if hp != "full_analysis" or hs:
            primary, secondary = hp, hs

    primary, secondary = _finalize_primary_and_secondary(primary, secondary, text, tickers)

    return {
        "tickers": tickers,
        "primary_intent": primary,
        "secondary_intents": secondary,
        "intent": primary,
    }


def extract_ticker_and_intent(query: str, model: Optional[str] = None) -> dict:
    """
    Backwards-compatible: first ticker plus intent (used by /analyze run_pipeline).
    """
    extracted = extract_tickers_and_intent(query, model=model)
    tickers = extracted.get("tickers") or []
    ticker = tickers[0] if tickers else None
    primary = str(extracted.get("primary_intent") or extracted.get("intent") or "full_analysis").strip().lower()
    secondary = extracted.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    return {
        "ticker": ticker,
        "intent": primary,
        "primary_intent": primary,
        "secondary_intents": secondary,
    }


def _detect_follow_up_query(query: str, tickers: list[str], conversation_state: Optional[dict[str, object]] = None) -> bool:
    """
    Heuristic follow-up detection:
    - likely follow-up when user references prior context and does not introduce a new ticker
    - uses optional conversation state when available
    """
    text = (query or "").strip()
    if not text:
        return False

    q = text.lower()
    has_ticker = bool(tickers)

    follow_up_markers = (
        r"^(and|also|what about|how about|then|okay|ok|yes|yeah|why|what if)\b"
    )
    has_follow_up_marker = bool(re.search(follow_up_markers, q))
    has_reference_pronoun = bool(re.search(r"\b(it|that|this|they|them|same one|that one)\b", q))
    is_short = len(text.split()) <= 8

    has_context = False
    if isinstance(conversation_state, dict):
        messages = conversation_state.get("messages")
        if isinstance(messages, list) and len(messages) > 0:
            has_context = True
        current = conversation_state.get("current_ticker")
        if isinstance(current, str) and current.strip():
            has_context = True

    if has_ticker:
        return False

    if has_context and (has_follow_up_marker or has_reference_pronoun or is_short):
        return True

    return has_follow_up_marker and not has_ticker


def understand_query(
    query: str,
    conversation_state: Optional[dict[str, object]] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Process user input and extract primary/secondary intent (fixed taxonomy), tickers, follow-up flag.
    """
    extracted = extract_tickers_and_intent(query, model=model)
    tickers = [str(t).strip().upper() for t in (extracted.get("tickers") or []) if str(t).strip()]
    primary = str(extracted.get("primary_intent") or extracted.get("intent") or "full_analysis").strip().lower()
    if primary not in ALLOWED_ANALYSIS_INTENTS:
        primary = "full_analysis"
    secondary = extracted.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    secondary = [str(s).strip().lower() for s in secondary if str(s).strip().lower() in ALLOWED_ANALYSIS_INTENTS]
    is_follow_up = _detect_follow_up_query(query, tickers, conversation_state=conversation_state)

    return {
        "intent": primary,
        "primary_intent": primary,
        "secondary_intents": secondary,
        "tickers": tickers,
        "is_follow_up": is_follow_up,
    }


def _format_secondary_intent_line(secondary: Optional[list[str]]) -> str:
    if not secondary:
        return "NONE"
    return ", ".join(secondary)


def format_primary_secondary_lines(primary: str, secondary: Optional[list[str]]) -> str:
    """Human-readable intent (matches the required Primary / Secondary labeling)."""
    p = (primary or "full_analysis").strip().lower()
    return f"Primary: {p}\nSecondary: {_format_secondary_intent_line(secondary)}"


def build_disciplined_value_investor_prompt(
    primary_intent: str,
    secondary_intents: Optional[list[str]],
    user_question: str,
    stock_data_block: str,
    *,
    step2_extra: str = "",
) -> str:
    """Shared analyst instructions + intent slots + user question + data block."""
    sec_line = _format_secondary_intent_line(secondary_intents)
    step2 = (
        "Step 2: Anchor in the data (metrics first)\n\n"
        "The Stock Data block is the source of truth.\n"
        "- From JSON: pair claims with concrete field names and values (e.g. undervaluation_percent: -12.3; peg_ratio: 1.4). "
        "Prefer digits and ratios over adjectives.\n"
        "- Lead with the numbers that matter for the user's intent—multiples, growth rates, margins, leverage, returns, "
        "yields—only where those fields exist.\n"
        "- When two metrics belong together (e.g. PEG vs EPS growth), state both figures in the same sentence or breath.\n"
        "- If a needed figure is missing, say so in one short clause; do not invent or estimate.\n\n"
        "When present, map metrics roughly as:\n"
        "- Price vs intrinsic / undervaluation → valuation\n"
        "- EPS trend, PEG → growth\n"
        "- Margins, ROE → quality\n"
        "- Debt, FCF, leverage → risk\n\n"
        "Do NOT:\n"
        "- add outside knowledge\n"
        "- use filler or generic praise not tied to a number\n"
        "- assume facts not in the data\n"
    )
    if step2_extra.strip():
        step2 = step2.rstrip() + "\n\n" + step2_extra.strip() + "\n"

    return (
        "You are a disciplined value-investing analyst.\n\n"
        "Lead with the numbers from the data; keep interpretation short and attached to those figures. "
        "Sound human—clear, natural sentences—but avoid throat-clearing and padding that isn't tied to a metric or quoted fact.\n\n"
        "---\n\n"
        "Step 1: Understand intent\n\n"
        "You are given:\n"
        "- Primary intent\n"
        "- Secondary intent(s) (optional)\n\n"
        "Follow this strictly:\n\n"
        "- If Primary = full_analysis → cover valuation + growth + profitability + risk using figures from the data\n"
        "- If Primary = valuation → focus mainly on valuation metrics\n"
        "- If Primary = risk → focus mainly on risk metrics and downside signals in the data\n"
        "- If Primary = growth → focus mainly on growth and trajectory metrics\n"
        "- If Primary = comparison → compare names on the same metrics side by side with numbers\n\n"
        "Secondary intent(s):\n"
        "- Weave in briefly as supporting stats only; do not let them crowd out the primary metrics\n\n"
        "---\n\n"
        f"{step2}"
        "---\n\n"
        "Step 3: Voice — natural but number-led\n\n"
        "- **Figure first**, then one tight clause on why it matters.\n"
        "- Aim for high signal: most sentences should include an explicit number, %, ratio, or quoted fact from Stock Data.\n"
        "- Cut filler: avoid setups like \"overall\", \"broadly\", \"it's interesting that\", \"in the current environment\" unless essential.\n"
        "- Readability: short paragraphs; you may chain 2–3 related stats in one paragraph—no rigid section headers.\n"
        "- Skip decorative adjectives (robust, strong, solid) unless immediately tied to a figure beside them.\n\n"
        "---\n\n"
        "Step 4: Scope\n\n"
        "- Narrow question → only the metrics needed to answer it, plus minimal supporting figures.\n"
        "- Broad question → walk valuation, growth, quality, and risk with **concrete figures** for each (still flowing prose, not labeled boxes).\n"
        "- Comparison → same metric names across tickers where possible; judge with numbers, then a clear preference.\n\n"
        "---\n\n"
        "Step 5: Close\n\n"
        "- End with a compact bottom line; when the data allow, include at least one quantitative hook in the last few sentences "
        "(e.g. undervaluation %, PEG, leverage).\n"
        "- No labels like \"Verdict:\"—but the stance (bullish / neutral / bearish) should be obvious.\n\n"
        "---\n\n"
        f"Primary Intent:\n{primary_intent}\n\n"
        f"Secondary Intent:\n{sec_line}\n\n"
        "---\n\n"
        f"User Question:\n{user_question}\n\n"
        "---\n\n"
        f"Stock Data:\n{stock_data_block}"
    )


def answer_query_with_context(
    query: str,
    analysis_json: dict,
    news_context: str,
    model: Optional[str] = None,
    primary_intent: str = "full_analysis",
    secondary_intents: Optional[list[str]] = None,
) -> str:
    """
    Answer using only provided stock analysis JSON and news context (disciplined value-investor template).
    """
    payload_json = json.dumps(analysis_json, indent=2, default=str)
    news_text = news_context.strip() if isinstance(news_context, str) and news_context.strip() else "(none provided)"
    combined = (
        "Structured financial snapshot (JSON):\n"
        f"{payload_json}\n\n"
        "Recent news (headline summary):\n"
        f"{news_text}"
    )
    step2_note = (
        "Fundamentals drive the answer: cite JSON field names and values first. "
        "Use news only as short reinforcement when it adds a dated fact or catalyst; do not let headlines replace the numbers."
    )
    prompt = build_disciplined_value_investor_prompt(
        primary_intent,
        secondary_intents,
        query,
        combined,
        step2_extra=step2_note,
    )
    prompt += (
        "\n\nRespond in plain prose only—dense on figures, light on filler. "
        "Do not end with a question to the user."
    )
    return call_openai(prompt, model=model)


def answer_news_only_query(
    query: str,
    news_context: str,
    model: Optional[str] = None,
    primary_intent: str = "full_analysis",
    secondary_intents: Optional[list[str]] = None,
) -> str:
    news_text = news_context.strip() if isinstance(news_context, str) and news_context.strip() else "(none provided)"
    step2_note = (
        "Stock Data is headline/source text only—do not invent fundamentals. "
        "Still be concrete: quote dates, percentages, or dollar amounts when they appear in the text; keep interpretation minimal."
    )
    prompt = build_disciplined_value_investor_prompt(
        primary_intent,
        secondary_intents,
        query,
        news_text,
        step2_extra=step2_note,
    )
    prompt += (
        "\n\nRespond in plain prose only—prioritize quoted facts and figures from the text. "
        "Do not end with a question to the user."
    )
    return call_openai(prompt, model=model)


def answer_financials_only_query(
    query: str,
    analysis_json: dict,
    model: Optional[str] = None,
    primary_intent: str = "growth",
    secondary_intents: Optional[list[str]] = None,
) -> str:
    payload_json = json.dumps(analysis_json, indent=2, default=str)
    step2_note = (
        "JSON only—no separate news. Name each metric field you use and give its value; stack the most decision-relevant numbers."
    )
    prompt = build_disciplined_value_investor_prompt(
        primary_intent,
        secondary_intents,
        query,
        payload_json,
        step2_extra=step2_note,
    )
    prompt += (
        "\n\nRespond in plain prose only—metric-dense, minimal narrative padding. "
        "Do not end with a question to the user."
    )
    return call_openai(prompt, model=model)


def _infer_recommendation_from_narrative(text: str) -> str:
    """Map bottom-line wording to buy/hold/sell for API consumers."""
    tail = (text or "")[-1600:].lower()
    if not tail.strip():
        return "hold"
    if re.search(r"\b(sell|bearish|avoid|short|overvalued|too rich)\b", tail) and not re.search(
        r"\b(buy|bullish|undervalued)\b", tail[-500:]
    ):
        return "sell"
    if re.search(r"\b(buy|bullish|undervalued|attractive)\b", tail) and not re.search(
        r"\b(sell|bearish)\b", tail[-500:]
    ):
        return "buy"
    if "hold" in tail or "neutral" in tail:
        return "hold"
    return "hold"


def summary_agent(
    valuation_output: dict,
    news_context: str = "",
    primary_intent: str = "full_analysis",
    secondary_intents: Optional[list[str]] = None,
    user_question: str = "",
    model: Optional[str] = None,
) -> dict:
    """
    Narrative answer using the disciplined value-investor template.

    Returns { "explanation", "recommendation" ("buy"|"hold"|"sell"), "reasoning" }.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "explanation": "AI summary unavailable because OPENAI_API_KEY is not set.",
            "recommendation": "hold",
            "reasoning": (
                "Set OPENAI_API_KEY before starting uvicorn "
                "(for example: export OPENAI_API_KEY=sk-..., then restart)."
            ),
        }

    payload_json = json.dumps(valuation_output, indent=2, default=str)
    news_text = news_context.strip() if isinstance(news_context, str) and news_context.strip() else "(none provided)"
    combined = (
        "Structured financial snapshot (JSON):\n"
        f"{payload_json}\n\n"
        "Recent news (headline summary):\n"
        f"{news_text}"
    )
    p = str(primary_intent or "full_analysis").strip().lower()
    if p not in ALLOWED_ANALYSIS_INTENTS:
        p = "full_analysis"
    sec = secondary_intents if isinstance(secondary_intents, list) else []
    sec = [s for s in sec if isinstance(s, str) and s.strip().lower() in ALLOWED_ANALYSIS_INTENTS]

    step2_note = (
        "Open with the strongest 4–8 figures from the JSON (field name + value), then weave in news only if it adds a dated catalyst. "
        "If news is thin, say so in one clause and stay on the numbers."
    )
    prompt = build_disciplined_value_investor_prompt(
        p,
        sec,
        user_question or "(no question text; give a concise view of the name using the data)",
        combined,
        step2_extra=step2_note,
    )
    prompt += (
        "\n\nRespond in plain prose only (no JSON)—figure-led, low fluff. "
        "Do not end with a question to the user."
    )
    raw_text = call_openai(prompt, model=model)
    rec = _infer_recommendation_from_narrative(raw_text)
    return {
        "explanation": raw_text.strip(),
        "recommendation": rec,
        "reasoning": "",
    }


# ----------------------------
# Pipeline Layer
# ----------------------------
def run_pipeline(ticker: str, user_query: Optional[str] = None):
    """
    1. Fetch raw market data (data_agent).
    2. Run valuation (valuation_agent).
    3. Build chart series (build_plot_data).
    4. Fetch and format recent news context.
    5. After valuation succeeds, generate AI summary (summary_agent) and attach to the response.
    """
    symbol = ticker.upper()
    extracted = (
        extract_ticker_and_intent(user_query, model=OPENAI_MODEL_DEFAULT)
        if isinstance(user_query, str) and user_query.strip()
        else {
            "ticker": symbol,
            "intent": "full_analysis",
            "primary_intent": "full_analysis",
            "secondary_intents": [],
        }
    )
    primary = str(extracted.get("primary_intent") or extracted.get("intent") or "full_analysis").strip().lower()
    secondary = extracted.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    raw_data = data_agent(symbol)
    valuation = valuation_agent(raw_data)
    plot_data = build_plot_data(raw_data)
    headlines = fetch_news(symbol)
    news_context = format_news_context(headlines)

    if valuation is None:
        return None

    summary = summary_agent(
        valuation,
        news_context=news_context,
        primary_intent=primary,
        secondary_intents=secondary,
        user_question=user_query or "",
        model=OPENAI_MODEL_DEFAULT,
    )

    return {
        "ticker": symbol,
        "analysis": valuation,
        "plot": plot_data,
        "intent": primary,
        "primary_intent": primary,
        "secondary_intents": secondary,
        "intent_lines": format_primary_secondary_lines(primary, secondary),
        "news_context": news_context,
        "summary": summary,
    }


def build_multi_ticker_context(
    tickers: list[str],
    query: str,
    intent: str = "comparison",
    secondary_intents: Optional[list[str]] = None,
) -> dict:
    """
    Run the pipeline per ticker and return a structured comparison-ready payload.
    This object is reusable for logging, debugging, or LLM prompting.
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for t in tickers:
        sym = (t or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)

    MAX_MULTI_TICKERS = 5
    normalized = normalized[:MAX_MULTI_TICKERS]

    stocks: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for sym in normalized:
        raw_data = data_agent(sym)
        valuation = valuation_agent(raw_data)
        if valuation is None:
            skipped.append(
                {
                    "ticker": sym,
                    "reason": "No valuation result under current data constraints",
                }
            )
            continue

        headlines = fetch_news(sym)
        news_context = format_news_context(headlines)
        stocks.append(
            {
                "ticker": sym,
                "analysis": valuation,
                "news_context": news_context,
            }
        )

    comparison_requested = str(intent or "").strip().lower() == "comparison"
    sec = secondary_intents if isinstance(secondary_intents, list) else []
    return {
        "query": query,
        "intent": intent,
        "primary_intent": str(intent or "").strip().lower(),
        "secondary_intents": sec,
        "comparison_requested": comparison_requested,
        "requested_tickers": normalized,
        "analyzed_tickers": [s["ticker"] for s in stocks],
        "skipped_tickers": skipped,
        "stocks": stocks,
    }


def build_multi_ticker_llm_input(context: dict[str, object]) -> str:
    """Multi-ticker comparison prompt using the disciplined value-investor template."""
    query = str(context.get("query", "")).strip()
    primary = str(context.get("primary_intent") or context.get("intent", "comparison")).strip().lower()
    if primary not in ALLOWED_ANALYSIS_INTENTS:
        primary = "comparison"
    secondary = context.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    secondary = [s for s in secondary if isinstance(s, str) and s.strip().lower() in ALLOWED_ANALYSIS_INTENTS]
    stocks = context.get("stocks")
    if not isinstance(stocks, list):
        stocks = []

    payload = {
        "primary_intent": primary,
        "secondary_intents": secondary,
        "comparison_requested": bool(context.get("comparison_requested")),
        "requested_tickers": context.get("requested_tickers", []),
        "analyzed_tickers": context.get("analyzed_tickers", []),
        "skipped_tickers": context.get("skipped_tickers", []),
        "stocks": [],
    }

    for stock in stocks:
        if not isinstance(stock, dict):
            continue
        payload["stocks"].append(
            {
                "ticker": stock.get("ticker"),
                "analysis": stock.get("analysis", {}),
                "news_context": stock.get("news_context", ""),
            }
        )

    block = json.dumps(payload, indent=2, default=str)
    step2_note = (
        "Multiple tickers in JSON: line up the same ratios and growth stats across names with actual values side by side; "
        "pick a winner using those numbers, not generalities."
    )
    prompt = build_disciplined_value_investor_prompt(
        primary,
        secondary,
        query,
        block,
        step2_extra=step2_note,
    )
    return (
        prompt
        + "\n\nRespond in plain prose only—comparison by metrics first. Do not end with a question to the user."
    )


def run_multi_stock_pipeline(tickers: list[str], query: str) -> str:
    """
    Comparative multi-stock analysis:
    - Run pipeline per ticker
    - Combine into structured comparison context
    - Build LLM-ready comparison prompt
    """
    u = understand_query(query, conversation_state=None, model=OPENAI_MODEL_DEFAULT)
    primary = str(u.get("primary_intent") or u.get("intent") or "full_analysis").strip().lower()
    secondary = u.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []

    if _wants_news_only_route(query) and primary != "comparison":
        news_sections: list[str] = []
        for sym in tickers:
            sym_u = (sym or "").strip().upper()
            if not sym_u:
                continue
            headlines = fetch_news(sym_u)
            nc = format_news_context(headlines)
            if len(tickers) > 1:
                news_sections.append(f"{sym_u}:\n{nc}")
            else:
                news_sections.append(nc)
        if not news_sections:
            raise HTTPException(status_code=400, detail="No tickers for news request.")
        return answer_news_only_query(
            query,
            "\n\n".join(news_sections),
            model=OPENAI_MODEL_DEFAULT,
            primary_intent=primary,
            secondary_intents=secondary,
        )

    context = build_multi_ticker_context(
        tickers,
        query=query,
        intent=primary,
        secondary_intents=secondary,
    )
    results = context.get("stocks") if isinstance(context.get("stocks"), list) else []
    if not results:
        raise HTTPException(status_code=404, detail="No valuation result for any requested ticker")

    if len(results) == 1:
        single = results[0] if isinstance(results[0], dict) else {}
        return answer_query_with_context(
            query,
            single.get("analysis", {}),  # type: ignore[arg-type]
            str(single.get("news_context", "")),
            model=OPENAI_MODEL_DEFAULT,
            primary_intent=primary,
            secondary_intents=secondary,
        )

    prompt = build_multi_ticker_llm_input(context)
    return call_openai(prompt, model=OPENAI_MODEL_DEFAULT)


def build_response_generation_prompt(structured_input: dict[str, object]) -> str:
    """
    Build the disciplined value-investor prompt from structured pipeline outputs.
    """
    query = str(structured_input.get("query", "")).strip()
    primary = str(
        structured_input.get("primary_intent") or structured_input.get("intent") or "full_analysis"
    ).strip().lower()
    if primary not in ALLOWED_ANALYSIS_INTENTS:
        primary = "full_analysis"
    secondary = structured_input.get("secondary_intents")
    if not isinstance(secondary, list):
        secondary = []
    secondary = [s for s in secondary if isinstance(s, str) and s.strip().lower() in ALLOWED_ANALYSIS_INTENTS]
    stocks = structured_input.get("stocks", [])
    if not isinstance(stocks, list):
        stocks = []

    payload = {
        "primary_intent": primary,
        "secondary_intents": secondary,
        "stocks": [],
    }
    for stock in stocks:
        if not isinstance(stock, dict):
            continue
        payload["stocks"].append(
            {
                "ticker": stock.get("ticker"),
                "analysis": stock.get("analysis", {}),
                "news_context": stock.get("news_context", ""),
            }
        )

    block = json.dumps(payload, indent=2, default=str)
    step2_extra = (
        "Multiple tickers: for each name, lead with its key JSON figures before contrasting; align on the same metric names when both have them."
        if len(payload["stocks"]) > 1
        else "Single ticker: stack the highest-signal metrics from the JSON first; use news only as short backup for catalysts."
    )
    prompt = build_disciplined_value_investor_prompt(
        primary,
        secondary,
        query,
        block,
        step2_extra=step2_extra,
    )
    return (
        prompt
        + "\n\nRespond in plain prose only—numbers forward, scannable. Do not end with a question to the user."
    )


def generate_response_from_structured_input(
    structured_input: dict[str, object],
    model: Optional[str] = None,
) -> str:
    """Generate final answer from structured pipeline outputs."""
    stocks = structured_input.get("stocks", [])
    if not isinstance(stocks, list) or not stocks:
        return (
            "I couldn't build a usable stock snapshot from the available data. "
            "Try another ticker or refine the query."
        )
    prompt = build_response_generation_prompt(structured_input)
    return call_openai(prompt, model=model or OPENAI_MODEL_DEFAULT)


def orchestrate_query_response(
    query_text: str,
    session_state: dict[str, object],
    model: Optional[str] = None,
) -> dict[str, object]:
    """
    Modular end-to-end orchestration:
    1) Process query
    2) Extract intent + tickers
    3) Resolve context with conversation state
    4) Run data pipeline for required tickers
    5) Generate response from structured inputs
    6) Update conversation state and return final payload
    """
    text = str(query_text or "").strip()
    if not text:
        return {
            "response": "Please share a stock question so I can analyze it.",
            "intent": "full_analysis",
            "tickers": [],
            "is_follow_up": False,
            "context": {"has_ticker": False},
            "structured_input": {
                "intent": "full_analysis",
                "primary_intent": "full_analysis",
                "secondary_intents": [],
                "tickers": [],
                "stocks": [],
            },
        }

    add_conversation_message(session_state, "user", text)
    previous_ticker = session_state.get("current_ticker")
    previous_ticker_norm = (
        previous_ticker.strip().upper()
        if isinstance(previous_ticker, str) and previous_ticker.strip()
        else None
    )

    understood = understand_query(text, conversation_state=session_state, model=model or OPENAI_MODEL_DEFAULT)
    extracted_tickers = understood.get("tickers", [])
    if not isinstance(extracted_tickers, list):
        extracted_tickers = []

    context = resolve_query_context(extracted_tickers, session_state)
    tickers = context.get("tickers", []) if isinstance(context.get("tickers"), list) else []
    intent = str(understood.get("intent", "full_analysis")).strip().lower()
    secondary_intents = understood.get("secondary_intents")
    if not isinstance(secondary_intents, list):
        secondary_intents = []
    secondary_intents = [
        str(s).strip().lower()
        for s in secondary_intents
        if isinstance(s, str) and str(s).strip().lower() in ALLOWED_ANALYSIS_INTENTS
    ]

    # Support follow-up comparison: "compare it with MSFT"
    if (
        intent == "comparison"
        and len(tickers) == 1
        and previous_ticker_norm
        and previous_ticker_norm != tickers[0]
    ):
        tickers = [previous_ticker_norm, tickers[0]]
        session_state["current_ticker"] = tickers[-1]

    if not context.get("has_ticker"):
        clarification = context.get("clarification_message")
        message = (
            str(clarification).strip()
            if isinstance(clarification, str) and clarification.strip()
            else "Please provide at least one stock ticker (for example: AAPL)."
        )
        add_conversation_message(session_state, "assistant", message)
        return {
            "response": message,
            "intent": intent,
            "tickers": [],
            "is_follow_up": bool(understood.get("is_follow_up")),
            "context": context,
            "structured_input": {
                "intent": intent,
                "primary_intent": intent,
                "secondary_intents": secondary_intents,
                "tickers": [],
                "stocks": [],
            },
        }

    if intent == "comparison" and len(tickers) < 2:
        message = (
            "I can run a comparison, but I need at least two tickers "
            "(for example: NVDA vs AMD)."
        )
        add_conversation_message(session_state, "assistant", message, ticker=tickers[-1] if tickers else None)
        return {
            "response": message,
            "intent": intent,
            "tickers": tickers,
            "is_follow_up": bool(understood.get("is_follow_up")),
            "context": context,
            "structured_input": {
                "intent": intent,
                "primary_intent": intent,
                "secondary_intents": secondary_intents,
                "tickers": tickers,
                "stocks": [],
            },
        }

    if _wants_news_only_route(text) and intent != "comparison":
        stock_rows: list[dict[str, object]] = []
        news_sections: list[str] = []
        for sym in tickers:
            headlines = fetch_news(sym)
            nc = format_news_context(headlines)
            stock_rows.append({"ticker": sym, "analysis": {}, "news_context": nc})
            if len(tickers) > 1:
                news_sections.append(f"{sym}:\n{nc}")
            else:
                news_sections.append(nc)
        combined_news = "\n\n".join(news_sections)
        response_text = answer_news_only_query(
            text,
            combined_news,
            model=model or OPENAI_MODEL_DEFAULT,
            primary_intent=intent,
            secondary_intents=secondary_intents,
        )
        structured_input: dict[str, object] = {
            "query": text,
            "intent": intent,
            "primary_intent": intent,
            "secondary_intents": secondary_intents,
            "intent_lines": format_primary_secondary_lines(intent, secondary_intents),
            "tickers": tickers,
            "stocks": stock_rows,
            "requested_tickers": tickers,
            "analyzed_tickers": tickers,
            "skipped_tickers": [],
            "is_follow_up": bool(understood.get("is_follow_up")),
        }
        current_ticker = session_state.get("current_ticker")
        ticker_for_msg = current_ticker if isinstance(current_ticker, str) and current_ticker.strip() else None
        add_conversation_message(session_state, "assistant", response_text, ticker=ticker_for_msg)
        return {
            "response": response_text,
            "intent": intent,
            "tickers": tickers,
            "is_follow_up": bool(understood.get("is_follow_up")),
            "context": context,
            "structured_input": structured_input,
        }

    multi_context = build_multi_ticker_context(
        tickers,
        query=text,
        intent=intent,
        secondary_intents=secondary_intents,
    )
    stocks = multi_context.get("stocks", []) if isinstance(multi_context.get("stocks"), list) else []

    structured_input = {
        "query": text,
        "intent": intent,
        "primary_intent": intent,
        "secondary_intents": secondary_intents,
        "intent_lines": format_primary_secondary_lines(intent, secondary_intents),
        "tickers": tickers,
        "stocks": stocks,
        "requested_tickers": multi_context.get("requested_tickers", []),
        "analyzed_tickers": multi_context.get("analyzed_tickers", []),
        "skipped_tickers": multi_context.get("skipped_tickers", []),
        "is_follow_up": bool(understood.get("is_follow_up")),
    }

    if _wants_financials_only_route(text) and len(stocks) == 1 and intent != "comparison":
        s0 = stocks[0]
        if isinstance(s0, dict):
            response_text = answer_financials_only_query(
                text,
                s0.get("analysis", {}),
                model=model or OPENAI_MODEL_DEFAULT,
                primary_intent=intent,
                secondary_intents=secondary_intents,
            )
        else:
            response_text = generate_response_from_structured_input(
                structured_input, model=model or OPENAI_MODEL_DEFAULT
            )
    else:
        response_text = generate_response_from_structured_input(structured_input, model=model or OPENAI_MODEL_DEFAULT)
    current_ticker = session_state.get("current_ticker")
    ticker_for_msg = current_ticker if isinstance(current_ticker, str) and current_ticker.strip() else None
    add_conversation_message(session_state, "assistant", response_text, ticker=ticker_for_msg)

    return {
        "response": response_text,
        "intent": intent,
        "tickers": tickers,
        "is_follow_up": bool(understood.get("is_follow_up")),
        "context": context,
        "structured_input": structured_input,
    }


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User message to send to the model")
    model: Optional[str] = Field(default=None, description="Override model (default: gpt-4.1-mini)")


class ChatResponse(BaseModel):
    response: str
    model: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question from the user")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversation memory (last ticker).",
    )


# ----------------------------
# API Layer
# ----------------------------
@app.get("/analyze")
def analyze(ticker: str, query: Optional[str] = None):
    result = run_pipeline(ticker, user_query=query)
    if result is None:
        raise HTTPException(status_code=404, detail="No valuation result for this ticker")
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """Send a prompt to GPT and return the model reply (requires OPENAI_API_KEY)."""
    model = body.model or OPENAI_MODEL_DEFAULT
    text = call_openai(body.prompt, model=model)
    return ChatResponse(response=text, model=model)


def _envyagent_static_dir() -> Path:
    return Path(__file__).resolve().parent


@app.get("/", include_in_schema=False)
def root():
    """Serve the chat UI at the site root (same file as /chat.html)."""
    path = _envyagent_static_dir() / "chat.html"
    if path.exists():
        return FileResponse(str(path), media_type="text/html")
    return {"status": "ok"}


@app.get("/chat.html", include_in_schema=False)
def chat_page():
    """Serve the chat UI at http://localhost:8000/chat.html (same origin as /query)."""
    path = _envyagent_static_dir() / "chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found next to main.py")
    return FileResponse(str(path), media_type="text/html")


@app.post("/query", response_model=ChatResponse)
def query(body: QueryRequest, request: Request):
    """Accept a natural language query and return the model reply."""
    model = OPENAI_MODEL_DEFAULT

    # Session memory: track current ticker + short conversation history.
    session_key_raw = (
        body.session_id
        or request.headers.get("X-Session-Id")
        or request.headers.get("x-session-id")
        or (request.client.host if request.client else None)
        or "anonymous"
    )
    session_key = str(session_key_raw).strip()
    session_state = get_conversation_state(session_key)
    result = orchestrate_query_response(body.query, session_state=session_state, model=model)
    text = str(result.get("response", "")).strip()
    if not text:
        text = "I couldn't generate a response from the current data."
    return ChatResponse(response=text, model=model)

