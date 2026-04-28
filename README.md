# EnvyAgent

FastAPI-based multi-agent stock analysis API with two chat UIs (`chat.html`, `streamlit_app.py`).

## Features

- Natural-language stock Q&A via `POST /query`
- Multi-agent orchestration (intent, ticker resolution, data, valuation, response)
- Single and multi-ticker analysis (`AAPL`, `NVDA vs AMD`)
- Follow-up memory (reuses last ticker per session)
- Route-specific handling:
  - news-only queries
  - financials-only queries
  - comparison queries
  - full analysis queries
- Structured analysis endpoint via `GET /analyze`
- Direct model passthrough endpoint via `POST /chat`
- Built-in web chat at `/` and `/chat.html`

## Multi-Agent Structure

All logic lives in `main.py` and is organized in layers:

1. **Conversation/State layer**
   - in-memory `session_memory`
   - stores `current_ticker` + recent messages per session
2. **Understanding layer**
   - extracts tickers, `primary_intent`, `secondary_intents`, follow-up signal
   - supports company-name-to-ticker resolution via FMP search
3. **Routing layer**
   - decides clarification vs news-only vs financials-only vs full/comparison flows
4. **Data agent**
   - fetches profile, statements, dividends, history, TTM EPS from FMP
5. **Valuation agent**
   - computes intrinsic value proxy, undervaluation, ratios, growth, returns
   - can reject low-confidence cases (`None`)
6. **Response generation**
   - builds structured prompts and calls OpenAI Responses API
   - final orchestration entrypoint: `orchestrate_query_response()`

## Intent + Context Model

Primary intent taxonomy:

- `valuation`
- `risk`
- `growth`
- `comparison`
- `full_analysis`

Context rules:

- explicit tickers in current query take priority
- otherwise uses stored `current_ticker`
- comparison requires at least two tickers
- follow-up comparison can merge previous ticker + newly mentioned ticker

## API

### `POST /query`

Main endpoint for stock questions.

Request:

```json
{
  "query": "Compare NVDA vs AMD on valuation and risk",
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "response": "Narrative answer text...",
  "model": "gpt-4.1-mini"
}
```

### `GET /analyze`

Single-ticker structured output (JSON pipeline result).

Example:

```text
/analyze?ticker=AAPL&query=optional+query
```

Returns ticker analysis, plot data, intent metadata, news context, and summary.

### `POST /chat`

Direct OpenAI call (no stock pipeline).

Request:

```json
{
  "prompt": "Explain PEG ratio simply.",
  "model": "gpt-4.1-mini"
}
```

## UI

- `chat.html` (served by FastAPI at `/` and `/chat.html`)
  - calls `POST /query`
  - stores browser session id in localStorage
- `streamlit_app.py`
  - calls `POST /query`
  - reads API base from `ENVY_API_URL` or Streamlit secrets
  - supports runtime API URL override in UI

If opening `chat.html` from `file://`, set:

```js
localStorage.setItem("ENVY_API_BASE_URL", "http://127.0.0.1:8000");
```

## Setup

```bash
cd "/Users/onirnarahari/Documents/EnvyAgent"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Chat UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Config + Ops Notes

- Default model: `OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"`
- OpenAI requests use `store: false`
- Session memory is in-process (lost on restart)
- Multi-ticker analysis is capped at 5 tickers
- CORS is open (`allow_origins=["*"]`)

## Deployment (Render)

- If Render is connected to this repo/branch with Auto Deploy ON, GitHub pushes auto-deploy.
- For Streamlit + Render split, set Streamlit `ENVY_API_URL` to your Render API URL.

## Security

Current issue:

- FMP key is hardcoded as `API_KEY` in `main.py`

Recommended:

1. Move to env var (`FMP_API_KEY`)
2. Rotate/revoke committed key
3. Use secrets management for local + Render

## Files

- `main.py` - FastAPI app + orchestration
- `chat.html` - browser UI
- `streamlit_app.py` - Streamlit UI
- `requirements.txt` - dependencies
