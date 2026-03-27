# EnvyAgent

FastAPI backend for stock Q&A: natural-language **`/query`**, valuation + news from Financial Modeling Prep, and OpenAI for answers. The only web UI in this folder is **`chat.html`** (also served at **`/`**).

## Setup

```bash
cd EnvyAgent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI API key (required for `/query`, `/chat`, and ticker extraction):

```bash
export OPENAI_API_KEY="sk-..."
```

## Run the server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Chat UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or [http://127.0.0.1:8000/chat.html](http://127.0.0.1:8000/chat.html) (same page)
- **API docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Main HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Natural-language questions; multi-ticker compare; uses in-memory **session** via `session_id` or `X-Session-Id` for last-ticker memory |
| `POST` | `/chat` | Raw prompt → OpenAI (no stock pipeline) |
| `GET` | `/analyze` | JSON pipeline: data + valuation + plot + summary for one `ticker` |

## Frontend note

`chat.html` calls `POST /query` on the **same origin** when you open it via the server above. If you ever open `chat.html` from disk (`file://`), set in the browser console:

```js
localStorage.setItem("ENVY_API_BASE_URL", "http://127.0.0.1:8000")
```

## Configuration in code

- **`OPENAI_MODEL_DEFAULT`** and OpenAI **`store: false`** are set in `main.py` (see `call_openai`).
- **Financial Modeling Prep** API key is currently embedded as `API_KEY` in `main.py` — for production, move it to an environment variable and rotate keys that were committed.

## Project layout

- `main.py` — FastAPI app, agents, `/query`, static serving of `chat.html`
- `chat.html` — Browser chat UI
- `requirements.txt` — Python dependencies
