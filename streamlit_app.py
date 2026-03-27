"""
Streamlit chat client for the EnvyAgent API (POST /query — same as chat.html).

Set ENVY_API_URL or Streamlit secret ENVY_API_URL (default: http://127.0.0.1:8000).
Run API: uvicorn main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import uuid
import requests
import streamlit as st

INTRO = (
    "Ask me about stocks. Include a ticker like AAPL/MSFT, or a name like Apple/Microsoft. "
    "If you omit a ticker later, I'll reuse your last one for this browser session."
)


def get_api_base() -> str:
    try:
        if "ENVY_API_URL" in st.secrets:
            return str(st.secrets["ENVY_API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return os.environ.get("ENVY_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")


def post_query(api_base: str, query: str, session_id: str, timeout: int = 120) -> str:
    r = requests.post(
        f"{api_base}/query",
        json={"query": query, "session_id": session_id},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("response")
    return text if isinstance(text, str) and text.strip() else "(No response)"


def inject_styles() -> None:
    st.markdown(
        """
<style>
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .block-container {
    padding-top: 0.75rem !important;
    max-width: 880px !important;
  }
  [data-testid="stAppViewContainer"] > .main {
    background:
      radial-gradient(1000px 600px at 15% -10%, rgba(99, 102, 241, 0.22), transparent 50%),
      radial-gradient(800px 500px at 95% 0%, rgba(45, 212, 191, 0.1), transparent 45%),
      #0b0f1a !important;
  }
  [data-testid="stHeader"] { background: transparent !important; }
  footer { visibility: hidden !important; height: 0 !important; }

  [data-testid="stChatMessage"] {
    animation: fadeIn 0.25s ease-out !important;
  }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    border-radius: 14px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    line-height: 1.55 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.04) !important;
  }
  /* Chat composer: one solid fill everywhere (nested rows were transparent → two-tone bar) */
  [data-testid="stChatInput"] {
    --chat-input-fill: #14161f;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    background: var(--chat-input-fill) !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: rgba(165, 180, 252, 0.45) !important;
    box-shadow: 0 0 0 1px rgba(165, 180, 252, 0.25) !important;
  }
  [data-testid="stChatInput"] div {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: inherit !important;
    gap: 0 !important;
  }
  [data-testid="stChatInput"] textarea {
    border: none !important;
    border-radius: 0 !important;
    outline: none !important;
    box-shadow: none !important;
    background-color: inherit !important;
    color: rgba(248, 250, 252, 0.95) !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
  }
  /* Send icon button — neutral (theme primary was adding orange/red) */
  [data-testid="stChatInput"] button {
    border: none !important;
    background: transparent !important;
    color: rgba(226, 232, 240, 0.85) !important;
    box-shadow: none !important;
  }
  [data-testid="stChatInput"] button:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #f8fafc !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Finance Agent",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": INTRO}]
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    api_base = get_api_base()

    st.markdown("### Finance Agent")
    st.caption("Ask about stocks, valuations, and news")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Step 2: user message is already on screen; fetch reply (shows spinner below history)
    if st.session_state.pending_query is not None:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        try:
            with st.spinner("Thinking..."):
                reply = post_query(api_base, q, st.session_state.session_id)
        except requests.RequestException as e:
            reply = f"**Request failed:** {e}"
        st.session_state.messages = [
            *st.session_state.messages,
            {"role": "assistant", "content": reply},
        ]
        st.rerun()

    # Step 1: new question — save user line only, rerun so the question appears before the API call
    if prompt := st.chat_input("Type your question...", key="finance_chat_input"):
        st.session_state.messages = [
            *st.session_state.messages,
            {"role": "user", "content": prompt},
        ]
        st.session_state.pending_query = prompt
        st.rerun()


if __name__ == "__main__":
    main()
