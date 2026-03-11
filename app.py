"""
NexusRAG — Multi-Agent Intelligence Platform
Streamlit Frontend · OpenRouter API · Mobile-friendly
"""

import streamlit as st
import os, sys, json, tempfile, base64, re
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusRAG · Multi-Agent AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",  # always open
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #070b14;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f1f3d 0%, #070b14 50%, #0a0514 100%);
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #3b5bdb; border-radius: 2px; }

/* ── sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}

/* ── mobile responsive ── */
@media (max-width: 768px) {
    /* tabs smaller text */
    .stTabs [data-baseweb="tab"] { font-size: 0.72rem !important; padding: 0.35rem 0.45rem !important; }
    /* bigger tap targets on buttons */
    .stButton > button { min-height: 46px !important; }
    /* bubbles nearly full width */
    .bubble-user, .bubble-agent { max-width: 98% !important; font-size: 0.82rem !important; }
    /* hide long descriptions to save space */
    .agent-bar-desc { display: none !important; }
    /* smaller title */
    .nexus-title { font-size: 1.75rem !important; }
}

/* ── sidebar always visible and styled ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f1e 0%,#070b14 100%) !important;
    border-right: 1px solid #1e2d4a !important;
    min-width: 270px !important;
    transform: none !important;
    visibility: visible !important;
}
/* Collapse/expand button styling */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    background: #1a2744 !important;
    border: 1px solid #3b5bdb !important;
    border-radius: 8px !important;
    color: #c7d2fe !important;
    width: 40px !important;
    height: 40px !important;
}
/* Always show the collapsed control so user can re-open */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── header ── */
.nexus-header { text-align: center; padding: 1rem 0 0.8rem; }
.nexus-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1;
    background: linear-gradient(135deg, #7c6df2 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.nexus-sub {
    font-family: 'Space Mono', monospace; color: #475569; font-size: 0.72rem;
    letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.3rem;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a2744 0%, #111827 100%) !important;
    color: #c7d2fe !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; text-align: left !important;
    transition: all 0.2s ease !important; padding: 0.6rem 1rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #7c6df2 !important; color: #fff !important;
    box-shadow: 0 4px 16px rgba(124,109,242,0.25) !important;
    transform: translateY(-1px) !important;
}

/* ── chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #1e2d4a, #1a1f2e);
    border: 1px solid #2d4a7a; border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.1rem; max-width: 75%;
    color: #c7d2fe; font-size: 0.88rem; line-height: 1.5;
    margin: 0.4rem 0 0.4rem auto;
    word-break: break-word;
}
.bubble-agent {
    background: linear-gradient(135deg, #111827, #0d1526);
    border: 1px solid #1e3a5f; border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1.1rem; max-width: 85%;
    color: #e2e8f0; font-size: 0.88rem; line-height: 1.6;
    margin: 0.4rem 0;
    word-break: break-word;
}
.bubble-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.3rem; opacity: 0.7;
}
.bubble-user  .bubble-label { color: #3b82f6; }
.bubble-agent .bubble-label { color: #7c6df2; }

/* ── inputs ── */
.stTextArea textarea {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.9rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #7c6df2 !important;
    box-shadow: 0 0 0 2px rgba(124,109,242,0.2) !important;
}
.stTextInput input {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus { border-color: #7c6df2 !important; }
.stSelectbox > div > div {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    background: #0a0f1e; border: 1px dashed #1e3a5f;
    border-radius: 12px; padding: 1rem;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1e; border-radius: 8px; padding: 3px; gap: 3px;
    border: 1px solid #1e2d4a; flex-wrap: nowrap; overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #64748b !important;
    border-radius: 6px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b5bdb, #7c6df2) !important;
    color: #fff !important;
}

/* ── expander ── */
.streamlit-expanderHeader {
    background: #0a0f1e !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #7c6df2 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important;
}

/* ── badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #0d1526; border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 0.2rem 0.7rem; font-family: 'Space Mono', monospace;
    font-size: 0.65rem; color: #64748b; margin: 0.1rem;
}
.badge.on   { border-color: #10b981; color: #10b981; background: #052e16; }
.badge.off  { border-color: #374151; color: #374151; }
.badge.info { border-color: #3b82f6; color: #3b82f6; }
.badge.warn { border-color: #f59e0b; color: #f59e0b; }

/* ── alerts ── */
hr { border-color: #1e2d4a !important; }
[data-testid="metric-container"] {
    background: #0a0f1e; border: 1px solid #1e2d4a; border-radius: 8px; padding: 0.5rem;
}
.stSuccess { background: #052e16 !important; border-left: 4px solid #10b981 !important; border-radius: 6px !important; }
.stError   { background: #2d0a0a !important; border-left: 4px solid #ef4444 !important; border-radius: 6px !important; }
.stInfo    { background: #0c1a2e !important; border-left: 4px solid #3b82f6 !important; border-radius: 6px !important; }
.stWarning { background: #2d1a00 !important; border-left: 4px solid #f59e0b !important; border-radius: 6px !important; }
pre { background: #060a12 !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; }

/* ── pills ── */
.src-pill {
    display: inline-block; background: #0d1526; border: 1px solid #1e3a5f;
    border-radius: 4px; padding: 0.1rem 0.5rem;
    font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #64748b; margin: 0.1rem;
}
.ts-pill {
    display: inline-block; background: #0d1f3a; border: 1px solid #3b5bdb;
    border-radius: 4px; padding: 0.15rem 0.6rem;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #7c6df2;
    margin: 0.15rem; text-decoration: none;
}

/* ── misc ── */
.empty-state {
    text-align: center; padding: 3rem; opacity: 0.35;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #475569;
}
.agent-bar {
    display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;
    background: #0a0f1e; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 0.7rem 1rem; flex-wrap: wrap;
}
.agent-bar-icon { font-size: 1.4rem; }
.agent-bar-name { font-weight: 700; font-size: 0.95rem; color: #c7d2fe; }
.agent-bar-desc { font-size: 0.72rem; color: #475569; font-family: 'Space Mono', monospace; margin-top: 0.1rem; }
.delegate-tag {
    display: inline-block; background: #2d1a00; border: 1px solid #f59e0b;
    color: #f59e0b; border-radius: 4px; padding: 0.05rem 0.4rem;
    font-family: 'Space Mono', monospace; font-size: 0.6rem; margin-left: 0.4rem;
}
.info-card {
    padding: 0.8rem; background: #0a1628; border-radius: 8px;
    border-left: 3px solid #7c6df2; margin-top: 0.5rem;
}
.info-card-title { color: #e2e8f0; font-weight: 600; }
.info-card-sub   { color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem; }

/* ── API key input box ── */
.api-key-box {
    background: #0a1628; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 1rem; margin-bottom: 0.8rem;
}
.api-key-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #64748b; text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

# API key is read from st.secrets inside agents.py — nothing stored here
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL   = ""

_ss("agents_ready",    False)
_ss("orchestrator",    None)
_ss("active_agent",    "chat")
_ss("messages",        [])
_ss("rag_ingested",    False)
_ss("video_ingested",  False)
_ss("data_loaded",     False)
_ss("data_filename",   "")
_ss("data_shape",      "")
_ss("data_columns",    [])
_ss("_boot_error",     "")
_ss("video_url_saved",   "")
_ss("video_lang_saved",  "en")


# ── Init agents — runs automatically on first load ────────────────────────────
def init_agents():
    from agents import MultiAgentOrchestrator, set_api_key
    set_api_key(OPENROUTER_API_KEY, OPENROUTER_MODEL)
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = MultiAgentOrchestrator()
    st.session_state.agents_ready = True
    st.session_state._boot_error  = ""

if not st.session_state.agents_ready:
    try:
        init_agents()
    except Exception as e:
        st.session_state._boot_error = str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────
def push_msg(role, content, agent="", chart=None, code=None, lang="python",
             sources=None, research_sources=None, queries=None, timestamps=None, delegated=False):
    st.session_state.messages.append({
        "role": role, "content": content, "agent": agent,
        "chart": chart, "code": code, "lang": lang,
        "sources":          sources or [],
        "research_sources": research_sources or [],
        "queries":          queries or [],
        "timestamps":       timestamps or [],
        "delegated":        delegated,
    })


def get_context():
    return {
        "rag_ingested":   st.session_state.rag_ingested,
        "video_ingested": st.session_state.video_ingested,
        "data_loaded":    st.session_state.data_loaded,
        "data_filename":  st.session_state.data_filename,
    }


def render_message(msg):
    role    = msg["role"]
    content = msg["content"]
    agent   = msg.get("agent", "NEXUS")

    if role == "user":
        st.markdown(f"""
<div class="bubble-user">
  <div class="bubble-label">▸ You</div>
  {content}
</div>""", unsafe_allow_html=True)
    else:
        del_tag = '<span class="delegate-tag">↗ delegated</span>' if msg.get("delegated") else ""
        st.markdown(f"""
<div class="bubble-agent">
  <div class="bubble-label">⬡ {agent}{del_tag}</div>
  {content.replace(chr(10), "<br>")}
</div>""", unsafe_allow_html=True)

        if msg.get("chart"):
            st.image(base64.b64decode(msg["chart"]), use_container_width=True)
        if msg.get("code"):
            st.code(msg["code"], language=msg.get("lang", "python"))
        if msg.get("sources"):
            pills = "".join(f'<span class="src-pill">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f'<div style="margin-top:0.4rem">{pills}</div>', unsafe_allow_html=True)
        if msg.get("timestamps"):
            ts_html = "".join(
                f'<a href="{t["yt_link"]}" target="_blank" class="ts-pill">⏱ {t["timestamp"]}</a>'
                for t in msg["timestamps"] if t.get("yt_link")
            )
            if ts_html:
                st.markdown(f'<div style="margin-top:0.4rem">{ts_html}</div>', unsafe_allow_html=True)
        if msg.get("research_sources"):
            with st.expander(f"📚 {len(msg['research_sources'])} Sources"):
                for s in msg["research_sources"][:12]:
                    if s.get("url"):
                        st.markdown(f"- [{s['title']}]({s['url']})")
        if msg.get("queries"):
            with st.expander("🔍 Research Queries Used"):
                for q in msg["queries"]:
                    st.markdown(f"`{q}`")


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo ──────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1rem">
  <div style="font-size:1.8rem;font-weight:800;
    background:linear-gradient(135deg,#7c6df2,#3b82f6,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:-0.02em">⬡ NEXUSRAG</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
    color:#475569;letter-spacing:0.15em;margin-top:0.2rem">MULTI-AGENT AI</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Status ────────────────────────────────────────────
    if st.session_state.agents_ready:
        st.markdown(
            f'<span class="badge on">● ONLINE · {OPENROUTER_MODEL}</span>',
            unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge off">● OFFLINE</span>', unsafe_allow_html=True)
        if st.button("🔄 Retry", use_container_width=True):
            try:
                init_agents(); st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")


    st.markdown("---")

    # ── Agent selector ────────────────────────────────────
    st.markdown("**⬡ Select Agent**")

    AGENTS = [
        ("chat",     "🤖", "General Chatbot",  "Smart hub · memory · auto-delegates"),
        ("rag",      "📄", "Document Q&A",     "PDF · DOCX · TXT semantic search"),
        ("video",    "🎬", "YouTube RAG",      "YouTube URL → Q&A with timestamps"),
        ("data",     "📊", "Data Analyst",     "CSV/Excel analysis + charts"),
        ("code",     "💻", "Code Generator",   "Generate · Explain · Debug"),
        ("research", "🔬", "Web Researcher",   "Multi-step live web research"),
        ("auto",     "🧠", "Auto-Route",       "LLM picks the best agent"),
    ]

    for aid, icon, name, desc in AGENTS:
        is_active = st.session_state.active_agent == aid
        label = f"{icon} **{name}** ←" if is_active else f"{icon} {name}"
        if st.button(label, key=f"sbtn_{aid}", use_container_width=True, help=desc):
            st.session_state.active_agent = aid
            st.rerun()

    st.markdown("---")

    # ── Status badges ─────────────────────────────────────
    st.markdown("**📊 System Status**")
    c1, c2 = st.columns(2)
    c1.markdown(f'<span class="badge {"on" if st.session_state.rag_ingested else "off"}">📄 Docs</span>',   unsafe_allow_html=True)
    c2.markdown(f'<span class="badge {"on" if st.session_state.video_ingested else "off"}">🎬 Video</span>', unsafe_allow_html=True)
    c1.markdown(f'<span class="badge {"on" if st.session_state.data_loaded else "off"}">📊 Data</span>',    unsafe_allow_html=True)
    c2.markdown(f'<span class="badge info">{len(st.session_state.messages)} msgs</span>',                    unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.agents_ready and st.session_state.orchestrator:
            st.session_state.orchestrator.chatbot.clear_history()
        st.rerun()

    if (st.session_state.active_agent == "chat"
            and st.session_state.agents_ready
            and len(st.session_state.messages) > 4):
        if st.button("📝 Summarize Conversation", use_container_width=True):
            with st.spinner("Summarizing..."):
                s = st.session_state.orchestrator.chatbot.get_summary()
            st.info(s)


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
# Force sidebar open on every load via JS
st.markdown("""
<script>
window.addEventListener('load', function() {
    // Find and click the sidebar expand button if sidebar is collapsed
    setTimeout(function() {
        var btns = window.parent.document.querySelectorAll('[data-testid="stSidebarCollapsedControl"] button');
        if (btns.length > 0) { btns[0].click(); }
    }, 300);
});
</script>
<div class="nexus-header">
  <div class="nexus-title">NEXUSRAG</div>
  <div class="nexus-sub">Multi-Agent Intelligence · OpenRouter · LangChain · FAISS</div>
</div>""", unsafe_allow_html=True)


# ── Boot error ────────────────────────────────────────────────────────────────
if st.session_state._boot_error:
    st.error(f"⚠️ {st.session_state._boot_error}")

if not st.session_state.agents_ready:
    st.error("⚠️ Could not connect. Please refresh the page.")
    if st.button("🔄 Retry Connection"):
        try:
            init_agents(); st.rerun()
        except Exception as e:
            st.error(str(e))
    st.stop()


orch   = st.session_state.orchestrator
active = st.session_state.active_agent

# ── Auto-restore video after server restart ────────────────────────────────
if (st.session_state.get("video_ingested")
        and st.session_state.get("video_url_saved")
        and not orch.video_rag.is_ready()):
    try:
        orch.video_rag.ingest(
            st.session_state.video_url_saved,
            language=st.session_state.get("video_lang_saved", "en"),
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab_chat, tab_ingest, tab_viz = st.tabs(["💬 Chat", "📥 Ingest Data", "📊 Visualize"])


# ════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════
with tab_chat:

    AGENT_META = {
        "chat":     ("🤖", "General Chatbot",  "Conversational AI with memory — answers anything, auto-delegates to specialists"),
        "rag":      ("📄", "Document Q&A",     "Semantic search over your uploaded documents"),
        "video":    ("🎬", "YouTube RAG",      "Paste a YouTube URL · transcript fetched · Q&A with timestamps"),
        "data":     ("📊", "Data Analyst",     "Intelligent data analysis with AI-generated visualizations"),
        "code":     ("💻", "Code Generator",   "Generate, explain, or debug code in any language"),
        "research": ("🔬", "Web Researcher",   "Multi-step web research → structured reports"),
        "auto":     ("🧠", "Auto-Route",       "LLM-powered intent detection picks the optimal agent"),
    }
    icon, name, desc = AGENT_META[active]

    st.markdown(f"""
<div class="agent-bar">
  <div class="agent-bar-icon">{icon}</div>
  <div>
    <div class="agent-bar-name">{name}</div>
    <div class="agent-bar-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Per-agent controls ────────────────────────────────
    code_language  = "Python"
    code_mode      = "Generate"
    code_error     = ""
    research_depth = "standard"

    if active == "code":
        cc1, cc2, cc3 = st.columns([2, 2, 3])
        code_language = cc1.selectbox("Language",
            ["Python","JavaScript","TypeScript","Java","C++","Rust","Go","SQL","Bash","Swift","Kotlin"],
            label_visibility="collapsed")
        code_mode = cc2.selectbox("Mode", ["Generate","Explain","Debug"], label_visibility="collapsed")
        if code_mode == "Debug":
            code_error = cc3.text_input("Error (optional)",
                placeholder="Paste error message...", label_visibility="collapsed")

    if active == "research":
        rc1, _ = st.columns([2, 5])
        research_depth = rc1.selectbox("Depth", ["quick","standard","deep"], label_visibility="collapsed")

    # ── Messages ──────────────────────────────────────────
    if st.session_state.messages:
        for msg in st.session_state.messages:
            render_message(msg)
    else:
        welcome = {
            "chat":     "👋 Hi! I'm **NEXUS**, your central AI assistant.\n\nI answer questions directly, and when needed I automatically delegate to the right specialist — documents, video, data, code, or research.\n\nWhat can I help you with?",
            "rag":      "📄 Upload your documents in the **Ingest Data** tab, then ask me anything about them.",
            "video":    "🎬 Paste a **YouTube URL** in the Ingest Data tab, then ask anything — I'll answer with clickable timestamp links.",
            "data":     "📊 Load a CSV or Excel file in the **Ingest Data** tab, then ask for analysis or a chart.",
            "code":     "💻 Tell me what you need:\n\n• **Generate** — describe the code you want\n• **Explain** — paste any code to understand it\n• **Debug** — paste broken code with the error message",
            "research": "🔬 Give me any topic and I'll research it across the web and write a structured report.",
            "auto":     "🧠 Ask anything — I'll automatically pick the best agent for your request.",
        }.get(active, "Ready.")
        st.markdown(f"""
<div class="bubble-agent">
  <div class="bubble-label">⬡ NEXUS · READY</div>
  {welcome.replace(chr(10), "<br>")}
</div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────
    placeholders = {
        "chat":     "Ask me anything — I'll answer or delegate to the right agent...",
        "rag":      "Ask a question about your uploaded documents...",
        "video":    "Ask about the YouTube video — topics, quotes, timestamps...",
        "data":     "Ask for analysis or describe a chart you want...",
        "code":     "Describe the code you need, or paste code to explain/debug...",
        "research": "Enter a topic for deep multi-step research...",
        "auto":     "Ask anything — the best agent will handle it...",
    }

    with st.form("chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([6, 1])
        user_input = fc1.text_area(
            "msg",
            placeholder=placeholders.get(active, "Type here..."),
            height=90,
            label_visibility="collapsed",
        )
        send = fc2.form_submit_button("Send\n⬡", use_container_width=True)

    # ── Process ───────────────────────────────────────────
    if send and user_input.strip():
        push_msg("user", user_input)

        with st.spinner(f"⬡ {name} thinking..."):
            try:
                ctx = get_context()

                if active == "chat":
                    result = orch.chatbot.smart_reply(user_input, orch, context_info=ctx)
                    push_msg("assistant", result["answer"],
                             agent="🤖 General Chatbot",
                             chart=result.get("chart"),
                             code=result.get("code"),
                             lang=result.get("language", "python"),
                             sources=result.get("sources", []),
                             research_sources=result.get("research_sources", []),
                             queries=result.get("queries", []),
                             delegated=result.get("delegated", False))

                elif active == "rag":
                    r = orch.rag.query(user_input)
                    push_msg("assistant", r["answer"], agent="📄 Document Q&A", sources=r.get("sources", []))

                elif active == "video":
                    if not orch.video_rag.is_ready():
                        push_msg("assistant",
                                 "Video not loaded. Go to the Ingest tab, paste a YouTube URL, and click Load & Index Video.",
                                 agent="System")
                    else:
                        r = orch.video_rag.query(user_input)
                        push_msg("assistant", r.get("answer", ""),
                                 agent="🎬 YouTube RAG",
                                 timestamps=r.get("timestamps", []))

                elif active == "data":
                    r = orch.data_analysis.analyze(user_input)
                    push_msg("assistant", r["answer"], agent="📊 Data Analyst", chart=r.get("chart"))

                elif active == "code":
                    if code_mode == "Generate":
                        r = orch.code_gen.generate(user_input, language=code_language)
                        explanation = r.get("explanation") or f"Here's your {code_language} code:"
                        push_msg("assistant", explanation,
                                 agent="💻 Code Generator",
                                 code=r.get("code", ""),
                                 lang=code_language.lower())
                    elif code_mode == "Explain":
                        answer = orch.code_gen.explain(user_input)
                        push_msg("assistant", answer, agent="💻 Code Generator")
                    elif code_mode == "Debug":
                        r = orch.code_gen.debug(user_input, error=code_error)
                        push_msg("assistant", r.get("explanation", "Fixed code:"),
                                 agent="💻 Code Generator",
                                 code=r.get("fixed_code", ""),
                                 lang="python")

                elif active == "research":
                    r = orch.researcher.research(user_input, depth=research_depth)
                    push_msg("assistant", r["report"],
                             agent="🔬 Web Researcher",
                             research_sources=r.get("sources", []),
                             queries=r.get("queries_used", []))

                elif active == "auto":
                    route = orch.route(user_input)
                    if route in ("chat", "general"):
                        result = orch.chatbot.smart_reply(user_input, orch, context_info=ctx)
                        push_msg("assistant", f"*[Auto → 🤖 Chatbot]*\n\n{result['answer']}",
                                 agent="🧠 Auto-Route",
                                 chart=result.get("chart"),
                                 code=result.get("code"),
                                 sources=result.get("sources", []),
                                 research_sources=result.get("research_sources", []))
                    elif route == "rag":
                        r = orch.rag.query(user_input)
                        push_msg("assistant", f"*[Auto → 📄 Docs]*\n\n{r['answer']}",
                                 agent="🧠 Auto-Route", sources=r.get("sources", []))
                    elif route == "video":
                        r = orch.video_rag.query(user_input)
                        push_msg("assistant", f"*[Auto → 🎬 Video]*\n\n{r.get('answer','')}",
                                 agent="🧠 Auto-Route", timestamps=r.get("timestamps", []))
                    elif route == "data":
                        r = orch.data_analysis.analyze(user_input)
                        push_msg("assistant", f"*[Auto → 📊 Data]*\n\n{r['answer']}",
                                 agent="🧠 Auto-Route", chart=r.get("chart"))
                    elif route == "code":
                        r = orch.code_gen.generate(user_input)
                        push_msg("assistant", f"*[Auto → 💻 Code]*\n\n{r.get('explanation','')}",
                                 agent="🧠 Auto-Route",
                                 code=r.get("code", ""), lang="python")
                    elif route == "research":
                        r = orch.researcher.research(user_input)
                        push_msg("assistant", f"*[Auto → 🔬 Research]*\n\n{r['report']}",
                                 agent="🧠 Auto-Route",
                                 research_sources=r.get("sources", []),
                                 queries=r.get("queries_used", []))
                    else:
                        result = orch.chatbot.chat(user_input, ctx)
                        push_msg("assistant", result["answer"], agent="🧠 Auto-Route")

            except Exception as e:
                import traceback
                full_err = traceback.format_exc()
                push_msg("assistant",
                         f"**Error:** {str(e)}",
                         agent="System")

        st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 2 — INGEST DATA
# ════════════════════════════════════════════════════════════
with tab_ingest:
    ic1, ic2 = st.columns([1, 1])

    # Documents
    with ic1:
        st.markdown("### 📄 Documents")
        st.caption("PDF · TXT · MD")
        doc_files = st.file_uploader(
            "Upload docs", accept_multiple_files=True,
            type=["pdf","txt","docx","doc","md"], key="du",
            label_visibility="collapsed")
        if doc_files and st.button("⬡ Ingest Documents", key="b_docs", use_container_width=True):
            paths = []
            for f in doc_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
                tmp.write(f.read()); tmp.close(); paths.append(tmp.name)
            with st.spinner("Building vector index..."):
                try:
                    msg = orch.rag.ingest(paths)
                    st.session_state.rag_ingested = True
                    st.success(f"✅ {msg}")
                    push_msg("assistant",
                             f"**Documents ready:** {', '.join(f.name for f in doc_files)}\n\n"
                             "Go to Chat → 📄 Document Q&A to ask questions.",
                             agent="System")
                except Exception as e:
                    st.error(f"❌ {e}")
        if st.session_state.rag_ingested:
            st.markdown('<span class="badge on">● Documents Ready</span>', unsafe_allow_html=True)

    # Dataset
    with ic2:
        st.markdown("### 📊 Dataset")
        st.caption("CSV · Excel · JSON")
        data_file = st.file_uploader(
            "Upload data", type=["csv","xlsx","xls","json"],
            key="dfu", label_visibility="collapsed")
        if data_file and st.button("⬡ Load Dataset", key="b_data", use_container_width=True):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(data_file.name).suffix)
            tmp.write(data_file.read()); tmp.close()
            try:
                msg = orch.data_analysis.load_data(tmp.name)
                df  = orch.data_analysis.df
                st.session_state.data_loaded   = True
                st.session_state.data_filename = data_file.name
                st.session_state.data_shape    = f"{df.shape[0]:,} × {df.shape[1]}"
                st.session_state.data_columns  = df.columns.tolist()
                st.success(f"✅ {msg}")
                st.dataframe(df.head(8), use_container_width=True)
                push_msg("assistant",
                         f"**Dataset loaded:** `{data_file.name}` — {df.shape[0]:,} rows × {df.shape[1]} cols\n\n"
                         "Go to Chat → 📊 Data Analyst or the Visualize tab.",
                         agent="System")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.session_state.data_loaded:
            st.markdown('<span class="badge on">● Data Ready</span>', unsafe_allow_html=True)
            with st.expander("📋 Data Summary"):
                st.text(orch.data_analysis.get_summary())

    st.markdown("<hr>", unsafe_allow_html=True)

    # YouTube
    st.markdown("### 🎬 YouTube Video")
    st.caption("Paste any YouTube link — transcript fetched automatically, no download needed")

    yt_col1, yt_col2 = st.columns([3, 1])
    yt_url  = yt_col1.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...  or  https://youtu.be/...",
        label_visibility="collapsed", key="yt_url_input")
    yt_lang = yt_col2.selectbox(
        "Lang",
        ["en","hi","es","fr","de","ja","ko","pt","ar","ru"],
        key="yt_lang", label_visibility="collapsed")

    if yt_url:
        _m = re.search(r'(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_\-]{11})', yt_url)
        if _m:
            _vid = _m.group(1)
            st.markdown(f"""
<div style="display:flex;gap:1rem;align-items:center;margin:0.5rem 0;
            padding:0.8rem;background:#0f1829;border-radius:8px;border:1px solid #1e3a5f;
            flex-wrap:wrap">
  <img src="https://img.youtube.com/vi/{_vid}/mqdefault.jpg"
       style="width:140px;border-radius:6px;flex-shrink:0">
  <div style="min-width:0">
    <div style="font-size:0.7rem;color:#94a3b8;font-family:Space Mono,monospace">VIDEO ID: {_vid}</div>
    <a href="{yt_url}" target="_blank"
       style="color:#7c6df2;font-size:0.82rem;word-break:break-all">{yt_url}</a>
  </div>
</div>""", unsafe_allow_html=True)

        if st.button("⬡ Load & Index YouTube Video", key="b_yt", use_container_width=True):
            with st.spinner("Fetching transcript and indexing..."):
                try:
                    msg  = orch.video_rag.ingest(yt_url, language=yt_lang)
                    info = orch.video_rag.get_info()
                    if orch.video_rag.is_ready():
                        st.session_state.video_ingested  = True
                        st.session_state.video_url_saved = yt_url
                        st.session_state.video_lang_saved = yt_lang
                        st.success(f"✅ {msg}")
                    else:
                        st.session_state.video_ingested = False
                        st.error(f"❌ {msg}")
                    if info.get("title"):
                        st.markdown(f"""
<div class="info-card">
  <div class="info-card-title">📹 {info["title"]}</div>
  <div class="info-card-sub">
    📺 {info.get("channel","?")} &nbsp;·&nbsp;
    ⏱ {info.get("duration","?")} &nbsp;·&nbsp;
    💬 {info.get("transcript_segments",0)} segments indexed
  </div>
</div>""", unsafe_allow_html=True)
                    push_msg("assistant",
                             f"**YouTube video ready:** {info.get('title', yt_url)}\n\n"
                             f"Channel: {info.get('channel','?')} · {info.get('transcript_segments',0)} segments indexed.\n\n"
                             "Go to Chat → 🎬 YouTube RAG to ask questions with timestamps.",
                             agent="System")
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state.video_ingested:
        st.markdown('<span class="badge on">● YouTube Video Ready</span>', unsafe_allow_html=True)
        info = orch.video_rag.get_info()
        if info.get("title"):
            m1, m2, m3 = st.columns(3)
            t = info["title"]
            m1.metric("Title",    (t[:22]+"...") if len(t) > 22 else t)
            m2.metric("Duration", info.get("duration", "?"))
            m3.metric("Segments", info.get("transcript_segments", 0))

            s1, s2 = st.columns(2)
            if s1.button("📝 Brief Summary", key="yt_brief", use_container_width=True):
                with st.spinner("Summarizing..."):
                    r = orch.video_rag.summarize("brief")
                    push_msg("assistant",
                             f"### 📹 {r.get('title','Video')} — Summary\n\n{r['summary']}",
                             agent="🎬 YouTube RAG")
                st.rerun()
            if s2.button("📋 Full Summary", key="yt_full", use_container_width=True):
                with st.spinner("Summarizing..."):
                    r = orch.video_rag.summarize("detailed")
                    push_msg("assistant",
                             f"### 📹 {r.get('title','Video')} — Detailed Summary\n\n{r['summary']}",
                             agent="🎬 YouTube RAG")
                st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 3 — VISUALIZE
# ════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("### 📊 Visualization Studio")

    if not st.session_state.data_loaded:
        st.markdown("""
<div class="empty-state">
  <div style="font-size:2.5rem">📊</div>
  <div style="margin-top:0.8rem">Load a dataset in the Ingest Data tab first</div>
  <div style="font-size:0.7rem;margin-top:0.4rem">Supports CSV · Excel · JSON</div>
</div>""", unsafe_allow_html=True)
    else:
        df       = orch.data_analysis.df
        cols     = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        st.markdown(
            f'<span class="badge on">● {orch.data_analysis.file_name}</span> '
            f'<span class="badge info">{df.shape[0]:,} rows × {df.shape[1]} cols</span>',
            unsafe_allow_html=True)
        st.markdown("")

        # AI chart
        st.markdown("**🤖 AI Chart Builder**")
        st.caption("Describe what you want — AI picks the best chart type and columns automatically")
        ai1, ai2 = st.columns([5, 1])
        ai_prompt = ai1.text_input(
            "AI chart query",
            placeholder="e.g. Bar chart of average salary by department",
            label_visibility="collapsed", key="ai_chart_q")
        if ai2.button("⬡ Go", use_container_width=True, key="ai_go") and ai_prompt:
            with st.spinner("Analyzing and rendering..."):
                try:
                    r = orch.data_analysis.analyze(ai_prompt)
                    if r.get("answer"):
                        st.info(r["answer"])
                    if r.get("chart"):
                        st.image(base64.b64decode(r["chart"]), use_container_width=True)
                    else:
                        st.warning("No chart generated. Try a different description.")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Manual chart
        st.markdown("**🎛 Manual Chart Builder**")
        vc1, vc2, vc3, vc4 = st.columns([2, 2, 2, 2])
        chart_type  = vc1.selectbox("Type",  ["bar","line","scatter","histogram","pie","heatmap","box"])
        x_col       = vc2.selectbox("X Axis", ["(auto)"] + cols)
        y_col       = vc3.selectbox("Y Axis", ["(auto)"] + num_cols if num_cols else ["(auto)"])
        chart_title = vc4.text_input("Title", placeholder="Chart title...", label_visibility="collapsed")

        if st.button("⬡ Render Chart", use_container_width=True, key="manual_chart"):
            x = None if x_col == "(auto)" else x_col
            y = None if y_col == "(auto)" else y_col
            with st.spinner("Rendering..."):
                try:
                    b64 = orch.data_analysis.custom_chart(
                        chart_type, x, y, chart_title or chart_type.title())
                    if b64:
                        st.image(base64.b64decode(b64), use_container_width=True)
                    else:
                        st.error("Chart failed. Try different columns.")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Data preview
        pc1, pc2 = st.columns([3, 2])
        with pc1:
            st.markdown("**Data Preview**")
            n_rows = st.slider("Rows", 5, min(200, len(df)), 10)
            st.dataframe(df.head(n_rows), use_container_width=True)
        with pc2:
            st.markdown("**Numeric Summary**")
            st.dataframe(df.describe(), use_container_width=True)
