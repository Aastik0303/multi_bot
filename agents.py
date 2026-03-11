"""
agents.py — NexusRAG Agent Stack
Uses OpenRouter API (OpenAI-compatible).
No API key pool. Single key set at runtime via set_api_key().
"""

from __future__ import annotations

import io, base64, json, re, subprocess, sys, tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# OpenAI-compatible client pointing at OpenRouter
from openai import OpenAI

from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    LOADERS_OK = True
except ImportError:
    LOADERS_OK = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_OK = True
except ImportError:
    TRANSCRIPT_OK = False

try:
    import yt_dlp as _yt_dlp; YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

try:
    import requests as _req; REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from duckduckgo_search import DDGS; DDGS_OK = True
except ImportError:
    DDGS_OK = False


# ── OPENROUTER CLIENT ─────────────────────────────────────────────────────────

OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
_client: Optional[OpenAI] = None


def set_api_key(key: str, model: str = "google/gemini-2.0-flash-exp:free"):
    """Call this once with the user's OpenRouter API key before using any agent."""
    global OPENROUTER_MODEL, _client
    OPENROUTER_MODEL = model
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key.strip(),
    )


def llm_call(messages: List[Dict], temperature: float = 0.1) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant text or an error string.
    """
    if _client is None:
        return "Error: API key not set. Please enter your OpenRouter API key in the sidebar."
    try:
        resp = _client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        err = str(e)
        if any(x in err.lower() for x in ("401", "403", "invalid", "unauthorized")):
            return "Error: Invalid API key. Please check your OpenRouter API key."
        if any(x in err.lower() for x in ("429", "quota", "rate")):
            return "Error: Rate limit reached. Please wait a moment and try again."
        return f"Error: {err}"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── VECTOR STORE HELPERS ───────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())


def load_documents(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        ext = Path(p).suffix.lower()
        try:
            if ext == ".pdf" and LOADERS_OK:
                docs.extend(PyPDFLoader(p).load())
            elif ext in (".txt", ".md") and LOADERS_OK:
                docs.extend(TextLoader(p, encoding="utf-8").load())
            elif ext == ".csv" and LOADERS_OK:
                docs.extend(CSVLoader(p).load())
            else:
                text = Path(p).read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": p}))
        except Exception as e:
            docs.append(Document(page_content=f"Error loading {p}: {e}", metadata={"source": p}))
    return docs


# ── CHART HELPERS ──────────────────────────────────────────────────────────────

def _dark(ax, fig):
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── RAG AGENT ─────────────────────────────────────────────────────────────────

class RAGAgent:
    name = "RAG Agent"

    def __init__(self):
        self._vs = None
        self._sources: List[str] = []

    def ingest(self, file_paths: List[str]) -> str:
        docs = load_documents(file_paths)
        if not docs:
            return "No documents loaded."
        self._vs      = build_vectorstore(docs)
        self._sources = list({d.metadata.get("source", "?") for d in docs})
        return f"Ingested {len(docs)} chunks from {len(file_paths)} file(s)."

    def query(self, question: str) -> Dict:
        if self._vs is None:
            return {"answer": "No documents loaded yet. Please upload files first.", "sources": []}
        docs = self._vs.as_retriever(search_kwargs={"k": 5}).invoke(question)
        if not docs:
            return {"answer": "No relevant content found.", "sources": []}
        context = "\n\n".join(d.page_content for d in docs)
        sources  = list({d.metadata.get("source", "?") for d in docs})
        answer   = llm_call([
            {"role": "system", "content": "Answer based ONLY on the context provided. If the answer is not in the context, say so clearly."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ])
        return {"answer": answer, "sources": sources}


# ── YOUTUBE RAG AGENT ─────────────────────────────────────────────────────────

class VideoRAGAgent:
    """
    YouTube RAG Agent.
    Tries all transcript methods silently.
    Falls back to AI-generated topic segments if no transcript — never errors out.
    """
    name = "YouTube RAG Agent"

    def __init__(self):
        self._vs:          Optional[FAISS] = None
        self._chunks:      list  = []
        self._transcript:  str   = ""
        self.video_id:     str   = ""
        self.video_url:    str   = ""
        self.title:        str   = "Unknown"
        self.channel:      str   = "Unknown"
        self.duration:     str   = "Unknown"
        self.thumbnail:    str   = ""
        self.source_type:  str   = "unknown"   # "transcript" | "ai_generated"

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        for pattern in [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
            r"(?:embed/)([0-9A-Za-z_-]{11})",
        ]:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _secs_to_ts(seconds: float) -> str:
        s = int(seconds)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    @staticmethod
    def _parse_seg(seg) -> tuple:
        if isinstance(seg, dict):
            return seg.get("start", 0), seg.get("duration", 0), seg.get("text", "").strip()
        return getattr(seg, "start", 0), getattr(seg, "duration", 0), getattr(seg, "text", "").strip()

    def _get_metadata(self, video_id: str):
        if YTDLP_OK:
            try:
                opts = {"quiet": True, "no_warnings": True, "skip_download": True}
                with _yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info("https://www.youtube.com/watch?v=" + video_id, download=False)
                self.title    = info.get("title",    "Unknown")
                self.channel  = info.get("uploader", "Unknown")
                dur           = info.get("duration", 0)
                self.duration = self._secs_to_ts(dur) if dur else "Unknown"
                return
            except Exception:
                pass
        if REQUESTS_OK:
            try:
                r = _req.get(
                    "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v="
                    + video_id + "&format=json", timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    self.title   = d.get("title",       "Unknown")
                    self.channel = d.get("author_name", "Unknown")
            except Exception:
                pass

    def _fetch_transcript(self, video_id: str, language: str = "en") -> list:
        if not TRANSCRIPT_OK:
            return []

        def _parse_all(result) -> list:
            out = []
            for seg in result:
                s, d, txt = self._parse_seg(seg)
                if str(txt).strip():
                    out.append({"start": float(s), "duration": float(d), "text": str(txt).strip()})
            return out

        # Attempt 1: direct get_transcript
        for lang_list in [[language], ["en"], ["en-US"], ["en-GB"], [language, "en"]]:
            try:
                raw = _parse_all(YouTubeTranscriptApi.get_transcript(video_id, languages=lang_list))
                if raw:
                    return raw
            except Exception:
                continue

        # Attempt 2: instance-based API (youtube-transcript-api >= 1.0)
        try:
            api = YouTubeTranscriptApi()
            for lang_list in [[language, "en"], ["en"], []]:
                try:
                    result = api.fetch(video_id, languages=lang_list) if lang_list else api.fetch(video_id)
                    raw = _parse_all(result)
                    if raw:
                        return raw
                except Exception:
                    continue
        except Exception:
            pass

        # Attempt 3: list all transcripts, try each
        try:
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            all_t = list(tlist)

            def _priority(t):
                lm = t.language_code.startswith(language) or t.language_code.startswith("en")
                return (0 if not t.is_generated else 1, 0 if lm else 1)

            all_t.sort(key=_priority)
            for t_obj in all_t:
                try:
                    raw = _parse_all(t_obj.fetch())
                    if raw:
                        return raw
                except Exception:
                    continue
            if all_t:
                try:
                    raw = _parse_all(all_t[0].translate("en").fetch())
                    if raw:
                        return raw
                except Exception:
                    pass
        except Exception:
            pass

        return []

    def _ai_fallback_chunks(self) -> List[Document]:
        response = llm_call([
            {"role": "user", "content": (
                f"You are analysing a YouTube video.\n"
                f"Title: {self.title}\nChannel: {self.channel}\n\n"
                "Generate 12 detailed topic segments that likely appear in this video.\n"
                "Format EACH segment on its own line as:\n"
                "[Topic N] <one or two sentence description>\n\n"
                "Generate 12 segments now:"
            )},
        ], temperature=0.3)
        lines = [l.strip() for l in response.splitlines() if l.strip().startswith("[")]
        if not lines:
            lines = [f"[Overview] Video titled '{self.title}' by {self.channel}."]
        docs = []
        for i, line in enumerate(lines):
            docs.append(Document(
                page_content=line,
                metadata={"start_sec": i * 60, "timestamp": self._secs_to_ts(i * 60), "source": self.video_url},
            ))
        return docs

    def _chunks_from_transcript(self, raw: list, chunk_secs: int = 60) -> List[Document]:
        docs = []
        cur_text, cur_start, cur_end = [], None, 0.0
        for seg in raw:
            s, d, txt = seg["start"], seg["duration"], seg["text"]
            if cur_start is None:
                cur_start = s
            cur_text.append(txt)
            cur_end = s + d
            if (cur_end - cur_start) >= chunk_secs:
                ts = self._secs_to_ts(cur_start)
                docs.append(Document(
                    page_content="[" + ts + "] " + " ".join(cur_text),
                    metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url},
                ))
                cur_text, cur_start = [], None
        if cur_text and cur_start is not None:
            ts = self._secs_to_ts(cur_start)
            docs.append(Document(
                page_content="[" + ts + "] " + " ".join(cur_text),
                metadata={"start_sec": cur_start, "timestamp": ts, "source": self.video_url},
            ))
        return docs

    def ingest(self, youtube_url: str, language: str = "en") -> str:
        self._vs = None; self._chunks = []; self._transcript = ""
        self.title = self.channel = self.duration = "Unknown"
        self.source_type = "unknown"

        vid = self.extract_video_id(youtube_url)
        if not vid:
            return "Error: Could not extract video ID from URL."

        self.video_id  = vid
        self.video_url = "https://www.youtube.com/watch?v=" + vid
        self.thumbnail = "https://img.youtube.com/vi/" + vid + "/hqdefault.jpg"
        self._get_metadata(vid)

        raw = self._fetch_transcript(vid, language)
        if raw:
            self._chunks     = raw
            self._transcript = "\n".join("[" + self._secs_to_ts(s["start"]) + "] " + s["text"] for s in raw)
            docs             = self._chunks_from_transcript(raw)
            self.source_type = "transcript"
            source_note      = f"{len(raw)} transcript segments"
        else:
            docs             = self._ai_fallback_chunks()
            self._transcript = "\n".join(d.page_content for d in docs)
            self.source_type = "ai_generated"
            source_note      = f"{len(docs)} AI-generated segments (transcript unavailable)"

        if not docs:
            return "Error: Could not build content chunks."
        try:
            self._vs = build_vectorstore(docs)
        except Exception as e:
            return "Error building index: " + str(e)

        return "Loaded: " + self.title + " | " + source_note + " | " + str(len(docs)) + " chunks indexed."

    def is_ready(self) -> bool:
        return self._vs is not None

    def query(self, question: str) -> Dict:
        if not self.is_ready():
            return {"answer": "Video not loaded. Please paste a YouTube URL and click Load.", "timestamps": []}
        docs       = self._vs.as_retriever(search_kwargs={"k": 5}).invoke(question)
        context    = "\n\n".join(d.page_content for d in docs)
        timestamps = [
            {"timestamp": d.metadata.get("timestamp", ""),
             "yt_link":   self.video_url + "&t=" + str(int(d.metadata.get("start_sec", 0))) + "s"}
            for d in docs
        ]
        src_note = ("Context is from the real video transcript." if self.source_type == "transcript"
                    else "Note: No transcript — context is AI-generated from video title/channel.")
        answer = llm_call([
            {"role": "system", "content": (
                "You are a helpful YouTube video assistant. " + src_note +
                " Answer using ONLY the context. Cite timestamps like [MM:SS]. If not in context, say so."
            )},
            {"role": "user", "content": (
                f'Video: "{self.title}" by {self.channel}\n\n'
                f"Relevant segments:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )},
        ])
        return {"answer": answer, "timestamps": timestamps, "video_url": self.video_url, "source_type": self.source_type}

    def summarize(self, style: str = "detailed") -> Dict:
        if not self._transcript:
            return {"summary": "No video loaded.", "title": ""}
        excerpt = self._transcript[:12000]
        instr = {
            "brief":   "Write a brief 3-5 sentence summary.",
            "bullets": "List the 10 most important points as bullet points with timestamps.",
        }.get(style, "Write a structured summary: Overview, Key Topics, Main Insights, Conclusion.")
        summary = llm_call([
            {"role": "user", "content": f'Video: "{self.title}" by {self.channel}\n\nContent:\n{excerpt}\n\n{instr}'},
        ], temperature=0.2)
        return {"summary": summary, "title": self.title, "channel": self.channel,
                "video_url": self.video_url, "thumbnail": self.thumbnail, "source_type": self.source_type}

    def get_info(self) -> Dict:
        return {
            "video_id":            self.video_id,
            "title":               self.title,
            "channel":             self.channel,
            "duration":            self.duration,
            "video_url":           self.video_url,
            "thumbnail":           self.thumbnail,
            "transcript_segments": len(self._chunks),
            "indexed":             self._vs is not None,
            "source_type":         self.source_type,
        }

    @staticmethod
    def is_youtube_url(text: str) -> bool:
        return bool(re.search(r"(youtube\.com|youtu\.be)", text, re.IGNORECASE))


# ── DATA ANALYSIS AGENT ───────────────────────────────────────────────────────

class DataAnalysisAgent:
    name = "Data Analysis Agent"

    def __init__(self):
        self.df:        Optional[pd.DataFrame] = None
        self.file_name: str = ""

    def load_data(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".csv":              self.df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"): self.df = pd.read_excel(path)
            elif ext == ".json":           self.df = pd.read_json(path)
            else:                          return f"Unsupported format: {ext}"
        except Exception as e:
            return f"Error: {e}"
        self.file_name = Path(path).name
        return (f"Loaded '{self.file_name}': {self.df.shape[0]:,} rows × {self.df.shape[1]} cols. "
                f"Columns: {', '.join(self.df.columns.tolist())}")

    def get_summary(self) -> str:
        if self.df is None:
            return "No data loaded."
        buf = io.StringIO()
        self.df.info(buf=buf)
        return (f"Shape: {self.df.shape}\n\nStats:\n{self.df.describe(include='all').to_string()}\n\n"
                f"Nulls:\n{self.df.isnull().sum().to_string()}")

    def analyze(self, question: str) -> Dict:
        if self.df is None:
            return {"answer": "No data loaded. Please upload a file first.", "chart": None}
        sample = self.df.head(5).to_string()
        dtypes = self.df.dtypes.to_string()
        resp = llm_call([
            {"role": "system", "content": "You are a data analyst. Reply in JSON only — no markdown fences."},
            {"role": "user",   "content": (
                f"Question: {question}\n\nColumns:\n{dtypes}\n\nSample rows:\n{sample}\n\n"
                'Reply with exactly this JSON structure:\n'
                '{"answer":"analysis text","chart_type":"bar|line|scatter|histogram|pie|heatmap|box",'
                '"x_col":"column name or null","y_col":"numeric column name or null","title":"chart title"}'
            )},
        ], temperature=0.2)
        try:
            m    = re.search(r'\{.*\}', resp, re.DOTALL)
            plan = json.loads(m.group()) if m else {"answer": resp}
        except Exception:
            return {"answer": resp, "chart": None}
        chart = None
        if plan.get("chart_type"):
            chart = self.custom_chart(plan.get("chart_type", "bar"),
                                      plan.get("x_col"), plan.get("y_col"),
                                      plan.get("title", "Chart"))
        return {"answer": plan.get("answer", resp), "chart": chart}

    def custom_chart(self, chart_type: str, x_col=None, y_col=None, title="Chart") -> Optional[str]:
        if self.df is None:
            return None
        df  = self.df
        num = df.select_dtypes(include=np.number).columns.tolist()
        cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if x_col not in df.columns: x_col = cat[0] if cat else (num[0] if num else None)
        if y_col not in df.columns: y_col = num[0] if num else None
        pal = sns.color_palette("viridis", 12)
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        _dark(ax, fig)
        try:
            if chart_type == "bar" and x_col and y_col:
                d = df.groupby(x_col)[y_col].mean().reset_index().head(15)
                ax.bar(d[x_col].astype(str), d[y_col], color=pal)
                plt.xticks(rotation=45, ha="right", color="white")
            elif chart_type == "line" and y_col:
                s = df[y_col].dropna().head(150)
                ax.plot(range(len(s)), s.values, color="#7c6df2", linewidth=2)
            elif chart_type == "scatter" and x_col and y_col:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color=pal[3], s=40)
            elif chart_type == "histogram" and y_col:
                ax.hist(df[y_col].dropna(), bins=30, color=pal[4])
            elif chart_type == "pie" and x_col:
                c = df[x_col].value_counts().head(8)
                ax.pie(c.values, labels=c.index.astype(str), autopct="%1.1f%%",
                       colors=pal, textprops={"color": "white"})
            elif chart_type == "heatmap" and len(num) >= 2:
                sns.heatmap(df[num].corr(), ax=ax, cmap="viridis", annot=True, fmt=".2f")
            elif chart_type == "box" and y_col:
                ax.boxplot(df[y_col].dropna(), patch_artist=True, boxprops=dict(facecolor=pal[1]))
            else:
                if num: ax.bar(df[num].mean().index, df[num].mean().values, color=pal)
            ax.set_title(title, color="white", fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f"Chart error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        return _b64(fig)


# ── CODE GENERATOR AGENT ──────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    m = re.search(r"```[\w]*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class CodeGeneratorAgent:
    name = "Code Generator"

    def generate(self, request: str, language: str = "Python") -> Dict:
        resp = llm_call([
            {"role": "system", "content": f"You are an expert {language} programmer. Write clean, concise, commented code. Always wrap code in a code block."},
            {"role": "user",   "content": f"Write {language} code for: {request}\n\nProvide the code block then a short explanation."},
        ], temperature=0.2)
        code        = _extract_code(resp)
        explanation = re.sub(r"```[\w]*\n?.*?```", "", resp, flags=re.DOTALL).strip()
        return {"code": code, "explanation": explanation, "language": language}

    def explain(self, code: str) -> str:
        return llm_call([
            {"role": "system", "content": "Explain code clearly and simply for a beginner."},
            {"role": "user",   "content": f"Explain this code:\n\n```\n{code}\n```"},
        ])

    def debug(self, code: str, error: str = "") -> Dict:
        resp = llm_call([
            {"role": "system", "content": "You are a debugging expert. Identify the bug and return the fixed code in a code block."},
            {"role": "user",   "content": f"Error: {error or 'unknown'}\n\nCode:\n```\n{code}\n```\n\nExplain the bug and show the fixed code."},
        ])
        return {"fixed_code": _extract_code(resp), "explanation": resp}

    def run(self, python_code: str) -> Dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            tmp = f.name
        try:
            p = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=10)
            return {"stdout": p.stdout[:2000], "stderr": p.stderr[:500], "success": p.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Timed out.", "success": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "success": False}


# ── DEEP RESEARCHER AGENT ─────────────────────────────────────────────────────

class DeepResearcherAgent:
    name = "Deep Researcher"

    def _search(self, query: str, n: int = 5) -> List[Dict]:
        if not DDGS_OK:
            return []
        try:
            with DDGS() as ddgs:
                return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                        for r in ddgs.text(query, max_results=n)]
        except Exception:
            return []

    def research(self, topic: str, depth: str = "standard") -> Dict:
        n = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)
        q_resp = llm_call([
            {"role": "user", "content": f"Generate {n} search queries for: '{topic}'\nReturn ONLY a JSON array of {n} strings."},
        ], temperature=0.3)
        try:
            m       = re.search(r'\[.*?\]', q_resp, re.DOTALL)
            queries = json.loads(m.group()) if m else [topic]
        except Exception:
            queries = [topic]

        all_results = []
        for q in queries[:n]:
            all_results.extend(self._search(q, 4))

        if not all_results:
            report = llm_call([
                {"role": "user", "content": f"Write a comprehensive research report on: {topic}\nInclude: Overview, Key Facts, Analysis, Conclusion."},
            ], temperature=0.2)
            return {"report": report, "queries_used": queries, "sources_found": 0, "sources": []}

        context = "\n\n".join(
            f"[{i+1}] {r['title']}\n{r['snippet']}\n{r['url']}"
            for i, r in enumerate(all_results[:15])
        )
        report = llm_call([
            {"role": "user", "content": (
                f"Write a comprehensive markdown report on: **{topic}**\n\n"
                "Sections: ## Overview, ## Key Findings, ## Analysis, ## Conclusion, ## Sources\n\n"
                f"Research data:\n{context}"
            )},
        ], temperature=0.2)
        sources = [{"title": r["title"], "url": r["url"]} for r in all_results if r.get("url")][:12]
        return {"report": report, "queries_used": queries, "sources_found": len(all_results), "sources": sources}


# ── GENERAL CHATBOT ───────────────────────────────────────────────────────────

_PERSONA = ("You are NEXUS, a helpful AI assistant. Be concise and friendly. "
            "You have specialists available: documents, YouTube, data analysis, code, research.")


class GeneralChatbotAgent:
    name = "General Chatbot"

    def __init__(self):
        self._history: List[Dict] = []

    def chat(self, message: str, context_info: Dict = None) -> Dict:
        msgs = [{"role": "system", "content": _PERSONA}]
        for t in self._history[-20:]:
            msgs.append({"role": t["role"], "content": t["content"]})
        msgs.append({"role": "user", "content": message})
        reply = llm_call(msgs, temperature=0.7)
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": reply})
        return {"answer": reply}

    def detect_intent(self, message: str, context_info: Dict = None) -> str:
        ctx  = context_info or {}
        resp = llm_call([
            {"role": "user", "content": (
                f'Message: "{message}"\nLoaded context: {json.dumps(ctx)}\n\n'
                'Which agent is best? Reply ONE word only:\n'
                '"direct" | "rag" | "video" | "data" | "code" | "research"\n'
                'Use rag/video/data only if those are loaded in context.'
            )},
        ], temperature=0.0)
        intent = resp.strip().lower().split()[0] if resp.strip() else "direct"
        return intent if intent in {"direct", "rag", "video", "data", "code", "research"} else "direct"

    def smart_reply(self, message: str, orchestrator: Any, context_info: Dict = None) -> Dict:
        ctx    = context_info or {}
        intent = self.detect_intent(message, ctx)
        if intent == "direct":
            return self.chat(message, ctx)
        result = {"delegated": True, "intent": intent}
        try:
            if intent == "rag":
                r = orchestrator.rag.query(message)
                result.update({"answer": r["answer"], "sources": r.get("sources", [])})
            elif intent == "video":
                r = orchestrator.video_rag.query(message)
                result["answer"] = r.get("answer", "")
            elif intent == "data":
                r = orchestrator.data_analysis.analyze(message)
                result.update({"answer": r["answer"], "chart": r.get("chart")})
            elif intent == "code":
                r = orchestrator.code_gen.generate(message)
                result.update({"answer": r.get("explanation", ""), "code": r.get("code", ""), "language": "python"})
            elif intent == "research":
                r = orchestrator.researcher.research(message)
                result.update({"answer": r["report"], "research_sources": r.get("sources", []),
                               "queries": r.get("queries_used", [])})
        except Exception:
            return self.chat(message, ctx)
        self._history.append({"role": "user",      "content": message})
        self._history.append({"role": "assistant", "content": result.get("answer", "")})
        return result

    def clear_history(self):
        self._history.clear()

    def get_summary(self) -> str:
        if not self._history:
            return "No conversation yet."
        excerpt = "\n".join(f"{t['role'].upper()}: {t['content'][:200]}" for t in self._history[-20:])
        return llm_call([{"role": "user", "content": f"Summarize this conversation in 3-5 sentences:\n\n{excerpt}"}])


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    def __init__(self):
        self.rag           = RAGAgent()
        self.video_rag     = VideoRAGAgent()
        self.data_analysis = DataAnalysisAgent()
        self.code_gen      = CodeGeneratorAgent()
        self.researcher    = DeepResearcherAgent()
        self.chatbot       = GeneralChatbotAgent()

    def route(self, query: str) -> str:
        resp = llm_call([
            {"role": "user", "content": f'Classify: "{query}"\nOptions: rag|video|data|code|research|chat\nOne word only.'},
        ], temperature=0.0).strip().lower().split()[0]
        return resp if resp in {"rag", "video", "data", "code", "research", "chat"} else "chat"
