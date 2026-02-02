import streamlit as st
import os, json, pickle, datetime, io, time
import pandas as pd
import numpy as np
import faiss

from typing import TypedDict, List, Annotated
from sentence_transformers import SentenceTransformer
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# ---------------- LangGraph ----------------
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------------- LLM ----------------
from openai import OpenAI

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Defense Agentic Extraction", page_icon="🛡️", layout="wide")

# ==========================================================
# 2. REQUIRED COLUMNS
# ==========================================================
REQUIRED_COLUMNS = [
    "Customer Region", "Customer Country", "Customer Operator",
    "Supplier Region", "Supplier Country", "Domestic Content",
    "Market Segment", "System Type (General)", "System Type (Specific)",
    "System Name (General)", "System Name (Specific)", "System Piloting",
    "Supplier Name", "Program Type", "Expected MRO Contract Duration (Months)",
    "Quantity", "Value Certainty", "Value (Million)", "Currency",
    "Value (USD$ Million)", "Value Note\n(If Any)", "G2G/B2G",
    "Signing Month", "Signing Year", "Description of Contract",
    "Additional Notes (Internal Only)", "Source Link(s)",
    "Contract Date", "Reported Date (By SGA)",
    "KB Validation Notes", "Confidence Score"
]

# ==========================================================
# 3. PROMPTS (UNCHANGED)
# ==========================================================
GEOGRAPHY_PROMPT = """You are a Defense Geography Analyst... (same as yours)"""
SYSTEM_CLASSIFIER_PROMPT = """You are a Senior Defense System Classification Analyst..."""
CONTRACT_EXTRACTOR_PROMPT = """You are a Defense Contract Financial Analyst..."""

# ==========================================================
# 4. KB LOADING
# ==========================================================
@st.cache_resource
def load_kb_resources(kb_dir):
    if not kb_dir or not os.path.exists(kb_dir):
        return None
    index = faiss.read_index(os.path.join(kb_dir, "system_kb.faiss"))
    with open(os.path.join(kb_dir, "system_kb_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return {"index": index, "meta": meta, "embedder": embedder}

def get_kb_hit(resources, text):
    if not resources:
        return {}, 0.0
    emb = resources["embedder"].encode([text], normalize_embeddings=True).astype("float32")
    scores, idxs = resources["index"].search(emb, 1)
    if idxs[0][0] < 0:
        return {}, 0.0
    return resources["meta"][idxs[0][0]], float(scores[0][0])

def call_llm(prompt, key, provider):
    client = OpenAI(
        api_key=key if provider == "openrouter" else f"{key}:agentic",
        base_url="https://openrouter.ai/api/v1"
        if provider == "openrouter"
        else "https://llmfoundry.straive.com/openai/v1/"
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(r.choices[0].message.content)

# ==========================================================
# 5. STATE
# ==========================================================
class AgentState(TypedDict):
    text: str
    date: str
    url: str
    kb_meta: dict
    kb_score: float
    row: dict
    rows: list
    messages: Annotated[List[AnyMessage], add_messages]

# ==========================================================
# 6. WORKFLOW (LLM FIRST → KB VALIDATION)
# ==========================================================
def build_workflow(kb_resources, foundry_key, openrouter_key):

    def kb_router(state: AgentState):
        kb, score = get_kb_hit(kb_resources, state["text"])
        return {**state, "kb_meta": kb, "kb_score": score}

    def sourcing(state: AgentState):
        return {
            **state,
            "row": {
                "Description of Contract": state["text"],
                "Contract Date": state["date"],
                "Source Link(s)": state["url"],
                "Reported Date (By SGA)": datetime.date.today().isoformat()
            }
        }

    def geography(state: AgentState):
        row = state["row"].copy()
        res = call_llm(
            GEOGRAPHY_PROMPT + "\n" + state["text"],
            foundry_key or openrouter_key,
            "foundry" if foundry_key else "openrouter"
        )
        row.update(res)
        return {**state, "row": row}

    def system(state: AgentState):
        row = state["row"].copy()
        res = call_llm(
            SYSTEM_CLASSIFIER_PROMPT + "\n" + state["text"],
            foundry_key or openrouter_key,
            "foundry" if foundry_key else "openrouter"
        )
        row.update(res)
        return {**state, "row": row}

    def contract(state: AgentState):
        row = state["row"].copy()
        ctx = f"Reference Date: {state['date']}\n\n{state['text']}"
        res = call_llm(
            CONTRACT_EXTRACTOR_PROMPT + "\n" + ctx,
            foundry_key or openrouter_key,
            "foundry" if foundry_key else "openrouter"
        )

        val = float(res.get("value_in_millions", 0))
        cur = res.get("currency_code", "USD")

        row.update({
            "Supplier Name": res.get("extracted_supplier"),
            "Program Type": res.get("program_type"),
            "Value Certainty": res.get("value_certainty"),
            "G2G/B2G": res.get("g2g_b2g"),
            "Signing Month": res.get("signing_month"),
            "Signing Year": res.get("signing_year"),
            "Value (Million)": round(val, 3),
            "Currency": cur,
            "Value (USD$ Million)": round(val if cur == "USD" else val * 1.1, 3),
            "Value Note\n(If Any)": res.get("value_note")
        })

        return {**state, "rows": [row]}

    def kb_validation(state: AgentState):
        r = state["rows"][0].copy()
        kb = state["kb_meta"]
        notes = {}

        for f in ["Supplier Country", "Market Segment", "System Name (General)"]:
            if kb.get(f) and r.get(f):
                notes[f] = "Match" if kb[f] == r[f] else "Conflict"

        r["KB Validation Notes"] = json.dumps(notes)
        r["Confidence Score"] = 90 if "Conflict" not in notes.values() else 65

        return {**state, "rows": [r]}

    def evaluation(state: AgentState):
        return state

    def formatter(state: AgentState):
        return state

    graph = StateGraph(AgentState)
    graph.add_node("KBRouter", kb_router)
    graph.add_node("Sourcing", sourcing)
    graph.add_node("Geography", geography)
    graph.add_node("System", system)
    graph.add_node("Contract", contract)
    graph.add_node("KBValidation", kb_validation)
    graph.add_node("Evaluation", evaluation)
    graph.add_node("ExcelFormatter", formatter)

    graph.add_edge(START, "KBRouter")
    graph.add_edge("KBRouter", "Sourcing")
    graph.add_edge("Sourcing", "Geography")
    graph.add_edge("Geography", "System")
    graph.add_edge("System", "Contract")
    graph.add_edge("Contract", "KBValidation")
    graph.add_edge("KBValidation", "Evaluation")
    graph.add_edge("Evaluation", "ExcelFormatter")
    graph.add_edge("ExcelFormatter", END)

    return graph.compile()

# ==========================================================
# 7. STREAMLIT UI
# ==========================================================
st.title("🛡️ Defense Agentic Pipeline")

foundry_key = st.sidebar.text_input("LLM Foundry Key", type="password")
openrouter_key = st.sidebar.text_input("OpenRouter Key", type="password")
kb_path = st.sidebar.text_input("KB Path")

uploaded = st.file_uploader("Upload Source Excel", type=["xlsx"])

if uploaded and st.button("🚀 Start Extraction"):
    kb = load_kb_resources(kb_path)
    app = build_workflow(kb, foundry_key, openrouter_key)

    df = pd.read_excel(uploaded)
    all_rows = []

    for _, r in df.iterrows():
        state = {
            "text": str(r["Contract Description"]),
            "date": str(r["Contract Date"]),
            "url": str(r["Source URL"]),
            "kb_meta": {},
            "kb_score": 0.0,
            "row": {},
            "rows": [],
            "messages": []
        }

        for event in app.stream(state):
            for v in event.values():
                if isinstance(v, dict) and "rows" in v:
                    all_rows.extend(v["rows"])

    df_out = pd.DataFrame(all_rows)
    df_out = df_out.reindex(columns=REQUIRED_COLUMNS, fill_value="")

    st.success("Extraction Complete")
    st.dataframe(df_out, use_container_width=True)
