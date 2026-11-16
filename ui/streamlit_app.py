"""Streamlit UI for the OmniSearch AI FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API = os.getenv("OMNISEARCH_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/api"
TIMEOUT = (5, 300)

st.set_page_config(page_title="OmniSearch AI", page_icon="O", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.03em; }
.block-container { padding-top: 2rem; max-width: 1440px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101827,#17243a); }
[data-testid="stSidebar"] * { color: #e8eef8; }
div[data-testid="stMetric"] { background: #f6f8fc; border: 1px solid #e4e9f2; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        timeout = kwargs.pop("timeout", TIMEOUT)
        response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Backend request failed: {detail[:700]}")
        return None


def show_results(results: list[dict[str, Any]], *, images: bool = False) -> None:
    if not results:
        st.info("No results. Index documents or images, then try again.")
        return
    for rank, item in enumerate(results, 1):
        title = item.get("document_name") or item.get("source") or item.get("path") or item.get("document_id") or item.get("image_id") or f"Result {rank}"
        page = f" | page {item['page_number']}" if item.get("page_number") is not None else ""
        score = item.get("score")
        with st.container(border=True):
            st.markdown(f"**{rank}. {title}**{page}")
            if isinstance(score, (int, float)):
                st.caption(f"Score: {score:.4f}")
            if images and item.get("path") and os.path.isfile(item["path"]):
                st.image(item["path"], width=360)
            content = item.get("content") or item.get("ocr_text") or item.get("document_reference")
            if content:
                st.write(content)


def main() -> None:
    st.sidebar.markdown("# OmniSearch AI")
    st.sidebar.caption("Research workspace | local first")
    page = st.sidebar.radio("WORKSPACE", [
        "Dashboard", "Document Management", "AI Search", "Multimodal Search",
        "RAG Assistant", "Research Analytics",
    ], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"API | {API}")

    if page == "Dashboard":
        st.title("Research dashboard")
        st.write("A unified workspace for document, semantic, multimodal, and evidence-grounded search.")
        data = api_request("GET", "/dashboard")
        if data:
            cards = st.columns(4)
            cards[0].metric("Documents", data.get("document_count", 0))
            cards[1].metric("Text chunks", data.get("chunk_count", 0))
            cards[2].metric("Indexed images", data.get("image_count", 0))
            cards[3].metric("Searches recorded", data.get("search_count", 0))
            left, right = st.columns(2)
            with left:
                st.subheader("Search activity")
                counts = data.get("method_counts", {})
                st.bar_chart(counts) if counts else st.caption("Usage appears after your first query.")
                recent = data.get("recent_searches", [])
                if recent:
                    st.subheader("Recent latency (ms)")
                    st.line_chart({"Latency": [row["latency_ms"] for row in recent]})
            with right:
                st.subheader("Configured models")
                for name, model in data.get("models", {}).items():
                    st.markdown(f"**{name.replace('_', ' ').title()}**  \n`{model}`")
                st.caption("Models load when their feature is used. CUDA is checked at runtime.")

    elif page == "Document Management":
        st.title("Document management")
        st.write("Upload PDF, TXT, and DOCX files. Page-aware chunks and document references are retained.")
        files = st.file_uploader("Choose documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        embed = st.checkbox("Generate semantic vectors on upload", value=False, help="Loads Sentence Transformers and Chroma. The initial model load may download weights.")
        if files and st.button("Upload documents", type="primary"):
            progress = st.progress(0)
            for i, uploaded in enumerate(files, 1):
                result = api_request("POST", "/documents", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}, params={"embed": str(embed).lower()}, timeout=(5, 900))
                if result:
                    msg = f"{uploaded.name}: {result.get('chunk_count', 0)} chunks"
                    if result.get("duplicate"):
                        msg += " (duplicate skipped)"
                    st.success(msg)
                    if result.get("embedding_error"):
                        st.warning(f"Keyword indexing succeeded; embedding failed: {result['embedding_error']}")
                progress.progress(i / len(files))
        payload = api_request("GET", "/documents")
        documents = payload.get("documents", []) if payload else []
        st.subheader(f"Indexed documents | {len(documents)}")
        if not documents:
            st.info("Your library is empty. Upload a document to begin.")
        for doc in documents:
            with st.container(border=True):
                cols = st.columns([4, 1, 1, 1])
                cols[0].markdown(f"**{doc['name']}**\n\n`{doc['document_id'][:20]}...` | {doc['page_count']} pages | {doc['chunk_count']} chunks")
                cols[1].write(doc["file_type"].upper())
                cols[2].write("Semantic ready" if doc["dense_indexed"] else "Keyword ready")
                if cols[3].button("Delete", key=f"delete-{doc['document_id']}"):
                    if api_request("DELETE", f"/documents/{doc['document_id']}"):
                        st.rerun()

    elif page == "AI Search":
        st.title("AI search")
        with st.form("search"):
            query = st.text_input("Search query", placeholder="What evidence are you looking for?")
            a, b, c = st.columns([2, 1, 2])
            method_names = {"BM25 keyword": "keyword", "Dense semantic": "semantic", "Hybrid RRF": "hybrid", "Generative retrieval": "generative"}
            label = a.selectbox("Retrieval method", list(method_names))
            top_k = b.slider("Top K", 1, 20, 5)
            document_id = c.text_input("Document filter (optional)")
            submitted = st.form_submit_button("Search", type="primary")
        if submitted and query:
            result = api_request("POST", "/search", json={"query": query, "method": method_names[label], "top_k": top_k, "document_id": document_id or None})
            if result:
                st.caption(f"{len(result['results'])} results | {result['method']}")
                show_results(result["results"])

    elif page == "Multimodal Search":
        st.title("Multimodal search")
        st.caption("CLIP image and text encodings share a joint embedding space. Indexing loads the configured model on first use.")
        upload_tab, text_tab, image_tab = st.tabs(["Index images", "Text to image", "Image to image"])
        with upload_tab:
            image = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"], key="image_upload")
            payload = api_request("GET", "/documents")
            docs = payload.get("documents", []) if payload else []
            options = {"No association": ""}
            options.update({doc["name"]: doc["document_id"] for doc in docs})
            association = st.selectbox("Associate with document (optional)", list(options))
            ocr = st.checkbox("Extract text with EasyOCR", value=False)
            if image and st.button("Index image", type="primary"):
                indexed = api_request("POST", "/images", files={"file": (image.name, image.getvalue(), image.type or "image/png")}, params={"document_id": options[association], "ocr": str(ocr).lower()}, timeout=(5, 900))
                if indexed:
                    st.success(f"Image indexed: {indexed['image_id']}")
        with text_tab:
            with st.form("text_image"):
                query = st.text_input("Describe the image", placeholder="A blue circle on a white background")
                k = st.slider("Images to show", 1, 20, 5, key="image_top_k")
                submit = st.form_submit_button("Search images", type="primary")
            if submit and query:
                result = api_request("GET", "/images/search", params={"query": query, "top_k": k})
                if result:
                    show_results(result["results"], images=True)
        with image_tab:
            query_image = st.file_uploader("Choose query image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="image_query")
            k = st.slider("Similar images", 1, 20, 5, key="similar_top_k")
            if query_image and st.button("Find similar images", type="primary"):
                result = api_request("POST", "/images/search-by-image", files={"file": (query_image.name, query_image.getvalue(), query_image.type or "image/png")}, params={"top_k": k})
                if result:
                    show_results(result["results"], images=True)

    elif page == "RAG Assistant":
        st.title("RAG assistant")
        st.write("Ask questions against your document library. Ollama must be running locally; answers include source and page citations.")
        with st.form("rag"):
            question = st.text_area("Question", placeholder="What does the documentation say about...", height=100)
            top_k = st.slider("Evidence chunks", 1, 12, 5)
            submitted = st.form_submit_button("Ask OmniSearch", type="primary")
        if submitted and question:
            with st.spinner("Retrieving evidence and generating an answer..."):
                response = api_request("POST", "/rag", json={"question": question, "top_k": top_k}, timeout=(5, 300))
            if response:
                st.warning(response["answer"]) if response["insufficient_evidence"] else st.markdown(response["answer"])
                st.caption(f"Model: {response['model']}")
                if response.get("error"):
                    st.error(response["error"])
                if response.get("citations"):
                    st.subheader("Sources")
                    for cite in response["citations"]:
                        st.markdown(f"**[{cite['marker']}] {cite['document_name']}** | page {cite['page_number']}\n\n`{cite['source']}`")

    elif page == "Research Analytics":
        st.title("Research analytics")
        st.caption("Charts use saved evaluation reports and observed application search events. Missing methods do not receive synthesized metrics.")
        data = api_request("GET", "/analytics")
        runs = data.get("runs", []) if data else []
        if runs:
            choice = st.selectbox("Experiment report", range(len(runs)), format_func=lambda i: f"{runs[i].get('dataset', {}).get('name', 'dataset')} | {runs[i].get('run_id', 'run')}")
            report = runs[choice]
            aggregate = report.get("aggregates", {})
            if aggregate:
                import pandas as pd
                frame = pd.DataFrame(aggregate).T
                ranking = [key for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k") if key in frame]
                st.subheader("Ranking quality")
                st.bar_chart(frame[ranking])
                latency = [key for key in ("search_latency_ms", "embedding_generation_ms") if key in frame and frame[key].notna().any()]
                if latency:
                    st.subheader("Measured latency (ms)")
                    st.bar_chart(frame[latency])
                st.dataframe(frame, use_container_width=True)
            ds = report.get("dataset", {})
            st.caption(f"Dataset: {ds.get('name')} | split: {ds.get('split')} | seed: {report.get('seed')}")
        else:
            st.info("No evaluation reports yet. Run a configured experiment and write its report under outputs/ to compare methods.")
        events = data.get("search_events", []) if data else []
        if events:
            import pandas as pd
            st.subheader("Application search latency (ms)")
            st.line_chart(pd.DataFrame(events).set_index("timestamp")["latency_ms"])


if __name__ == "__main__":
    main()
