"""Gradio demo. Talks to the API over HTTP.

   python app/demo.py                      # API at http://127.0.0.1:8000
   PMRAG_API_URL=http://host:8000 python app/demo.py
"""

from __future__ import annotations

import os

import requests

DEFAULT_API = os.environ.get("PMRAG_API_URL", "http://127.0.0.1:8000")
EXAMPLES = [  # questions that exist in the corpus (PubMedQA)
    "Is withdrawal-induced anxiety in alcoholism based on beta-endorphin deficiency?",
    "Does high blood pressure reduce the risk of chronic low back pain?",
    "Can surgeon familiarization with current evidence lead to a change in practice?",
    "Is routine intraoperative frozen-section examination of sentinel lymph nodes in breast cancer worthwhile?",
]


def call_api(api_url: str, question: str, k: int, explain: bool) -> dict:
    r = requests.post(f"{api_url}/ask", json={"question": question, "k": int(k), "explain": bool(explain)}, timeout=300)
    r.raise_for_status()
    return r.json()


def format_result(res: dict) -> tuple[str, dict | None, str, str]:
    """-> (verdict markdown, probabilities, evidence markdown, details markdown)"""
    if res["decision"]:
        flag = "  \n*Low confidence: treat as inconclusive.*" if res["low_confidence"] else ""
        verdict = f"## {res['decision'].upper()}{flag}"
    else:
        verdict = "## No verdict (retrieval-only mode or no relevant abstracts)"
    if res.get("warning"):
        verdict += f"\n\n> {res['warning']}"
    if res.get("explanation"):
        verdict += f"\n\n**Evidence summary:** {res['explanation']}"
    evidence = "\n\n".join(
        f"**{e['rank']}. [PMID {e['doc_id']}](https://pubmed.ncbi.nlm.nih.gov/{e['doc_id']}/)**  (score {e['score']})  \n"
        f"{e['text'][:700]}{'...' if len(e['text']) > 700 else ''}" for e in res["evidence"]) or "_nothing retrieved_"
    t = res["timings_ms"]
    details = (f"{res['disclaimer']}  \nSearches **{res['corpus_size']:,}** PubMed abstracts only (not all of PubMed). "
               f"Backend `{res['backend']}`, retrieval `{res['retrieval_mode']}`, "
               f"{'calibrated' if res['calibrated'] else 'uncalibrated'}. "
               f"Latency {t['total']:.0f} ms (retrieval {t['retrieval']:.0f}, scoring {t['scoring']:.0f}).")
    return verdict, res["probabilities"], evidence, details


def build_demo(api_url: str = DEFAULT_API):
    import gradio as gr

    def ask_ui(question: str, k: int, explain: bool):
        if not question or len(question.strip()) < 3:
            return "Please enter a research question.", None, "", "", ""
        try:
            res = call_api(api_url, question.strip(), k, explain)
        except requests.RequestException as exc:
            return f"Could not reach the API at {api_url}: {exc}", None, "", "", ""
        verdict, probs, evidence, details = format_result(res)
        return verdict, probs, evidence, details, res["request_id"]

    def send_feedback(request_id: str, helpful: bool):
        if not request_id:
            return "Ask a question first."
        try:
            requests.post(f"{api_url}/feedback", json={"request_id": request_id, "helpful": helpful}, timeout=30).raise_for_status()
        except requests.RequestException:
            return "Could not record feedback."
        return "Thanks, feedback recorded."

    with gr.Blocks(title="PubMed Evidence Assistant") as demo:
        gr.Markdown("# PubMed Evidence Assistant\nAsk a biomedical **research** question. "
                    "The answer is a yes/no/maybe grounded in retrieved PubMed abstracts. "
                    "**Not medical advice.**")
        question = gr.Textbox(label="Research question", lines=2, placeholder="Does ... improve ...?")
        with gr.Row():
            k = gr.Slider(1, 8, value=5, step=1, label="Abstracts to retrieve")
            explain = gr.Checkbox(label="Also generate a short evidence summary (slower)", value=False)
        run = gr.Button("Ask", variant="primary")
        gr.Examples(examples=[[q] for q in EXAMPLES], inputs=[question])
        verdict = gr.Markdown()
        probs = gr.Label(label="Calibrated probabilities", num_top_classes=3)
        with gr.Row():
            up, down = gr.Button("Helpful"), gr.Button("Not helpful")
        fb = gr.Markdown()
        details = gr.Markdown()
        evidence = gr.Markdown()
        request_id = gr.State("")
        run.click(ask_ui, [question, k, explain], [verdict, probs, evidence, details, request_id])
        up.click(lambda rid: send_feedback(rid, True), [request_id], [fb])
        down.click(lambda rid: send_feedback(rid, False), [request_id], [fb])
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
