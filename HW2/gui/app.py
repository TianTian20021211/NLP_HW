from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gui_inference import DocumentPrediction, load_winner_predictor
from src.paths import TRANSCRIPTS_DIR


st.set_page_config(page_title="Transcript Inline Tagger", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .doc-view {
        border: 1px solid #d7dbe2;
        border-radius: 8px;
        max-height: 76vh;
        overflow-y: auto;
        padding: 0.75rem;
        background: #ffffff;
    }
    .sentence {
        margin: 0 0 0.38rem 0;
        padding: 0.42rem 0.55rem;
        border-left: 3px solid transparent;
        line-height: 1.48;
        font-size: 0.95rem;
    }
    .sentence.boilerplate {
        background: #fde2e2;
        border-left-color: #c92a2a;
    }
    .sentence.substantive {
        background: transparent;
    }
    .sentence-index {
        color: #667085;
        display: inline-block;
        font-size: 0.78rem;
        font-variant-numeric: tabular-nums;
        margin-right: 0.45rem;
        min-width: 2.4rem;
    }
    .prob {
        color: #667085;
        font-size: 0.78rem;
        margin-left: 0.45rem;
        white-space: nowrap;
    }
    .small-note {
        color: #667085;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved winner model...")
def _predictor():
    return load_winner_predictor()


@st.cache_data(show_spinner=False)
def _sample_files() -> list[str]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    return [str(path.relative_to(ROOT)) for path in sorted(TRANSCRIPTS_DIR.glob("*.txt"))]


def _read_selected_input(uploaded_file, pasted_text: str, sample_choice: str) -> tuple[str, str]:
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        try:
            return raw.decode("utf-8"), uploaded_file.name
        except UnicodeDecodeError as exc:
            raise ValueError("Uploaded file must be UTF-8 text.") from exc
    if pasted_text.strip():
        return pasted_text, "pasted_text.txt"
    if sample_choice:
        path = ROOT / sample_choice
        return path.read_text(encoding="utf-8"), path.name
    return "", ""


def _percent(part: int, total: int) -> float:
    return (100.0 * part / total) if total else 0.0


def _render_document(prediction: DocumentPrediction) -> None:
    chunks: list[str] = []
    for idx, sent in enumerate(prediction.sentences, start=1):
        css_class = "boilerplate" if sent.label == "Boilerplate" else "substantive"
        title = f"{sent.label}; P(substantive)={sent.p_substantive:.3f}"
        chunks.append(
            '<div class="sentence {css}" title="{title}">'
            '<span class="sentence-index">{idx:04d}</span>'
            "{text}"
            '<span class="prob">{prob:.3f}</span>'
            "</div>".format(
                css=css_class,
                title=html.escape(title, quote=True),
                idx=idx,
                text=html.escape(sent.text),
                prob=sent.p_substantive,
            )
        )
    st.markdown('<div class="doc-view">' + "\n".join(chunks) + "</div>", unsafe_allow_html=True)


st.title("Transcript Inline Tagger")

left, right = st.columns([0.28, 0.72], gap="large")
samples = _sample_files()
default_sample = "data/transcripts/C_Q4-2024.txt"
default_index = samples.index(default_sample) + 1 if default_sample in samples else 0

with left:
    uploaded = st.file_uploader("Transcript file", type=["txt"])
    sample = st.selectbox("Sample transcript", [""] + samples, index=default_index)
    pasted = st.text_area("Paste transcript", height=180)
    run = st.button("Tag transcript", type="primary", use_container_width=True)

prediction: DocumentPrediction | None = None
error_message = ""

if run:
    try:
        text, source_name = _read_selected_input(uploaded, pasted, sample)
        if not text.strip():
            error_message = "Choose a transcript file, paste text, or select a sample transcript."
        elif len(text.encode("utf-8")) > 10 * 1024 * 1024:
            error_message = "Input is over 10 MB. Use a smaller plain-text transcript."
        else:
            prediction = _predictor().predict_document(text, source_name)
    except Exception as exc:
        error_message = str(exc)

with left:
    if error_message:
        st.error(error_message)
    if prediction is not None:
        total = prediction.classified_count
        st.subheader("Statistics")
        st.metric(
            "Boilerplate",
            f"{prediction.boilerplate_count} ({_percent(prediction.boilerplate_count, total):.1f}%)",
        )
        st.metric(
            "Substantive",
            f"{prediction.substantive_count} ({_percent(prediction.substantive_count, total):.1f}%)",
        )
        st.metric("Classified sentences", f"{total}")
        meta = prediction.extraction_meta
        st.markdown(
            '<div class="small-note">'
            f"Model: {html.escape(prediction.family_id)}<br>"
            f"Threshold: {prediction.threshold:.3f}<br>"
            f"Short sentences dropped: {meta.get('short_sentence_count', 0)}<br>"
            f"Duplicate lines removed: {meta.get('duplicate_line_count', 0)}"
            "</div>",
            unsafe_allow_html=True,
        )

with right:
    if prediction is None:
        st.info("Run tagging to display the transcript.")
    elif prediction.classified_count == 0:
        st.warning("No sentences met the extraction threshold.")
    else:
        st.caption(prediction.source_name)
        _render_document(prediction)
