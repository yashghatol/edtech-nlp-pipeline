# app/app.py
"""
Privacy-Aware Essay Scorer — Streamlit application.

Session state stage machine:
  'input'  -> user pastes essay, clicks Detect PII
  'review' -> user sees flagged tokens highlighted in context, and chooses
              either to auto-redact + score, or edit the essay manually
  'edit'   -> user edits the raw essay text, then chooses to re-run PII
              detection or score the edited essay directly
  'score'  -> essay scored 1-6, with gauge, heuristic confidence,
              essay statistics, and LOO sentence-importance cards
"""

import os
import sys
import html as html_lib

# Make src/ importable when Streamlit runs from app/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import streamlit as st
import torch
import textstat

from src.utils.config import load_config
from src.pipeline.pii_inference import (
    apply_redactions,
    load_pii_model,
    run_pii_inference,
)
from src.pipeline.essay_inference import (
    compute_loo_importance,
    load_essay_model,
    nlp as essay_nlp,    # reuse the spaCy instance for word/sentence stats
)

# ── Config + device ────────────────────────────────────────────────────────────
cfg    = load_config("configs/config.yaml")
device = torch.device("cpu")


# ── Cached model loaders ───────────────────────────────────────────────────────
@st.cache_resource
def get_pii_model():
    """Load PII model from HF Hub once and cache for the lifetime of the app session."""
    return load_pii_model("yashghatol/edtech-pii-model", cfg, device)


@st.cache_resource
def get_essay_model():
    """Load essay scorer (fold 1) from HF Hub once and cache for the lifetime of the app session."""
    return load_essay_model("yashghatol/edtech-essay-model", device)


# ── Helper: render essay text with PII spans highlighted ───────────────────────
def render_highlighted_text(text: str, preds: list, flagged_indices: list) -> str:
    """Return HTML with flagged PII spans wrapped in <mark> tags.

    Spans are processed left-to-right; non-flagged text is HTML-escaped.
    """
    spans = sorted(
        ((preds[i]['start'], preds[i]['end'], preds[i]['label']) for i in flagged_indices),
        key=lambda s: s[0]
    )
    parts, last = [], 0
    for start, end, label in spans:
        parts.append(html_lib.escape(text[last:start]))
        entity = label.replace('B-', '').replace('I-', '')
        parts.append(
            '<mark style="background:#ffc107; color:#1a1a1a; padding:1px 4px; '
            f'border-radius:3px; font-weight:600;" title="{html_lib.escape(entity)}">'
            f'{html_lib.escape(text[start:end])}</mark>'
        )
        last = end
    parts.append(html_lib.escape(text[last:]))
    return "".join(parts).replace("\n", "<br>")


# ── Helper: circular score gauge (HTML/CSS, no extra deps) ─────────────────────
def render_score_gauge(score: int, max_score: int = 6) -> str:
    pct     = score / max_score
    degrees = pct * 360
    if score >= 5:
        ring_colour = "#28a745"   # green
    elif score >= 3:
        ring_colour = "#ffc107"   # amber
    else:
        ring_colour = "#dc3545"   # red

    stars = "★" * score + "☆" * (max_score - score)

    return f"""
    <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
      <div style="width:120px; height:120px; border-radius:50%;
                   background: conic-gradient({ring_colour} {degrees}deg, rgba(150,150,150,0.25) {degrees}deg 360deg);
                   display:flex; align-items:center; justify-content:center;
                   flex-shrink:0;">
        <div style="width:88px; height:88px; border-radius:50%;
                     background: rgba(0,0,0,0.55);
                     display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <span style="font-size:1.9rem; font-weight:700; color:#ffffff; line-height:1;">{score}</span>
          <span style="font-size:0.75rem; color:#e0e0e0;">out of {max_score}</span>
        </div>
      </div>
      <div>
        <div style="font-size:1.6rem; letter-spacing:2px; color:{ring_colour};">{stars}</div>
        <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">Predicted Score: {score} / {max_score}</div>
      </div>
    </div>
    """


# ── Helper: sentence importance card with high-contrast styling ────────────────
def render_sentence_card(sentence: str, importance: float) -> str:
    if importance > 0.005:
        accent = "#28a745"
        bg     = "rgba(40, 167, 69, 0.15)"
        arrow  = "▲"
        label  = "Helps score"
    elif importance < -0.005:
        accent = "#dc3545"
        bg     = "rgba(220, 53, 69, 0.15)"
        arrow  = "▼"
        label  = "Hurts score"
    else:
        accent = "#6c757d"
        bg     = "rgba(108, 117, 125, 0.15)"
        arrow  = "■"
        label  = "Neutral"

    return f"""
    <div style="background:{bg}; border-left:4px solid {accent};
                 padding:10px 14px; border-radius:6px; margin:6px 0;">
      <div style="font-weight:700; color:{accent}; margin-bottom:4px; font-size:0.9rem;">
        {arrow} {importance:+.3f} &nbsp;—&nbsp; {label}
      </div>
      <div style="font-size:0.97rem;">{html_lib.escape(sentence)}</div>
    </div>
    """


# ── Helper: essay statistics panel ──────────────────────────────────────────────
def render_essay_stats(text: str, n_pii_redacted, pii_skipped: bool):
    doc       = essay_nlp(text)
    n_words   = sum(1 for t in doc if not t.is_space and not t.is_punct)
    n_sents   = sum(1 for _ in doc.sents)

    try:
        grade = textstat.flesch_kincaid_grade(text)
        grade_display = f"Grade {grade:.1f}"
    except Exception:
        grade_display = "N/A"

    pii_display = "skipped" if pii_skipped else str(n_pii_redacted)

    st.subheader("Essay Statistics")
    
    # Row 1
    cols_row_1 = st.columns(4)
    cols_row_1[0].metric("Words", n_words)
    cols_row_1[1].metric("Sentences", n_sents)
    cols_row_1[2].metric("PII Entities Removed", pii_display)
    cols_row_1[3].metric("Readability", grade_display)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy-Aware Essay Scorer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Privacy-Aware Essay Scorer")
st.markdown(
    "**✓** Detect personal information &nbsp;&nbsp; "
    "**✓** Redact sensitive data &nbsp;&nbsp; "
    "**✓** Score essays on a 1–6 rubric &nbsp;&nbsp; "
    "**✓** Explain model decisions sentence-by-sentence"
)
st.divider()

# ── Session state initialisation ───────────────────────────────────────────────
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Essay input
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == 'input':
    essay = st.text_area(
        "Paste your essay here:",
        height=300,
        placeholder="Start typing or paste your essay..."
    )
    if st.button("🔍 Detect PII", type="primary") and essay.strip():
        with st.spinner("Detecting PII — this may take a moment on first run..."):
            pii_model, pii_tok = get_pii_model()
            preds = run_pii_inference(essay, pii_model, pii_tok, cfg, device)
        st.session_state.essay = essay
        st.session_state.preds = preds
        st.session_state.stage = 'review'
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — PII review: show what gets redacted, choose auto-redact or edit
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == 'review':
    essay   = st.session_state.essay
    preds   = st.session_state.preds
    flagged = [i for i, p in enumerate(preds) if p['label'] != 'O']

    st.subheader("Step 2 — Review Detected PII")

    if not flagged:
        st.success("No PII detected.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Continue to Scoring", type="primary"):
                st.session_state.redacted_text = essay
                st.session_state.n_redacted    = 0
                st.session_state.pii_skipped   = False
                st.session_state.stage         = 'score'
                st.rerun()
        with col2:
            if st.button("✏️ Edit Essay Manually"):
                st.session_state.stage = 'edit'
                st.rerun()
        st.stop()

    st.write(f"Found **{len(flagged)}** PII token(s). Highlighted below — uncheck any you don't want redacted:")

    st.markdown(
        f"<div style='border:1px solid rgba(150,150,150,0.3); border-radius:6px; "
        f"padding:12px; margin-bottom:12px; line-height:1.6;'>"
        f"{render_highlighted_text(essay, preds, flagged)}</div>",
        unsafe_allow_html=True
    )

    redact_choices = {}
    for i in flagged:
        p     = preds[i]
        label = p['label'].replace('B-', '').replace('I-', '')
        redact_choices[i] = st.checkbox(
            f"**{p['token']}** — {label}",
            value=True,
            key=f"redact_{i}"
        )

    n_selected = sum(redact_choices.values())
    redact_pct = n_selected / max(len(flagged), 1)
    st.caption(f"Redacting {n_selected} / {len(flagged)} flagged tokens ({redact_pct:.0%})")

    st.markdown("**What would you like to do?**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back"):
            st.session_state.stage = 'input'
            st.rerun()
    with col2:
        if st.button("✏️ Edit Essay Manually"):
            st.session_state.stage = 'edit'
            st.rerun()
    with col3:
        if st.button("🔒 Redact Automatically & Score →", type="primary"):
            chosen   = {i for i, v in redact_choices.items() if v}
            redacted = apply_redactions(essay, preds, chosen)
            st.session_state.redacted_text = redacted
            st.session_state.n_redacted    = len(chosen)
            st.session_state.pii_skipped   = False
            st.session_state.stage         = 'score'
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2b — Manual edit
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == 'edit':
    st.subheader("Step 2 — Edit Your Essay")
    st.caption("Make any changes you'd like — remove or rewrite flagged personal information yourself.")

    edited = st.text_area(
        "Edit essay text:",
        value=st.session_state.essay,
        height=300,
        key="edited_essay"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Review"):
            st.session_state.stage = 'review'
            st.rerun()
    with col2:
        if st.button("🔍 Re-run PII Detection"):
            with st.spinner("Re-running PII detection..."):
                pii_model, pii_tok = get_pii_model()
                preds = run_pii_inference(edited, pii_model, pii_tok, cfg, device)
            st.session_state.essay = edited
            st.session_state.preds = preds
            st.session_state.stage = 'review'
            st.rerun()
    with col3:
        if st.button("📊 Score This Essay →", type="primary"):
            st.session_state.essay         = edited
            st.session_state.redacted_text = edited
            st.session_state.n_redacted    = 0
            st.session_state.pii_skipped   = True
            st.session_state.stage         = 'score'
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Essay scoring + gauge + confidence + stats + LOO heatmap
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == 'score':
    redacted = st.session_state.redacted_text

    st.subheader("Step 3 — Essay Score")

    with st.spinner("Scoring essay and computing sentence importance..."):
        essay_model, essay_tok = get_essay_model()
        base_score, importances = compute_loo_importance(
            redacted,
            essay_model,
            essay_tok,
            device,
            max_length=cfg['stage2']['max_length']
        )

    final_score = int(np.clip(np.round(base_score), 1, 6))

    # --- Score gauge + stars ---
    st.markdown(render_score_gauge(final_score), unsafe_allow_html=True)
    st.caption(f"Raw model output: {base_score:.3f}")

    # --- Heuristic confidence + possible range ---
    frac_dist  = abs(base_score - round(base_score))     # 0 (confident) .. 0.5 (uncertain)
    confidence = max(0.0, 1 - 2 * frac_dist)
    if base_score >= final_score:
        range_low, range_high = final_score, min(final_score + 1, 6)
    else:
        range_low, range_high = max(final_score - 1, 1), final_score

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{confidence:.0%}")
    col2.metric("Possible Range", f"{range_low}–{range_high}")
    st.caption(
        "ℹ️ Confidence and range are a heuristic derived from how close the raw "
        "model output is to the nearest integer score — not a calibrated probability."
    )

    st.divider()

    # --- Essay statistics panel ---
    render_essay_stats(
        redacted,
        st.session_state.get('n_redacted', 0),
        st.session_state.get('pii_skipped', False)
    )

    st.divider()

    # --- Sentence importance cards ---
    st.subheader("Sentence Importance (Leave-One-Out)")
    st.caption(
        "▲ Green = sentence raises the score  |  "
        "▼ Red = sentence lowers the score  |  "
        "■ Grey = neutral"
    )
    for row in importances:
        st.markdown(render_sentence_card(row['sentence'], row['importance']), unsafe_allow_html=True)

    st.divider()
    st.caption(
        "⚠️ **Known limitation:** The essay scorer was trained on unredacted essays. "
        "Scoring a redacted or manually-edited essay introduces a minor distribution shift — "
        "treat scores as approximate."
    )

    if st.button("📝 Score another essay"):
        for key in ('essay', 'preds', 'redacted_text', 'n_redacted', 'pii_skipped'):
            st.session_state.pop(key, None)
        st.session_state.stage = 'input'
        st.rerun()