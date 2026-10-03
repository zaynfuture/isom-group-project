"""FairnessLens Streamlit entry point; business logic lives in the package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from isom_project.audit import build_audit_report, score_audit_frame, validate_audit_frame
from isom_project.config import METRICS_PATH, MODEL_DIR
from isom_project.data import generate_dataset, sample_audit_data
from isom_project.modeling import load_pipeline, model_is_available, predict_probabilities

MODEL_SOURCE = os.environ.get("HF_MODEL_ID", str(MODEL_DIR))
PAGES = ["Home", "Data Explorer", "Single Prediction", "Batch Audit", "Model Evidence", "Responsible Use"]

st.set_page_config(page_title="FairnessLens | Demographic Imputation Audit", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading the fine-tuned language model…")
def get_classifier():
    return load_pipeline(MODEL_SOURCE)


def predict(texts: list[str]):
    return predict_probabilities(get_classifier(), texts)


@st.cache_data(show_spinner=False)
def get_complete_dataset() -> pd.DataFrame:
    """Build the deterministic train/validation/test dataset for exploration."""
    return generate_dataset(6000)


def header(title: str, description: str):
    st.title(title)
    st.caption(description)


using_hub = bool(os.environ.get("HF_MODEL_ID"))
model_available = using_hub or model_is_available(MODEL_DIR)

with st.sidebar:
    st.markdown("## ⚖️ FairnessLens")
    st.caption("Demographic imputation validity for aggregate fairness audits")
    page = st.radio("Workflow", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Model status**")
    if model_available:
        st.success("Ready for inference")
        st.caption("Hugging Face Hub model" if using_hub else "Bundled fine-tuned model")
    else:
        st.error("Model artifact unavailable")
    st.divider()
    st.caption("Educational prototype · Uploaded data is not persisted")

if page == "Home":
    header("FairnessLens", "Assess whether imputed demographic probabilities preserve an aggregate fairness conclusion.")
    st.warning("FairnessLens estimates audit-level uncertainty. It must not assign identity or make decisions about an individual.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fine-tuned model", "2-layer BERT")
    c2.metric("Primary method", "Soft imputation")
    c3.metric("Stored uploads", "None")
    st.markdown("### What this application answers")
    for column, title, body in zip(st.columns(3), ["1 · Estimate", "2 · Compare", "3 · Decide"], [
        "Produce a probability for Group B instead of treating a proxy as a known identity.",
        "Compare observed and imputed fairness metrics when benchmark labels exist.",
        "Assess whether imputation preserves the direction and magnitude of an audit finding.",
    ]):
        with column:
            st.markdown(f"#### {title}")
            st.write(body)
    st.markdown("### Recommended workflow")
    st.info("Acknowledge use boundary → validate data → run soft imputation → review fairness error → export evidence.")
    accepted = st.checkbox(
        "I will use outputs only for aggregate educational fairness analysis, not to identify or make decisions about a person.",
        value=st.session_state.get("intended_use_accepted", False),
    )
    st.session_state["intended_use_accepted"] = accepted
    if accepted:
        st.success("Use boundary acknowledged. Continue to Data Explorer, Single Prediction, or Batch Audit.")

elif page == "Data Explorer":
    header("Data Explorer", "Review the complete synthetic training, validation, and testing dataset.")
    dataset = get_complete_dataset()
    split_counts = dataset["split"].value_counts()
    columns = st.columns(4)
    columns[0].metric("All records", f"{len(dataset):,}")
    columns[1].metric("Training", f"{split_counts.get('train', 0):,}")
    columns[2].metric("Validation", f"{split_counts.get('validation', 0):,}")
    columns[3].metric("Testing", f"{split_counts.get('test', 0):,}")
    selected_split = st.selectbox("Dataset split", ["All", "Training", "Validation", "Testing"])
    split_map = {"Training": "train", "Validation": "validation", "Testing": "test"}
    displayed = dataset if selected_split == "All" else dataset.loc[dataset["split"] == split_map[selected_split]]
    scope_label = "complete dataset" if selected_split == "All" else f"{selected_split.lower()} split"
    st.caption(
        f"Showing all {len(displayed):,} records from the {scope_label}. Scroll vertically or horizontally inside the table."
    )
    st.dataframe(displayed, use_container_width=True, hide_index=True, height=560)
    st.download_button(
        f"Download {selected_split.lower()} data (CSV)",
        displayed.to_csv(index=False).encode("utf-8"),
        f"fairnesslens_{selected_split.lower()}_data.csv",
        "text/csv",
    )
    st.info("The testing split is held out from model training and checkpoint selection.")

elif page == "Single Prediction":
    header("Single Prediction", "Inspect the model interface with one synthetic record before a batch audit.")
    st.info("This probability demonstrates aggregation; it is not a statement of personal identity.")
    text = st.text_area("Synthetic profile text", "profile token marlow; region north; channel web; product credit; income band 3; tenure 5 years", height=120)
    allowed = st.session_state.get("intended_use_accepted", False)
    if not allowed:
        st.warning("Acknowledge the intended-use boundary on Home before running inference.")
    if st.button("Estimate probability", type="primary", disabled=not (allowed and model_available)):
        if not text.strip():
            st.error("Enter a non-empty synthetic profile.")
        else:
            try:
                probability = float(predict([text.strip()])[0])
                left, right = st.columns(2)
                left.metric("P(Group A)", f"{1 - probability:.1%}")
                right.metric("P(Group B)", f"{probability:.1%}")
                st.progress(probability, text="Probability mass assigned to Group B")
                st.caption("The 0.50 threshold is for inspection only; the audit uses the full probability.")
            except Exception as exc:
                st.error(f"Inference could not be completed: {exc}")

elif page == "Batch Audit":
    header("Batch Fairness Audit", "Validate a decision dataset, run batched inference, and compare fairness estimates.")
    allowed = st.session_state.get("intended_use_accepted", False)
    if not allowed:
        st.warning("Acknowledge the intended-use boundary on Home before running an audit.")
    with st.expander("CSV data contract"):
        st.markdown("- `text`: non-empty model input\n- `qualified`: benchmark outcome, 0/1\n- `decision`: system decision, 0/1\n- `group_b`: optional verified benchmark group, 0/1")
    source = st.radio("Data source", ["Built-in synthetic sample", "Upload CSV"], horizontal=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], disabled=source != "Upload CSV")
    frame = sample_audit_data(600) if source == "Built-in synthetic sample" else None
    if source == "Upload CSV" and uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The CSV could not be read: {exc}")
    valid = False
    if frame is not None:
        a, b, c = st.columns(3)
        a.metric("Records", f"{len(frame):,}")
        b.metric("Columns", len(frame.columns))
        c.metric("Benchmark labels", "Available" if "group_b" in frame else "Not supplied")
        st.caption(f"Showing all {len(frame):,} records. Scroll inside the table to inspect the complete batch.")
        st.dataframe(frame, use_container_width=True, hide_index=True, height=480)
        try:
            _, preview_warnings = validate_audit_frame(frame)
            valid = True
            st.success("Schema and binary-field validation passed.")
            for warning in preview_warnings:
                st.warning(warning)
        except ValueError as exc:
            st.error(str(exc))
        if st.button("Run audit", type="primary", disabled=not (allowed and model_available and valid)):
            try:
                with st.spinner("Scoring records and calculating fairness metrics…"):
                    scored, comparison, warnings = score_audit_frame(frame, predict)
                    report = build_audit_report(scored, comparison, MODEL_SOURCE)
                st.session_state["audit_result"] = (scored, comparison, report, warnings)
            except Exception as exc:
                st.error(f"The audit could not be completed: {exc}")
    if "audit_result" in st.session_state:
        scored, comparison, report, warnings = st.session_state["audit_result"]
        st.divider()
        st.markdown("### Audit results")
        for warning in warnings:
            st.warning(warning)
        st.caption("All gaps use Group B minus Group A. The primary estimate uses probability-weighted soft membership.")
        st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
        rows = [name for name in ["Observed group", "Imputed (soft)"] if name in comparison.index]
        st.bar_chart(comparison.loc[rows, ["demographic_parity_gap", "equal_opportunity_gap", "false_positive_rate_gap"]].T)
        if "Absolute error" in comparison.index:
            error = comparison.loc["Absolute error", "demographic_parity_gap"]
            observed = comparison.loc["Observed group", "demographic_parity_gap"]
            imputed = comparison.loc["Imputed (soft)", "demographic_parity_gap"]
            st.info(f"Observed DP gap: {observed:.3f}. Soft-imputed gap: {imputed:.3f}. Absolute error: {error:.3f}.")
        else:
            st.warning("Without verified group labels, this run cannot establish fairness-estimation accuracy.")
        d1, d2 = st.columns(2)
        d1.download_button("Download scored records (CSV)", scored.to_csv(index=False).encode(), "fairnesslens_scored_records.csv", "text/csv", use_container_width=True)
        d2.download_button("Download audit summary (JSON)", json.dumps(report, indent=2, allow_nan=True).encode(), "fairnesslens_audit_summary.json", "application/json", use_container_width=True)

elif page == "Model Evidence":
    header("Model Evidence", "Separate demographic classification quality from downstream audit validity.")
    if not METRICS_PATH.exists():
        st.warning("Evaluation evidence is unavailable. Run `python scripts/evaluate.py` after training.")
    else:
        metrics = json.loads(METRICS_PATH.read_text())
        classification = metrics["classification"]
        columns = st.columns(4)
        for column, key, label, help_text in zip(columns, ["macro_f1", "roc_auc", "brier_score", "balanced_accuracy"], ["Macro-F1", "ROC AUC", "Brier score", "Balanced accuracy"], ["Class-balanced performance.", "Ranking discrimination.", "Probability error; lower is better.", "Average recall across groups."]):
            column.metric(label, f"{classification[key]:.3f}", help=help_text)
        st.markdown("### Downstream fairness-estimation validity")
        st.dataframe(pd.DataFrame(metrics["fairness"]).T.style.format("{:.3f}"), use_container_width=True)
        st.warning("Strong classification does not guarantee a valid audit. Observed-versus-imputed error is the decision-relevant evidence.")
        with st.expander("Raw reproducibility evidence"):
            st.json(metrics)

else:
    header("Responsible Use", "Governance requirements for demographic-imputation research and aggregate auditing.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Intended use\n- Aggregate educational fairness screening\n- Comparing observed and imputed metrics\n- Studying proxy measurement error\n- Demonstrating a reproducible workflow")
        st.markdown("### Required controls\n- Lawful purpose and access\n- Representative benchmark validation\n- Calibration and subgroup error analysis\n- Human review and documented limitations\n- Minimal retention")
    with right:
        st.markdown("### Prohibited use\n- Assigning identity to a person\n- Eligibility, hiring, lending, pricing, or enforcement\n- Targeting, surveillance, or sensitive profiling\n- Treating a proxy as self-described data\n- Claiming legal compliance")
        st.markdown("### Demonstration boundary")
        st.write("The bundled dataset is synthetic and uses neutral Group A/Group B labels. Results do not validate any real demographic group.")
