from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from isom_project.config import METRICS_PATH, MODEL_DIR
from isom_project.data import sample_audit_data
from isom_project.metrics import comparison_table
from isom_project.modeling import load_pipeline, model_is_available, predict_probabilities


st.set_page_config(
    page_title="FairnessLens | Demographic Imputation Audit",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading fine-tuned Hugging Face model...")
def get_classifier():
    return load_pipeline(MODEL_DIR)


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "qualified", "decision"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["qualified"] = pd.to_numeric(result.qualified, errors="raise").astype(int)
    result["decision"] = pd.to_numeric(result.decision, errors="raise").astype(int)
    if not result.qualified.isin([0, 1]).all() or not result.decision.isin([0, 1]).all():
        raise ValueError("qualified and decision must contain only 0 or 1")
    result["group_b_probability"] = predict_probabilities(
        get_classifier(), result.text.fillna("").astype(str).tolist()
    )
    result["imputed_group"] = result.group_b_probability.map(
        lambda p: "Group B" if p >= 0.5 else "Group A"
    )
    return result


st.title("FairnessLens")
st.caption("Validate demographic imputation before using it in a fairness audit")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Single prediction", "Batch fairness audit", "Model evaluation", "Responsible use"])
    available = model_is_available(MODEL_DIR)
    st.divider()
    if available:
        st.success("Fine-tuned model available")
    else:
        st.error("Fine-tuned model not found")
        st.code("python scripts/train.py\npython scripts/evaluate.py", language="bash")

if page == "Overview":
    st.subheader("Business problem")
    st.write(
        "Organizations often need demographic information to test whether an AI system treats "
        "groups differently, but those attributes may be missing or restricted. FairnessLens "
        "tests whether a proxy-based demographic model preserves the fairness conclusion—not "
        "only whether it predicts labels accurately."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Fine-tuned 2-layer BERT")
    c2.metric("Evaluation levels", "2", help="Classification and downstream fairness")
    c3.metric("Sensitive real data", "None", help="The classroom demo uses synthetic groups")
    st.info(
        "The supplied dataset is synthetic and deliberately uses neutral Group A/Group B labels. "
        "It demonstrates the workflow but does not validate demographic inference in production."
    )

elif page == "Single prediction":
    st.subheader("Inspect one model prediction")
    text = st.text_area(
        "Synthetic customer profile text",
        "profile token marlow; region north; channel web; product credit; income band 3; tenure 5 years",
    )
    if st.button("Predict", type="primary", disabled=not available):
        probability = float(predict_probabilities(get_classifier(), [text])[0])
        st.metric("P(Group B)", f"{probability:.1%}")
        st.progress(probability)
        st.caption("This is a probabilistic proxy estimate, not a statement of personal identity.")

elif page == "Batch fairness audit":
    st.subheader("Audit a batch of decisions")
    st.write("Upload a CSV or use the built-in synthetic sample.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    frame = pd.read_csv(uploaded) if uploaded else sample_audit_data(250)
    st.caption("Required: text, qualified, decision. Optional benchmark: group_b.")
    st.dataframe(frame.head(10), use_container_width=True)
    if st.button("Run fairness audit", type="primary", disabled=not available):
        try:
            with st.spinner("Running batched inference and fairness metrics..."):
                scored = score_frame(frame)
                comparison = comparison_table(scored)
            st.subheader("Fairness comparison")
            st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
            chart = comparison.loc[
                [i for i in ["Observed group", "Imputed (soft)"] if i in comparison.index],
                ["group_a_selection_rate", "group_b_selection_rate"],
            ]
            st.bar_chart(chart)
            st.download_button(
                "Download scored records",
                scored.to_csv(index=False).encode("utf-8"),
                "fairness_audit_predictions.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

elif page == "Model evaluation":
    st.subheader("Held-out evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        classification = metrics["classification"]
        columns = st.columns(4)
        for column, key, label in zip(
            columns,
            ["macro_f1", "roc_auc", "brier_score", "balanced_accuracy"],
            ["Macro-F1", "ROC AUC", "Brier score", "Balanced accuracy"],
        ):
            column.metric(label, f"{classification[key]:.3f}")
        st.markdown("#### Downstream fairness")
        st.dataframe(pd.DataFrame(metrics["fairness"]).T.style.format("{:.3f}"), use_container_width=True)
        st.json(metrics, expanded=False)
    else:
        st.warning("Run `python scripts/evaluate.py` after training to populate this page.")

else:
    st.subheader("Responsible-use boundary")
    st.markdown(
        """
        - Use imputed attributes only for aggregate auditing, never for individual decisions.
        - Prefer lawfully collected, voluntary, self-described demographics when available.
        - Validate calibration, group-specific errors, and downstream fairness error on representative data.
        - Treat strong classification scores as insufficient evidence of audit validity.
        - Document purpose, access, retention, distribution shift, and human review.
        - Do not interpret synthetic Group A/Group B as real race, gender, ethnicity, or identity.
        """
    )
    st.warning("This classroom prototype is not approved for production or legal compliance decisions.")
