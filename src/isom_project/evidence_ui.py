"""English presentation of saved training evidence and business objectives."""
import json
import pandas as pd
import streamlit as st
from .config import MODEL_DIR, METRICS_PATH


def business_objectives():
    st.subheader("Business case: Meridian Financial Services")
    st.caption("Fictional company scenario; synthetic customer and decision data.")
    st.write("The Responsible AI team reviews an AI-assisted credit approval system when verified demographic labels are missing. Its objective is to measure how much demographic imputation changes an aggregate fairness finding before relying on that finding.")
    st.write("Primary users: model-risk analysts and data scientists. Output: an observed-versus-imputed comparison and downloadable audit evidence for human review.")
    if METRICS_PATH.exists():
        evidence = json.loads(METRICS_PATH.read_text())
        c = evidence["classification"]
        rows = []
        for label, target, value, passed in [
            ("Macro-F1", ">= 0.85", c["macro_f1"], c["macro_f1"] >= .85),
            ("ROC AUC", ">= 0.90", c["roc_auc"], c["roc_auc"] >= .90),
            ("Brier score", "<= 0.10", c["brier_score"], c["brier_score"] <= .10),
            ("Absolute DP-gap error", "<= 0.10", evidence["fairness"]["Absolute error"]["demographic_parity_gap"], evidence["fairness"]["Absolute error"]["demographic_parity_gap"] <= .10),
        ]:
            rows.append({"Measure": label, "Project target": target, "Held-out result": round(value, 4), "Status": "Met" if passed else "Not met"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("These project acceptance targets were added after the existing experiment. They are retrospective demo criteria, not independently validated business or legal thresholds. The demo supports 600 records; runtime varies by hosting environment.")


def system_pipelines():
    st.title("System Pipelines")
    st.write("Two connected processing workflows implement the business use case. One fine-tuned Hugging Face Transformer supplies demographic probabilities; the downstream audit computes aggregate statistics.")
    st.subheader("Pipeline 1 · Transformer demographic imputation")
    st.code("Profile text → BERT tokenizer → fine-tuned pretrained BERT → softmax → P(Group B)", language=None)
    st.markdown("Pretrained checkpoint: [google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)")
    st.write("The classification head is trained on synthetic Group A/Group B labels. Single Prediction and Batch Audit both use this inference workflow. The deployed artifact contains the fine-tuned weights and tokenizer.")
    st.subheader("Pipeline 2 · Probability-weighted fairness audit")
    st.code("Validated outcomes + P(Group B) → weighted group rates → fairness gaps → benchmark errors → CSV / JSON", language=None)
    st.write("Group B uses weight p and Group A uses weight 1 − p. The audit compares selection rates, equal opportunity and false-positive rates. Verified labels, when supplied, enable observed-versus-imputed error measurement. This workflow consumes Pipeline 1 and does not train a second model.")
    st.subheader("Offline training and evaluation")
    st.write("Start from Hugging Face pretrained weights. Fine-tuning is an explicit offline step in Colab; opening this app does not start training. The live app currently serves the previously fine-tuned checkpoint. The base checkpoint alone has no trained Group A/Group B classification head.")
    st.code("python scripts/generate_data.py\n# Run later in Colab when ready to fine-tune:\npython scripts/train.py --model-name google/bert_uncased_L-2_H-128_A-2\npython scripts/evaluate.py", language="bash")
    st.code("4,200 train → fine-tune pretrained BERT\n900 validation → select checkpoint by Macro-F1\nSaved checkpoint → train / validation / 900 test evaluation → evidence\nSaved model + tokenizer → Streamlit inference", language=None)
    st.caption("Course interpretation: these are two application workflows, not two distinct Hugging Face pipeline task types. The supplied slide does not define which interpretation the instructor requires.")
    st.subheader("Model storage after fine-tuning")
    st.code("zhengzhihust/fairnesslens-demographic-imputer", language=None)
    st.write("After fine-tuning and evaluation, use the Colab upload cell to publish the saved model and tokenizer to this Hugging Face repository. This is the configured upload destination; upload completion has not yet been verified. The app continues using its bundled model until the Hub artifact is available and HF_MODEL_ID is configured.")


def training_evidence(metrics):
    st.subheader("Pretrained model and fine-tuning configuration")
    metadata_path = MODEL_DIR / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        keys = ["base_model", "tokenizer", "epochs", "batch_size", "learning_rate", "max_length", "early_stopping_patience", "seed"]
        st.dataframe(pd.DataFrame([{"Parameter": key, "Value": str(metadata[key])} for key in keys if key in metadata]), hide_index=True, use_container_width=True)
    history_path = MODEL_DIR / "training_history.json"
    if history_path.exists():
        history = pd.DataFrame(json.loads(history_path.read_text()))
        st.subheader("Fine-tuning history")
        train = history.dropna(subset=["loss"])
        validation = history.dropna(subset=["eval_macro_f1"])
        left, right = st.columns(2)
        with left:
            st.caption("Training loss · logged mini-batch averages")
            st.line_chart(train.set_index("epoch")[["loss"]])
        with right:
            st.caption("Validation loss · full validation split per epoch")
            st.line_chart(validation.set_index("epoch")[["eval_loss"]])
        st.line_chart(validation.set_index("epoch")[["eval_macro_f1", "eval_roc_auc"]])
        best = validation.loc[validation["eval_macro_f1"].idxmax()]
        st.caption(f"Checkpoint selection: highest validation Macro-F1; best logged epoch {best['epoch']:.0f}, Macro-F1 {best['eval_macro_f1']:.4f}. Test metrics are excluded from checkpoint selection.")
        st.dataframe(validation[["epoch", "eval_loss", "eval_macro_f1", "eval_roc_auc", "eval_brier_score"]], hide_index=True, use_container_width=True)
    if "split_evaluation" in metrics:
        st.subheader("Saved-checkpoint evaluation across data splits")
        split_table = pd.DataFrame(metrics["split_evaluation"]).T
        st.dataframe(split_table[["records", "accuracy", "macro_f1", "roc_auc", "brier_score"]], use_container_width=True)
        st.caption(metrics["evaluation_protocol"])
    st.subheader("Held-out test confusion matrix")
    st.dataframe(pd.DataFrame(metrics["classification"]["confusion_matrix"], index=["Actual A", "Actual B"], columns=["Predicted A", "Predicted B"]), use_container_width=True)
