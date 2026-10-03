"""English Streamlit application for card spending analytics."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
import streamlit as st
from isom_project.spending import demo_transactions, validate_transactions, monthly_examples, labeled_cohorts, MCC
from isom_project.spend_models import BASE_MODEL, available, load, infer, ROOT, deployed, source_for
from isom_project.mcc_reference import SOURCE, VERSION, REFERENCE

st.set_page_config(page_title="SpendLens | Card Spending Analytics", page_icon="💳", layout="wide")
st.title("SpendLens")
st.caption("Card transaction analytics · Merchant classification · Future spending bands")
page = st.sidebar.radio("Workspace", ["Business Overview", "Transaction Explorer", "Spending Analytics", "Model Pipelines", "Cohort Analysis", "Model Evidence"])
source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload transactions"])
@st.cache_data
def demo():
    return demo_transactions()
@st.cache_resource(max_entries=1)
def model(task):
    return load(task)


def evidence(task):
    path = ROOT / "artifacts" / f"spendlens_{task}" / "evaluation.json"
    return json.loads(path.read_text()) if path.exists() else None


def task_status(task):
    result = evidence(task)
    if not result:
        return "Awaiting fine-tuning"
    score = result["test"]["test_macro_f1"]
    target_met = score >= .8 if task == "merchant" else score > result["baseline"]["macro_f1"]
    return "Trained · target met" if target_met else "Trained · target not met"

data = demo()
if source == "Upload transactions":
    uploaded = st.sidebar.file_uploader("Transactions CSV", type="csv")
    if uploaded is None:
        st.info("Upload anonymized transactions to continue. Required columns are available in the demo CSV template.")
        st.download_button("Download demo / template CSV", data.to_csv(index=False), "transactions.csv", "text/csv")
        st.stop()
    try:
        data = validate_transactions(pd.read_csv(uploaded, dtype={"mcc": str, "card_id": str}))
    except Exception as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
currency = st.sidebar.selectbox("Currency (amounts are never mixed)", sorted(data.currency.unique()))
data = data.loc[data.currency.eq(currency)].copy()
st.sidebar.caption("Uploads stay in session memory. Use anonymized card tokens, not payment card numbers. This public demo is not an ingestion endpoint for live customer records.")

if page == "Business Overview":
    st.subheader("Enterprise Spending Behavior Analytics")
    st.write("SpendLens helps enterprises analyze card spending behavior, understand category and portfolio trends, improve merchant-category coverage, and forecast future spending bands. Business analysts can compare aggregate behavior across authorized customer groups to support planning and portfolio management.")
    st.dataframe(pd.DataFrame([
        {"Objective": "Merchant category prediction", "Acceptance target": "Test Macro-F1 ≥ 0.80; compare majority baseline", "State": task_status("merchant")},
        {"Objective": "Next-month spending band", "Acceptance target": "Macro-F1 exceeds previous-2-month baseline", "State": task_status("forecast")},
        {"Objective": "Portfolio exploration", "Acceptance target": "Browse and summarize at least 500 transactions", "State": "Available"},
    ]), hide_index=True, use_container_width=True)
    st.write("Two task-specific models start from Hugging Face pretrained DistilBERT weights and fine-tune separately. Model Evidence reports the saved run results and simple baselines; the Colab notebook reproduces this workflow.")
    st.markdown(f"[Hugging Face base model](https://huggingface.co/distilbert/distilbert-base-uncased) · [Citi MCC reference]({SOURCE})")
    st.caption("MCC is a four-digit merchant classification. This demo uses a five-code crosswalk; it is not a complete universal industry taxonomy. Other codes remain unmapped.")

elif page == "Transaction Explorer":
    a,b,c = st.columns(3)
    a.metric("Transactions", f"{len(data):,}"); b.metric("Cards", data.card_id.nunique()); c.metric("Currency", currency)
    st.caption("All records are available below; scroll inside the table.")
    st.dataframe(data, height=550, hide_index=True, use_container_width=True)
    st.download_button("Download current transactions", data.to_csv(index=False), "transactions.csv", "text/csv")
    st.subheader("Reviewed MCC reference")
    st.dataframe(pd.DataFrame(REFERENCE).T.rename_axis("MCC"), use_container_width=True)
    st.caption(f"Source: user-supplied Citi manual. Version: {VERSION}. Descriptions are normalized summaries. Brands may differ; consult the source before expanding coverage.")
    st.markdown(f"[Open original MCC manual]({SOURCE})")

elif page == "Spending Analytics":
    st.caption("Positive amounts are purchases; negative amounts are refunds. Charts display net amounts in the selected currency. Dates are normalized to UTC.")
    a,b,c = st.columns(3)
    a.metric("Purchases", f"{data.loc[data.amount.gt(0),'amount'].sum():,.2f} {currency}")
    b.metric("Refunds", f"{-data.loc[data.amount.lt(0),'amount'].sum():,.2f} {currency}")
    c.metric("Net amount", f"{data.amount.sum():,.2f} {currency}")
    st.bar_chart(data.groupby("category").amount.sum())
    st.line_chart(data.set_index("timestamp").amount.resample("MS").sum())
    summary = data.groupby("card_id").agg(transactions=("transaction_id","size"),net_amount=("amount","sum"),average_transaction=("amount","mean"))
    st.dataframe(summary, height=400, use_container_width=True)

elif page == "Model Pipelines":
    task = st.selectbox("Pipeline", ["merchant", "forecast"])
    st.markdown(f"Pretrained model: `{BASE_MODEL}`")
    if available(task):
        st.caption(f"Inference checkpoint: {source_for(task)}")
    if task == "merchant":
        st.write("Merchant description → clean text and tokenize → fine-tuned DistilBERT → label probabilities → suggested category. Known MCC mappings stay authoritative; low-confidence suggestions require review.")
        texts = [text.strip() for text in st.text_area("Merchant descriptions, one per line", "Fresh Market 12\nHarbor Hotel 8").splitlines() if text.strip()]
    else:
        st.write("Two complete months of category purchase totals → serialized monthly context → fine-tuned DistilBERT → next-month purchase band. Inputs exclude future transactions and demographic labels.")
        st.caption("USD bands: LOW < 200; MEDIUM 200–399.99; HIGH ≥ 400. These demo thresholds require business calibration.")
        if currency != "USD":
            st.info("The configured forecast bands support USD only."); st.stop()
        examples = monthly_examples(data)
        st.dataframe(examples, height=350, use_container_width=True)
        st.caption("Historical backtest examples; the last observed month is excluded as a target. Complete monthly account coverage is required.")
        limit = st.selectbox("Records to score", [20, 120, 600], help="Limit inference work per run on shared cloud resources.")
        texts = examples.text.head(limit).tolist() if not examples.empty else []
    if not available(task):
        st.info("Awaiting fine-tuning. Configure the task checkpoint after Colab training, evaluation and upload to zhengzhihust.")
    if st.button("Run Transformer inference", disabled=not available(task) or not texts):
        try:
            labels, scores = infer(model(task), texts)
            result = pd.DataFrame({"input":texts,"prediction":labels,"probability":scores})
            result["review_required"] = result.probability < .7
            st.dataframe(result, use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")

elif page == "Cohort Analysis":
    st.write("Upload voluntarily provided demographic labels with card_id, consent, source and the selected field. Gender and ethnicity are never inferred from transactions. Missing labels stay Unknown.")
    field = st.selectbox("Authorized label", ["age_band", "gender", "ethnicity"])
    labels_file = st.file_uploader("Authorized labels CSV", type="csv")
    if labels_file:
        try:
            labels = pd.read_csv(labels_file, dtype={"card_id":str})
            summary = labeled_cohorts(data, labels, field)
            st.caption("Only category/group cells containing at least 10 distinct cards are displayed. This suppression alone is not a formal privacy guarantee. Cards are not necessarily unique people.")
            st.dataframe(summary, use_container_width=True)
            st.download_button("Download aggregate cohorts", summary.to_csv(index=False), "cohorts.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))

else:
    st.write("Task-specific evidence is generated only after fine-tuning. Previous FairnessLens demographic results do not evaluate these models.")
    for task in ["merchant", "forecast"]:
        st.subheader(task.title())
        path = ROOT / "artifacts" / f"spendlens_{task}" / "evaluation.json"
        if path.exists():
            result = json.loads(path.read_text())
            a,b = st.columns(2)
            a.metric("Test Macro-F1", f"{result['test']['test_macro_f1']:.3f}")
            b.metric("Baseline Macro-F1", f"{result['baseline']['macro_f1']:.3f}")
            st.write(task_status(task))
            st.caption(f"Baseline: {result['baseline']['name']}. Data: {result['data_source']}.")
            st.dataframe(pd.DataFrame(result['confusion_matrix'], index=result['labels'], columns=result['labels']), use_container_width=True)
            history_path = path.with_name("training_history.json")
            if history_path.exists():
                history = pd.DataFrame(json.loads(history_path.read_text()))
                validation = history.dropna(subset=['eval_macro_f1'])
                st.line_chart(validation.set_index('epoch')[['eval_loss','eval_macro_f1']])
            if deployed(task):
                st.markdown(f"[Hugging Face checkpoint](https://huggingface.co/{deployed(task)['repo_id']})")
            with st.expander("Complete evaluation evidence"):
                st.json(result)
        else:
            st.info("No trained checkpoint or evaluation evidence published yet.")
    st.write("Merchant evaluation holds out entire cards. Forecast evaluation uses chronological target-month splits; two prior months are inputs. Compare both tasks against a simple baseline. Synthetic performance is a software demonstration, not external business validation.")
