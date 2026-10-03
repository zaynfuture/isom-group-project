"""English Streamlit application for card spending analytics."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from model.data.transactions import demo_transactions, validate_transactions, monthly_examples, labeled_cohorts, MCC
from model.inference.registry import BASE_MODEL, BASE_MODELS, available, load, infer, ROOT, deployed, source_for
from model.inference.forecasting import numeric_examples, predict_amounts
from model.data.mcc_reference import SOURCE, VERSION, REFERENCE
from app.services.currency import (COMMON_CURRENCIES, currency_label, money, fetch_catalog,
    fetch_snapshot, FXBook, FXError, convert_transactions, display_forecast)

st.set_page_config(page_title="SpendLens | Card Spending Analytics", page_icon="💳", layout="wide")
st.title("SpendLens")
st.caption("Card transaction analytics · Merchant classification · Future spending bands · FairnessLens")
page = st.sidebar.radio("Workspace", ["Business Overview", "Transaction Explorer", "Spending Analytics", "Model Pipelines", "Cohort Analysis", "FairnessLens", "Model Evidence"])
source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload transactions"])
@st.cache_data
def demo():
    return demo_transactions()
@st.cache_resource(max_entries=1)
def model(task):
    return load(task)


@st.cache_data(ttl=3600, max_entries=32)
def cached_rates(currencies, start, end):
    return fetch_snapshot(currencies, start, end)


@st.cache_data(ttl=86400)
def currency_catalog():
    return fetch_catalog()


def book_for(frame, target):
    codes = tuple(sorted(set(frame.currency) | {target, "USD"}))
    return FXBook(cached_rates(codes,frame.timestamp.min().date(),frame.timestamp.max().date()))


def evidence(task):
    path = ROOT / "model" / "artifacts" / deployed(task).get("artifact_dir", f"spendlens_{task}") / "evaluation.json"
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
raw_data = data.copy()
currencies = sorted(set(COMMON_CURRENCIES) | set(raw_data.currency))
if st.sidebar.checkbox("Show all provider currencies", value=False):
    try:
        currencies = sorted(set(currencies) | set(currency_catalog()))
    except FXError as exc:
        st.sidebar.warning(str(exc))
currencies = ["USD"] + [c for c in currencies if c != "USD"]
currency = st.sidebar.selectbox("Reporting currency", currencies, index=0, format_func=currency_label)
st.sidebar.caption("Default: USD. Mixed-currency transactions are converted, not filtered out. Historical reference FX; not card-network settlement rates.")
fx_book = None
try:
    if not raw_data.currency.eq(currency).all():
        with st.spinner("Loading transaction-date exchange rates..."):
            fx_book = book_for(raw_data,currency)
    data = convert_transactions(raw_data,currency,fx_book)
except (FXError, ValueError) as exc:
    st.error(f"Currency conversion unavailable: {exc}")
    st.info("No partial or 1:1 substitution was used. Retry, select the original currency for a single-currency file, or check the date/currency coverage.")
    st.stop()
st.sidebar.caption("Uploads stay in session memory. Use anonymized card tokens, not payment card numbers. This public demo is not an ingestion endpoint for live customer records.")
with st.expander("Currency conversion details"):
    st.write(f"Reporting currency: {currency}. Original amounts and currency codes remain in transaction exports. Conversion uses CurrencyConverter with Frankfurter reference-rate data, using the latest observation on or before each UTC transaction date (at most 7 days old).")
    st.caption("Historical rates can be revised by providers; snapshots are retrieved today, not guaranteed publication-time vintages. No future-dated observations are used. Daily reference rates exclude bank spreads and transaction fees. Decimal arithmetic is used for conversion; analytical tables retain unrounded floating-point amounts and are not an accounting ledger.")
    fx_audit = data[["original_currency","currency","fx_rate","fx_source_date","fx_target_date","fx_provider","fx_retrieved_at","fx_snapshot_id"]].drop_duplicates()
    st.dataframe(fx_audit,height=220,hide_index=True,use_container_width=True)
    st.download_button("Download FX audit",fx_audit.to_csv(index=False),"spendlens_fx_audit.csv","text/csv")

if page == "Business Overview":
    st.subheader("Enterprise Spending Behavior Analytics")
    st.write("SpendLens helps enterprises analyze card spending behavior, understand category and portfolio trends, improve merchant-category coverage, and forecast future spending bands. Business analysts can compare aggregate behavior across authorized customer groups to support planning and portfolio management.")
    st.write("FairnessLens extends this workflow with gender, age-band and ethnicity spending-disparity screening using authorized labels, card-level comparisons and uncertainty estimates. Differences prompt contextual review; they do not alone demonstrate unfair treatment.")
    st.dataframe(pd.DataFrame([
        {"Objective": "Merchant category prediction", "Acceptance target": "Test Macro-F1 ≥ 0.80; compare majority baseline", "State": task_status("merchant")},
        {"Objective": "Next-month spending band", "Acceptance target": "Macro-F1 exceeds previous-2-month baseline", "State": task_status("forecast")},
        {"Objective": "Portfolio exploration", "Acceptance target": "Browse and summarize at least 500 transactions", "State": "Available"},
    ]), hide_index=True, use_container_width=True)
    st.write("Two task-specific Transformers are fine-tuned separately: MiniLM for merchant text classification and Chronos-Bolt Tiny for numeric monthly spending forecasts. Model Evidence reports actual results and baselines; the Colab notebook reproduces this workflow.")
    st.markdown(f"[MiniLM base model](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) · [Chronos-Bolt base model](https://huggingface.co/amazon/chronos-bolt-tiny) · [Citi MCC reference]({SOURCE})")
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
    a.metric("Purchases", money(data.loc[data.amount.gt(0),'amount'].sum(),currency))
    b.metric("Refunds", money(-data.loc[data.amount.lt(0),'amount'].sum(),currency))
    c.metric("Net amount", money(data.amount.sum(),currency))
    st.bar_chart(data.groupby("category").amount.sum())
    st.line_chart(data.set_index("timestamp").amount.resample("MS").sum())
    summary = data.groupby("card_id").agg(transactions=("transaction_id","size"),net_amount=("amount","sum"),average_transaction=("amount","mean"))
    st.dataframe(summary, height=400, use_container_width=True)

elif page == "Model Pipelines":
    task = st.selectbox("Pipeline", ["merchant", "forecast"])
    st.markdown(f"Pretrained model: `{BASE_MODELS[task]}`")
    if available(task):
        st.caption(f"Inference checkpoint: {source_for(task)}")
    if task == "merchant":
        st.write("Merchant description → clean text and tokenize → fine-tuned MiniLM → label probabilities → suggested category. Known MCC mappings stay authoritative; low-confidence suggestions require review. Softmax scores are not calibrated confidence guarantees.")
        texts = [text.strip() for text in st.text_area("Merchant descriptions, one per line", "Fresh Market 12\nHarbor Hotel 8").splitlines() if text.strip()]
    else:
        st.write("Two complete monthly purchase totals → numeric scaling and patching → fine-tuned Chronos-Bolt → next-month amount quantiles → spending band. Inputs exclude future transactions and demographic labels. Two observations provide very limited forecasting evidence.")
        st.caption("USD bands: LOW < 200; MEDIUM 200–399.99; HIGH ≥ 400. These demo thresholds require business calibration.")
        try:
            if not raw_data.currency.eq("USD").all() and fx_book is None:
                fx_book = book_for(raw_data,"USD")
            usd_data = convert_transactions(raw_data,"USD",fx_book)
        except FXError as exc:
            st.error(f"Cannot prepare USD model inputs: {exc}"); st.stop()
        examples = numeric_examples(usd_data)
        st.caption(f"Model inputs and band thresholds remain USD. Forecast amounts are displayed in {currency} using FX available at the forecast origin (end of the previous month), not future realized FX. The USD-trained model has not been validated on foreign-market behavior.")
        st.dataframe(examples, height=350, use_container_width=True)
        st.caption("Historical backtest examples; the last observed month is excluded as a target. Complete monthly account coverage is required.")
        limit = st.selectbox("Records to score", [20, 120, 600], help="Limit inference work per run on shared cloud resources.")
        texts = examples.context.head(limit).tolist() if not examples.empty else []
    if not available(task):
        st.info("Awaiting fine-tuning. Configure the task checkpoint after Colab training, evaluation and upload to zhengzhihust.")
    if st.button("Run Transformer inference", disabled=not available(task) or not texts):
        try:
            if task == "forecast":
                result = predict_amounts(model(task), texts)
                result = display_forecast(result,examples.target_month.head(len(result)).tolist(),currency,fx_book)
                result.insert(0,"card_id",examples.card_id.head(len(result)).tolist())
                result.insert(1,"target_month",examples.target_month.head(len(result)).tolist())
                st.caption("p10–p90 is a nominal 80% forecast interval, not a classification probability. Coverage must be validated. These are historical backtest forecasts, not live next-month predictions.")
                if evidence(task) and evidence(task)["test"]["test_macro_f1"] <= evidence(task)["baseline"]["macro_f1"]:
                    st.warning("This model has not beaten the baseline on spending-band Macro-F1. Treat it as an experimental model, not an automated business decision rule.")
            else:
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

elif page == "FairnessLens":
    from app.views.fairness import render
    render(data, source == "Synthetic demo")

else:
    st.write("Task-specific evidence is generated only after fine-tuning. Previous FairnessLens demographic results do not evaluate these models.")
    for task in ["merchant", "forecast"]:
        st.subheader(task.title())
        path = ROOT / "model" / "artifacts" / deployed(task).get("artifact_dir",f"spendlens_{task}") / "evaluation.json"
        if path.exists():
            result = json.loads(path.read_text())
            a,b = st.columns(2)
            a.metric("Test Macro-F1", f"{result['test']['test_macro_f1']:.3f}")
            b.metric("Baseline Macro-F1", f"{result['baseline']['macro_f1']:.3f}")
            st.write(task_status(task))
            st.caption(f"Baseline: {result['baseline']['name']}. Data: {result['data_source']}.")
            if "test_mae" in result["test"]:
                c,d = st.columns(2)
                c.metric("Test amount MAE (USD)",f"{result['test']['test_mae']:.2f}")
                d.metric("Baseline amount MAE (USD)",f"{result['baseline']['mae']:.2f}")
                st.caption(f"Nominal 80% interval empirical coverage: {result['test']['test_interval_80_coverage']:.1%}")
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
