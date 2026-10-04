"""English FairnessLens view; no automatic Streamlit multipage registration."""
import json
import pandas as pd
import streamlit as st
from app.services.disparity import FIELDS, METRICS, demo_labels, validate_labels, card_metrics, screen


def render(data, synthetic):
    st.write("FairnessLens · Spending disparity screening")
    st.write("Compare spending behavior across authorized gender, age and ethnicity groups. Identify differences for investigation, not demographic identities or automatic discrimination verdicts.")
    st.info("Spending differences alone do not establish unfair treatment. Decision fairness requires a defined business outcome (such as offer eligibility or approval), the relevant eligible population and contextual review. Those outcomes are not present in this transaction dataset.")
    template = pd.DataFrame(columns=["card_id", *FIELDS, "consent", "source"])
    st.download_button("Download demographic label template", template.to_csv(index=False), "authorized_labels_template.csv", "text/csv")
    if synthetic:
        choice = st.radio("Label source", ["Independent synthetic demo labels", "Upload authorized labels"])
    else:
        choice = "Upload authorized labels"
    if choice == "Independent synthetic demo labels":
        labels = demo_labels(data.card_id)
        st.caption("Demo demographics are independently generated and have no relationship to real identities. Ethnicity group names are placeholders. Demo results do not establish real-world disparities.")
    else:
        st.warning("Use only an approved, de-identified evaluation extract. Do not upload identifiable customer data to this public app.")
        uploaded = st.file_uploader("Authorized demographics CSV", type="csv", key="fairness_labels")
        if uploaded is None:
            st.info("Upload voluntarily supplied or otherwise authorized labels. Missing attributes remain Unknown and are excluded from comparisons.")
            return
        try:
            labels = validate_labels(pd.read_csv(uploaded, dtype=str))
        except Exception as exc:
            st.error(str(exc)); return
    field = st.selectbox("Demographic dimension", FIELDS, format_func=lambda x: {"gender":"Gender", "age_band":"Age band", "ethnicity":"Ethnicity"}[x])
    dates = st.date_input("Analysis window", (data.timestamp.min().date(), data.timestamp.max().date()))
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Select both start and end dates."); return
    filtered = data.loc[data.timestamp.dt.date.between(*dates)]
    if filtered.empty:
        st.info("No transactions in this window."); return
    metric = st.selectbox("Behavior metric", METRICS)
    category = st.selectbox("Spending category", sorted(filtered.category.unique())) if metric == METRICS[2] else None
    with st.expander("Comparison settings", expanded=False):
        minimum = st.number_input("Minimum distinct cards per group", 10, 500, 10)
        threshold = st.number_input("Minimum practical absolute gap", min_value=0.0, value=0.05 if metric == METRICS[2] else 0.0,
                                help="Share uses 0–1 units (0.05 = 5 percentage points); amount uses the selected currency; count uses purchases per card. Set before reviewing results.")
    values = card_metrics(filtered, metric, category)
    joined = values.merge(labels[["card_id", field]], on="card_id", how="left")
    counts = joined.loc[joined.value.notna() & joined[field].notna() & joined[field].ne("Unknown")].groupby(field).size()
    eligible = sorted(counts[counts >= minimum].index.tolist())
    st.caption(f"Cards with known {field}: {int(joined[field].notna().mul(joined[field].ne('Unknown')).sum())} / {len(joined)}. Only eligible groups appear in the reference selector; small groups are suppressed.")
    if len(eligible) < 2:
        st.warning("Insufficient data: at least two groups must meet the minimum card count."); return
    reference = st.selectbox("Reference group (comparison only, not a preferred group)", eligible)
    st.caption("Unit: observed card, not person. Positive purchases only; refunds are excluded. Counts and amounts cover the selected window, not active-month-normalized values. Category share is averaged across cards; cards with zero purchases have undefined shares. No-activity cards absent from the extract cannot be evaluated.")
    with st.expander("How comparisons work"):
        st.caption("999 seeded card-level resamples. Two-sided permutation tests; Holm correction across comparisons in this report only. Bootstrap 95% gap intervals are unadjusted. Repeated exploration across dimensions, dates or metrics is not corrected. Cards must be independent; shared owners, unequal observation coverage, income and location can confound results.")
    if st.button("Run disparity screening", type="primary"):
        try:
            with st.spinner("Comparing card-level behavior..."):
                report, coverage = screen(values, labels, field, reference, int(minimum), threshold)
            flags = int(report.review_flag.sum())
            st.metric("Comparisons requiring contextual review", flags)
            st.write("Observed disparities warrant further review." if flags else "No comparisons met both statistical and practical thresholds. This is not a finding that the system is fair.")
            st.dataframe(report, hide_index=True, use_container_width=True)
            st.bar_chart(report.set_index("group")[["gap"]])
            manifest = {"dimension":field, "metric":metric, "category":category, "reference":reference,
                        "currency":str(filtered.currency.iloc[0]), "start":str(dates[0]), "end":str(dates[1]),
                        "minimum_cards":int(minimum), "practical_gap":threshold, "seed":5240, "resamples":999,
                        "alpha":.05, "correction":"Holm within selected report", "label_source":choice,
                        "coverage":coverage, "interpretation":"Exploratory spending disparity, not proof of discrimination"}
            if "fx_snapshot_id" in filtered:
                manifest["fx_snapshots"] = sorted(filtered.fx_snapshot_id.unique().tolist())
                manifest["fx_policy"] = "UTC transaction-date reference FX, backward-only max 7 days; same reporting currency for all groups"
            with st.expander("Audit settings for this report"):
                st.json(manifest)
            st.download_button("Download aggregate disparity report", report.to_csv(index=False), "fairnesslens_disparities.csv", "text/csv")
            st.download_button("Download audit settings", json.dumps(manifest, indent=2), "fairnesslens_settings.json", "application/json")
        except Exception as exc:
            st.error(str(exc))
    st.markdown("Method context: [Fairlearn fairness assessment](https://fairlearn.org/v0.12/user_guide/assessment/perform_fairness_assessment.html). This screening uses NumPy/Pandas, not a Fairlearn model or certification.")
