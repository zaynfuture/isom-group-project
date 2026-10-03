# FairnessLens within SpendLens

## Objective and boundary
Enterprise analysts explore spending, then screen gender, age-band and ethnicity disparities using authorized labels. No demographic inference from transactions. Spending disparity is not a discrimination verdict. Approval, pricing, offer eligibility or model-error fairness requires additional outcome data and a defined target population; not implemented by this spending screen.

## Contract and workflow
Transactions use the existing validated schema. Labels contain unique card_id, gender, age_band, ethnicity, consent=true and nonempty source. Unknown labels are excluded with coverage reported. Synthetic labels are independently generated only for synthetic transactions, with no real-world demographic claims.

Select one currency, time window, dimension, metric, reference, minimum card count and practical gap before running. Metrics: positive purchase amount per observed card, purchase count per observed card, or category share per card (zero-purchase cards undefined). Reference is user-selected, not a preferred population. At least ten cards per group; at least two eligible groups. Interactive limit: 5,000 cards / 30 eligible groups.

## Statistics and interpretation
One observation per card. Two-sided permutation difference-in-means tests (999 resamples, seed 5240), Holm correction within the selected report, independently bootstrapped percentile 95% gap intervals (unadjusted). Flag only if corrected p < .05 AND absolute gap exceeds the selected practical threshold. No flag is not proof of fairness. Ratios undefined at zero reference mean. Repeated exploratory runs are not multiplicity-controlled. Independent/exchangeable cards are assumed; shared owners, income, geography, unequal account coverage and missing demographic labels may invalidate or confound conclusions. No-activity accounts missing from the extract are outside scope.

## Outputs and acceptance
Aggregate report, gap chart, coverage and downloadable settings with date window, currency, threshold, method and seed. No card-level demographic export. Small groups suppressed; suppression alone is not a privacy guarantee. Tests cover null/strong differences, Holm, thresholds, consent, duplicates, missing labels, zero purchases and Streamlit rendering/execution.

## Evidence
Local knowledge health checked; bounded context 1,102 estimated tokens (cap 1,500), observation search-2bb14069-c3ec-4e09-89ec-c25208614753. Returned passages passage-e6055c8c6efde651b7e93ea4e1ae8100, passage-3a9b7fea15ac98a0d2980d67385ba142 and passage-4273ba342a8cd06bcca4e87107d3e6a1 were unrelated and not used as fairness evidence; context was truncated. Gap: no relevant local fairness methodology. External primary reference: https://fairlearn.org/v0.12/user_guide/assessment/perform_fairness_assessment.html (disaggregated assessment and context). Implementation uses NumPy/Pandas, not Fairlearn certification or a new ML pipeline.
