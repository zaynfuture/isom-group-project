# Merchant evaluation: interpretation and release gap

The historical merchant score (Macro-F1 1.000) is retained, not rewritten. Its data generator uses five category-specific merchant names followed by a branch number. Holding out cards prevents card overlap but does not hold out merchant-name templates. The Model Evidence page now exposes this limitation and computes row-weighted exact-description/template overlap for the legacy fixture.

The former 0.8 merchant threshold was a project heuristic, not a documented industry standard. The app no longer reports “target met”; it reports external validation pending. A majority-class baseline is useful as a sanity check, but insufficient to demonstrate the value of a Transformer on template-driven data.

Before claiming business readiness, obtain authorized representative labels with merchant identity and timestamps, define a business-specific error budget, isolate independent merchants and future periods, and compare a lexical baseline against the Transformer. Select models and thresholds on validation only. Report class-level precision/recall, uncertainty, calibration and out-of-scope rejection. Do not retrain repeatedly against the current inspected test set or fabricate new benchmark results.

This change fixes evidence presentation and adds diagnostic/regression checks. It does not improve or replace model weights, prove real-world accuracy, or certify production readiness. The existing forecast also remains experimental where band Macro-F1 fails to beat its baseline.

References: [scikit-learn grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [metrics and dummy baselines](https://scikit-learn.org/stable/modules/model_evaluation.html).
