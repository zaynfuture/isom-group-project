# App/model separation and currency normalization

## Objective and boundaries
Serve enterprise spending analysis with a separately maintainable application and model lifecycle. This separation is not a claim of production security or financial settlement accuracy.

```
app/main.py                Streamlit orchestration
app/views/                 FairnessLens presentation
app/services/              FX normalization and disparity analysis
model/data/                Transaction contracts and MCC mapping
model/inference/           Pinned model loading and prediction
model/training/            Merchant and forecast fine-tuning
model/registry/            Hugging Face publication and verification
model/artifacts/           Versioned evaluation evidence and deployment manifest
```

The app may call data/inference interfaces, but never trains or publishes models during requests. Model modules do not import Streamlit or app modules. Root app.py preserves the Streamlit Cloud entry point. Old script and source imports are compatibility wrappers, not duplicate implementations.

## Currency contract
- Default reporting currency: USD. The selector offers 50 mainstream currencies; an optional provider catalog expands coverage (165 entries observed during verification; provider coverage may change).
- CurrencyConverter performs Decimal conversion using historical reference quotes obtained from Frankfurter v2. Babel formats currency codes and precision. No embedded or silently stale package rates are used.
- Transactions are converted before aggregation; original amounts and currency codes remain available. Refunds retain their sign. Mixed currencies must never be summed before conversion.
- For each currency use the latest available quote on or before the transaction date, at most seven days old. Missing, future, unsupported or stale rates stop the analysis. No implicit 1:1 fallback.
- Only currency codes and a date range go to the public FX provider, not card IDs, transaction values or customer data. Rate caching lasts one hour. Audit downloads include applied rates, dates, provider, retrieval time and snapshot identifiers.
- Rates are blended reference rates, not issuer settlement rates. Historical revisions can change subsequent results. Providers' terms and market/date coverage require review for production deployment.
- Forecast inputs and spending-band thresholds remain USD, matching training. Display conversion uses the month-end forecast origin, never future realized FX. Forecast USD columns remain in results. Currency conversion does not validate a model for a different country's spending patterns.

## Acceptance checks
Automated tests cover mixed currencies, refunds, identity conversion, invalid rates, unavailable/stale rates, future-data exclusion, forecast-origin conversion and app/model import boundaries. Streamlit smoke checks cover USD startup and alternate reporting currency. Existing model and FairnessLens regression tests must continue passing.

## Operations
Install requirements.txt for deployment, requirements-colab.txt for training, requirements-dev.txt for development. Run `streamlit run app.py`. Train with `python -m model.training.merchant` and `python -m model.training.chronos`. Publish with `python -m model.registry.upload` and verify with `python -m model.registry.verify`. Colab uses the same modules. Never commit model weights or credentials.

## Evidence and gaps
Local knowledge health was healthy (54 documents, 1,529 passages). The 609-token context bundle was capped at 1,500 tokens; search ID `search-77301464-3c52-4a01-bc01-43b44a71f3cf`. Passages `passage-f2cad529e3b28575970b6af9d7248bec` and `passage-b1424c4c6a6c9b61dbdd8c32457131cc` concern exchange-rate concepts, not the packages or architecture. Those local gaps required official sources:
- https://pypi.org/project/CurrencyConverter/
- https://frankfurter.dev/
- https://frankfurter.dev/python/

Authentication, controlled private-data hosting, monitoring, settlement-grade FX and representative real-world model validation remain separate production requirements.
