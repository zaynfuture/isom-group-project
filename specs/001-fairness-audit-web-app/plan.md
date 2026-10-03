# Implementation Plan

## Architecture

```text
app.py (composition and navigation)
  -> ui.py (English presentation components)
  -> audit.py (validation, scoring orchestration, report payload)
  -> modeling.py (cached caller supplies HF pipeline)
  -> metrics.py (pure classification and fairness calculations)
  -> artifacts/ (versioned model and evaluation evidence)
```

## Decisions

- Keep a single Streamlit entry point for Community Cloud compatibility.
- Keep business logic framework-independent and unit-testable.
- Use soft group membership weights as the primary audit estimate.
- Package the compact fine-tuned model in the repository; allow `HF_MODEL_ID` as an optional deployment override.
- Generate JSON reports in memory and never persist uploaded records.

## Verification

1. Static compile of application, scripts, and package.
2. Unit tests for data contracts and metrics.
3. Model reload equivalence and held-out evaluation.
4. Local Streamlit health check.
5. Validate the deployment manifest and repository file sizes.
