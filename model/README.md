# SpendLens model lifecycle

The primary experiment workflow is now `notebooks/01_merchant_experiment.ipynb` and `notebooks/02_forecast_experiment.ipynb`. Install `requirements-notebook.txt`; see `docs/notebooks.md` for model selection, training, evaluation, reload and optional publication. Training is never started by the web application.

Existing `model.training` and `model.registry` commands remain compatibility utilities for historical workflows. Their historical artifact paths and promotion behavior must not be assumed to support arbitrary new experiment folders. Notebook publication deliberately does not promote serving revisions.

Data contracts, inference, training and registry operations are separate packages. Small evaluation artifacts and verified Hub revisions are versioned under artifacts/. Weights and credentials must not be committed. Forecast training/input amounts and band thresholds use USD. FX display changes do not require retraining or republishing the existing models.
