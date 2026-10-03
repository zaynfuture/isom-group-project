# SpendLens model lifecycle

Install `requirements-colab.txt` for training. Use `python -m model.training.merchant --help` or `python -m model.training.chronos --help`; publish using `python -m model.registry.upload --help` and verify using `python -m model.registry.verify` after an authorized upload. Training is never started by the web application.

Data contracts, inference, training and registry operations are separate packages. Small evaluation artifacts and verified Hub revisions are versioned under artifacts/. Weights and credentials must not be committed. Forecast training/input amounts and band thresholds use USD. FX display changes do not require retraining or republishing the existing models.
