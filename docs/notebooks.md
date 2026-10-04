# Notebook experiment guide

## Execution

Install `requirements-notebook.txt` in a dedicated environment and open either notebook in `notebooks/`. The first setup cell locates the repository locally. In Colab it clones the repository when absent and installs notebook dependencies. Restart the kernel if previously imported packages conflict with newly installed versions. Full experiments benefit from a GPU; local CPU smoke verification is not a claim of execution inside Colab.

Environment variables must be set before running the configuration cells:

| Variable | Default | Meaning |
| --- | --- | --- |
| SPENDLENS_SMOKE | 1 | Small execution check; use 0 for a full experiment |
| SPENDLENS_DATASET | legacy | legacy five-category fixture or lifestyle fourteen-category fixture |

Run cells in order. Save or download the output directory before a temporary Colab runtime is deleted. Each run uses a unique directory; previous checkpoints are not overwritten.

## Evaluation contracts

Merchant partitions are disjoint by card. The notebook checks that training includes every label, compares validation macro-F1, and opens test evaluation only after selection. Model inputs are merchant descriptions, not MCC labels or demographic attributes. Templated synthetic descriptions may nevertheless make classification artificially easy.

Forecast splits use increasing target months. Cards can recur across chronological partitions because the task predicts future spending for existing cards. Inputs are prior monthly positive purchases in USD; refunds are analyzed in the app but excluded from this forecasting target. Validation pinball loss selects the base model and fine-tuned checkpoint. Final reporting includes MAE, quantile loss, interval coverage and derived-band performance against a simple baseline.

The legacy fixture has already been inspected during earlier project work. Its test partition is not a newly acquired external holdout. Neither fixture establishes real-world demographic fairness or production forecast performance.

## Outputs and publication

Run directories hold environment information, revision identifiers, a data hash and the selected model directory. The selected directory includes weights, evaluation, training history, reload verification, a model card and task-specific configuration. Inspect both the parent run directory and selected checkpoint when archiving evidence.

`PUBLISH=False` is the default. Publication requires a reviewed non-smoke run, explicit approval in the notebook and authenticated access to `zhengzhihust`. Never place access tokens in notebook cells or commit notebook outputs containing credentials. The allowlist uploads model files and aggregate evidence, not transaction CSVs. Production repository names are protected.

After upload, the notebook reloads the returned Hub revision and checks predictions. An upload can succeed even if a later verification fails; only a completed verification produces the final publication receipt. Do not describe an unverified upload as a verified deployment.

Uploading an experiment does not promote it to the app. Serving promotion requires a separate review of label schema, forecasting contract, evaluation evidence and registry compatibility. Existing historical registry-validation scripts must not be assumed to accept arbitrary new notebook output layouts.
