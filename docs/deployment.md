# Streamlit Cloud deployment

Deploy repository `zaynfuture/isom-group-project`, branch `main`, entry point `app.py`. Runtime dependencies remain in root `requirements.txt`. There is no independent React build, Node server or additional API service required by this change.

Notebook dependencies belong in `requirements-notebook.txt`, not the Cloud runtime. Training runs outside the application. Public model inference should not require a Hugging Face write token; never expose publication credentials through app inputs or commit them.

The application reads the existing deployment manifest in `model/artifacts/spendlens_deployment.json`. Keep explicit verified model revisions, task architectures and evaluation directories aligned. A notebook upload does not change this file.

Before promoting a newly trained experiment:

1. Review full-run provenance and held-out metrics; do not promote smoke results.
2. Verify the label schema and forecast currency/window contracts against the serving registry.
3. Reload the exact uploaded revision and test inference through the application interface.
4. Commit the reviewed evidence and explicit manifest changes together.
5. Deploy and check the public app before announcing completion.

The notebook/workflow update intentionally leaves serving weights and the user interface unchanged.
