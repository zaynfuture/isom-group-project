# SpendLens engineering instructions

Keep the Streamlit Cloud Python entry point. Do not introduce a separately hosted React/API application without a new user decision. The app serves inference; training lives in notebooks and model/training.

## Agent skills
### Issue tracker
Track specifications and work in GitHub Issues for zaynfuture/isom-group-project. See docs/agents/issue-tracker.md.
### Domain docs
Single-context: read GLOSSARY.md and relevant docs/adr/ before changes. See docs/agents/domain.md.

## Agreed test seams
Test public interfaces for data splitting, executable notebook smoke workflows, saved-model reload equivalence and Streamlit inference. Use one failing behavior test, then implementation, then the next vertical slice. Do not substitute syntax checks for actual training/evaluation runs.

## Evidence and safety
Keep legacy-five and lifestyle-fourteen data distinct. Select models/checkpoints using validation, never test. Record actual results and skipped runs honestly. Do not upload transactions or credentials. Hub publication is opt-in and serving-manifest promotion is a separate reviewed action.
