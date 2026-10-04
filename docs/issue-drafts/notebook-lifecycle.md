# Complete notebook-first model lifecycle on Streamlit Cloud

UNPUBLISHED: GitHub write access is unavailable. Intended repository: zaynfuture/isom-group-project.

## Acceptance
- Keep current Streamlit UI, Cloud entry point and pinned serving models unchanged.
- Expose preparation, candidate comparison, fine-tuning, final evaluation and save/reload in two complete Jupyter/Colab notebooks.
- Separate smoke software verification from full experiments and legacy data from lifestyle data.
- Gate publication to zhengzhihust; verify remote pinned reload; do not automatically promote.
- Regenerate current README, notebook guide and architecture documentation from actual implementation.
- Verify splits, notebook execution, local reload and Streamlit inference.

## Approved engineering setup
AGENTS.md; GitHub Issues; GLOSSARY.md + docs/adr/; public-interface tests. React migration cancelled because Streamlit Cloud compatibility is mandatory.
