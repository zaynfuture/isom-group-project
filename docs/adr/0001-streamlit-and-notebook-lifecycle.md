# ADR 0001: retain Streamlit Cloud; expose model lifecycle in notebooks

Status: accepted by user (Streamlit Cloud required; React migration cancelled).

The existing Streamlit entry point and app/model split remain. Independent React plus an HTTP backend would change hosting and is out of scope. React custom components remain a possible future UI change, not part of this work.

Two Jupyter notebooks implement candidate loading, validation-based selection, fine-tuning, final evaluation, save/reload and optional Hub publication in visible Python cells. Shared code contains only stable data/split/inference interfaces; notebooks do not shell out to opaque training commands. They run in local Jupyter or Colab, with a small actual smoke mode and larger full mode.

Run outputs are isolated and never overwrite the serving manifest. Default legacy data aligns with published label contracts; an explicit lifestyle mode can explore fourteen labels but cannot be silently promoted into the current five-label merchant app. Training, publication and promotion are separate actions.

The user approved test seams for splits, complete small notebook runs, save/reload and app inference. We follow the reviewed mattpocock/skills reference d81f3a183412e71a5b1e84ca21bc1a35eea03a60: explicit domain decisions, small vertical test-first slices and evidence-backed verification. Skills are not globally installed.
