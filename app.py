"""Streamlit Cloud entry point for SpendLens."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("spendlens_app.py")), run_name="__main__")
