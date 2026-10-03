"""Compatibility Streamlit entry point."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent/"app/main.py"),run_name="__main__")
