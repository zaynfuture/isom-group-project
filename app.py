"""Streamlit Cloud entry point for SpendLens."""
import runpy
from pathlib import Path


def main() -> None:
    """Launch the SpendLens Streamlit interface."""
    runpy.run_path(str(Path(__file__).resolve().parent / "app/main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
