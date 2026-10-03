"""Compatibility import; implementation moved to app.views.fairness."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.views.fairness import *  # noqa: F403
