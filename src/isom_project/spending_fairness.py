"""Compatibility import; implementation moved to app.services.disparity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.services.disparity import *  # noqa: F403
