"""Compatibility import; implementation moved to model.data.transactions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.data.transactions import *  # noqa: F403
