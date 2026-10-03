"""Compatibility import; implementation moved to model.inference.registry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.inference.registry import *  # noqa: F403
