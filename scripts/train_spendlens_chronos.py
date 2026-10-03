"""Compatibility command; implementation moved to model.training.chronos."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.training.chronos import main
if __name__ == "__main__":
    main()
