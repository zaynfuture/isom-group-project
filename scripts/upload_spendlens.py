"""Compatibility command; implementation moved to model.registry.upload."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.registry.upload import main
if __name__ == "__main__":
    main()
