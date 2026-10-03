from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isom_project.data import write_dataset


if __name__ == "__main__":
    path = write_dataset()
    print(f"Wrote {path}")

