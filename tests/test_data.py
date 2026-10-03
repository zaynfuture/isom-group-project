from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isom_project.data import generate_dataset, sample_audit_data


def test_dataset_is_reproducible_and_complete():
    first = generate_dataset(500, seed=10)
    second = generate_dataset(500, seed=10)
    assert first.equals(second)
    assert set(first.columns) == {"record_id", "text", "group_b", "qualified", "decision", "split"}
    assert set(first.split) == {"train", "validation", "test"}
    assert first.group_b.nunique() == 2
    assert first.text.str.len().min() > 30


def test_default_audit_sample_has_at_least_500_records():
    assert len(sample_audit_data()) >= 500


def test_complete_dataset_has_fixed_modeling_splits():
    counts = generate_dataset()["split"].value_counts().to_dict()
    assert counts == {"train": 4200, "validation": 900, "test": 900}
