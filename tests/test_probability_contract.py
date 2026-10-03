import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import pytest
from isom_project.audit import validate_audit_frame, score_audit_frame


def frame():
    return pd.DataFrame({"text": ["synthetic example"], "decision": [1], "qualified": [0]})


def test_fractional_binary_label_is_not_silently_truncated():
    data = frame()
    data["decision"] = .5
    with pytest.raises(ValueError):
        validate_audit_frame(data)


@pytest.mark.parametrize("values", [[float("nan")], [1.1], [], [[.2]]])
def test_invalid_probability_output_is_rejected(values):
    with pytest.raises(ValueError):
        score_audit_frame(frame(), lambda texts: values)
