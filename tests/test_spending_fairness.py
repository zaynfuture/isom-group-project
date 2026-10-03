import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from isom_project.spending_fairness import demo_labels, validate_labels, screen, holm, card_metrics, METRICS
from isom_project.spending import demo_transactions


def fixture():
    labels = demo_labels([str(i) for i in range(40)])
    labels["gender"] = ["A"]*20 + ["B"]*20
    values = pd.DataFrame({"card_id":labels.card_id, "value":[10.]*20 + [100.]*20})
    return values, labels


def test_strong_gap_and_equal_values():
    values, labels = fixture()
    report, coverage = screen(values, labels, "gender", "A", draws=199)
    assert report.iloc[0].gap == 90
    assert report.iloc[0].review_flag
    assert coverage["known_label_cards"] == 40
    values["value"] = 10.
    report, _ = screen(values, labels, "gender", "A", draws=99)
    assert report.iloc[0].p_holm == 1
    assert not report.iloc[0].review_flag


def test_validation_missing_and_small_groups():
    values, labels = fixture()
    bad = labels.copy(); bad.loc[0,"consent"] = False
    with pytest.raises(ValueError, match="consent"):
        validate_labels(bad)
    with pytest.raises(ValueError, match="duplicate"):
        validate_labels(pd.concat([labels, labels.iloc[:1]]))
    labels.loc[:10,"gender"] = " "
    with pytest.raises(ValueError, match="Insufficient"):
        screen(values, labels, "gender", "A", draws=99)
    assert validate_labels(labels).gender.eq("Unknown").sum() == 11


def test_holm_and_practical_threshold():
    assert np.allclose(holm([.03,.01,.2]), [.06,.03,.2])
    values, labels = fixture()
    report, _ = screen(values, labels, "gender", "A", practical_gap=100, draws=99)
    assert not report.iloc[0].review_flag


def test_card_aggregation_and_zero_share():
    data = demo_transactions()
    metrics = card_metrics(data, METRICS[1])
    assert len(metrics) == 120
    assert metrics.value.eq(64).all()
    data.loc[data.card_id.eq("CARD-0000"),"amount"] = -1
    shares = card_metrics(data, METRICS[2], "Grocery")
    assert shares.loc[shares.card_id.eq("CARD-0000"),"value"].isna().all()
    assert shares.value.dropna().between(0,1).all()


def test_page_renders_and_runs():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("FairnessLens").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Run disparity screening").click().run()
    assert not app.exception
    assert any(m.label == "Comparisons requiring contextual review" for m in app.metric)
