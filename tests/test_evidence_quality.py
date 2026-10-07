from pathlib import Path
from streamlit.testing.v1 import AppTest
from model.data.transactions import demo_transactions
from model.experiments import merchant_split


def test_merchant_template_overlap_is_exposed():
    from model.evaluation_quality import merchant_overlap
    result = merchant_overlap(merchant_split(demo_transactions()))
    assert result['test_card_overlap_count'] == 0
    assert result['test_template_overlap_rate'] == 1.0
    assert result['train_template_count'] == 5


def test_ui_does_not_certify_synthetic_scores():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=30).run()
    next(x for x in app.button if x.label == 'Review model performance').click().run()
    assert not app.exception
    assert not any('target met' in str(x.value) for x in app.markdown)
    assert any('unseen merchants' in x.value for x in app.warning)
    assert any(x.label == 'Synthetic test Macro-F1' for x in app.metric)
