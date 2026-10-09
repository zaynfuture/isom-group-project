from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_business_objectives_and_complete_model_partitions():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert any('Company / project organization: SpendLens' in x.value for x in app.markdown)
    assert any('Proposed acceptance criteria' in x.value for x in app.markdown)
    next(x for x in app.button if x.label == 'Review transactions').click().run()
    assert not app.exception
    counts = [len(x.value) for x in app.dataframe]
    for expected in [13213, 5376, 1152]:
        assert expected in counts
    assert [x.label for x in app.tabs] == ['Training', 'Validation', 'Testing']
    next(x for x in app.selectbox if x.label == 'Training dataset task').set_value('Forecast').run()
    assert not app.exception
    counts = [len(x.value) for x in app.dataframe]
    assert 360 in counts and counts.count(120) == 2
