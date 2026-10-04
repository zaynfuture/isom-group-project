"""Opt-in real-weight acceptance check through the public Streamlit interface."""
import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

@pytest.mark.skipif(os.getenv('SPENDLENS_RUN_MODEL_TESTS') != '1', reason='Requires cached or downloadable deployed model weights')
def test_streamlit_real_model_inference():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=120).run()
    next(x for x in app.button if x.label == 'Explore models').click().run()
    next(x for x in app.button if x.label == 'Suggest categories').click().run()
    assert not app.exception
    assert not app.error
    assert any('probability' in table.value.columns and len(table.value) == 2 for table in app.dataframe)
    next(x for x in app.radio if x.label == 'What would you like to do?').set_value('forecast').run()
    next(x for x in app.button if x.label == 'Run historical forecast').click().run()
    assert not app.exception
    assert not app.error
    assert any('p90' in table.value.columns and len(table.value) == 20 for table in app.dataframe)
