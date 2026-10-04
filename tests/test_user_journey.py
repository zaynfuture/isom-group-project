"""Offline UI acceptance checks for the guided analyst journey."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception

def test_guided_journey_and_default_currency():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.session_state['workspace'] == 'Business Overview'
    assert next(x for x in app.selectbox if x.label == 'Reporting currency').value == 'USD'
    click(app, 'Open prepare your data')
    assert app.session_state['workspace'] == 'Transaction Explorer'
    assert any(len(table.value) == 7680 for table in app.dataframe)
    click(app, 'Next: explore spending →')
    click(app, 'Next: explore model insights →')
    assert any(x.label == 'Suggest categories' for x in app.button)
    click(app, 'Next: compare groups →')
    assert app.session_state['workspace'] == 'FairnessLens'
    assert any(x.label == 'Run disparity screening' for x in app.button)

def test_upload_empty_state_and_navigation():
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
    next(x for x in app.radio if x.label=='Data source').set_value('Upload transactions').run()
    assert not app.exception
    assert any('Choose a CSV' in x.value for x in app.info)
    assert any('city, channel' in x.value for x in app.markdown)
    next(x for x in app.radio if x.label=='Data source').set_value('Synthetic demo').run()
    click(app, 'Explore demo spending')
    assert app.session_state['workspace']=='Spending Analytics'
