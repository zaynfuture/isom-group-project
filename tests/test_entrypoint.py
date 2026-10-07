"""The launcher is inert on import and exposes an explicit main function."""
import importlib.util
from pathlib import Path
from unittest.mock import patch


def test_main_is_explicit_and_import_has_no_launch_side_effect():
    entry = Path(__file__).resolve().parents[1] / 'app.py'
    spec = importlib.util.spec_from_file_location('spendlens_entrypoint', entry)
    module = importlib.util.module_from_spec(spec)
    with patch('runpy.run_path') as launch:
        spec.loader.exec_module(module)
        launch.assert_not_called()
        module.main()
        launch.assert_called_once_with(str(entry.parent / 'app/main.py'), run_name='__main__')
