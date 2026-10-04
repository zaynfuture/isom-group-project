"""Export aggregate smoke evidence without weights, transactions or notebook output."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = {'scope': 'Local CPU smoke execution; not full training or Colab-hosted verification', 'publication': 'disabled', 'runs': {}}
for task, notebook in [('merchant', '01_merchant_experiment'), ('forecast', '02_forecast_experiment')]:
    check = json.loads((ROOT / 'outputs/notebook-checks' / f'{notebook}.json').read_text())
    assert check['passed'], f'{notebook} has not passed'
    candidates = sorted((ROOT / 'outputs/notebook-runs').glob(f'{task}-legacy-*/selected/evaluation.json'))
    evaluation_path = candidates[-1]
    selected = evaluation_path.parent
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation['smoke']
    report['runs'][task] = {
        'execution': check,
        'run_id': selected.parent.name,
        'evaluation': evaluation,
        'reload': json.loads((selected / 'reload_verification.json').read_text()),
        'environment': json.loads((selected.parent / 'environment.json').read_text()),
    }
destination = ROOT / 'docs/verification/notebook-smoke.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(destination)
