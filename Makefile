.PHONY: data train train-merchant train-forecast evaluate app test

data:
	python scripts/generate_data.py

train: data
	python scripts/train.py

train-merchant:
	python -m model.training.merchant --task merchant --model-name microsoft/MiniLM-L12-H384-uncased --output-dir model/artifacts/spendlens_merchant_minilm

train-forecast:
	python -m model.training.chronos

evaluate:
	python scripts/evaluate.py

app:
	streamlit run app.py

test:
	pytest -q
