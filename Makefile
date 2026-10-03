.PHONY: data train evaluate app test

data:
	python scripts/generate_data.py

train: data
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

app:
	streamlit run app.py

test:
	pytest -q

