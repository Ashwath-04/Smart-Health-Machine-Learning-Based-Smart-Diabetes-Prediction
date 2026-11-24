# Convenience targets (on Windows without make, run the commands shown in README.md).
PY ?= python

.PHONY: install data train train-quick evaluate test api app mlflow docker-build docker-up notebooks clean

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

train:
	$(PY) scripts/train.py

train-quick:
	$(PY) scripts/train.py --quick

evaluate:
	$(PY) scripts/evaluate.py

test:
	$(PY) -m pytest -q

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker-build:
	docker build -t smart-health-diabetes:serve .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache **/__pycache__ catboost_info
