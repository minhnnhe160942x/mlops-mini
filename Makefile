.PHONY: up down logs test lint trigger predict reload

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

test:
	python -m pytest -q

lint:
	python -m ruff check .

# Trigger a run that simulates drifted input, so the gate retrains.
trigger:
	docker compose exec airflow-scheduler \
	  airflow dags trigger training_pipeline --conf '{"shift": 1.5, "batch_rows": 200}'

reload:
	curl -fsS -X POST http://localhost:8000/reload

predict:
	curl -fsS http://localhost:8000/model
