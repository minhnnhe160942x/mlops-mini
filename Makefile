.PHONY: up down logs ps test lint run corrupt repair predict health

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

test:
	python -m pytest -q

lint:
	python -m ruff check .

# Run the whole pipeline for one logical date, in the foreground.
run:
	docker compose exec airflow-scheduler airflow dags test wdbc_pipeline 2026-08-25

# Break 12% of the extract, then run again to watch validate refuse it.
corrupt:
	docker compose exec airflow-scheduler python /opt/airflow/scripts/corrupt_extract.py

repair:
	docker compose exec airflow-scheduler python /opt/airflow/scripts/corrupt_extract.py --repair

health:
	curl -s http://localhost:18011/health

predict:
	docker compose exec airflow-scheduler python /opt/airflow/scripts/sample_request.py > sample_request.json
	curl -s -X POST http://localhost:18011/predict \
	  -H 'content-type: application/json' -d @sample_request.json
