up:
	docker compose up --build lab
report:
	docker compose up report
local-fast:
	PYTHONPATH=. python -m src.run_all --fast
local-full:
	PYTHONPATH=. python -m src.run_all
