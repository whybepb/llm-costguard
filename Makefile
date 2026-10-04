PY ?= ./.venv/bin/python
PORT ?= 8000
export PYTHONPATH := $(CURDIR)

.PHONY: test serve serve-mlx dashboard trace ab sweep gate loadtest ci-gate report

test:            ## unit + integration tests (mock backend, no keys)
	$(PY) -m pytest -q

serve:           ## OpenAI-compatible proxy on :$(PORT) (mock backend unless COSTGUARD_BACKEND is set)
	COSTGUARD_BACKEND=$${COSTGUARD_BACKEND:-mock} $(PY) -m uvicorn costguard.server:get_app --factory --port $(PORT)

serve-mlx:       ## same, with real local models (Apple silicon)
	COSTGUARD_BACKEND=mlx $(PY) -m uvicorn costguard.server:get_app --factory --port $(PORT)

dashboard:       ## savings dashboard
	$(PY) -m streamlit run dashboard/app.py

trace:           ## build the frozen replay trace
	$(PY) -m eval.build_trace

sweep:           ## semantic-cache threshold sweep (hit rate vs false-hit rate)
	$(PY) -m eval.sweep_threshold

gate:            ## eval gate for model downshift -> configs/router_gate.json
	$(PY) -m eval.gate_router

ab:              ## cumulative ablation A/B on the frozen trace (uses cassettes)
	$(PY) -m eval.run_ab

ci-gate:         ## CI eval gate (replay mode, no keys)
	$(PY) -m eval.ci_gate

loadtest:        ## proxy overhead + throughput against the mock upstream
	bash loadtest/run.sh

report:          ## regenerate docs/RESULTS.md (README tables are pasted from it) from eval/results/*.json
	$(PY) -m eval.report
