# Task Planner scripts

Python code is grouped by responsibility:

- `core/`: road-network, domain-model and graph primitives.
- `algorithms/`: ALNS and the metaheuristic schedulers.
- `runtime/`: real-time scheduling, execution and event-driven reallocation.
- `experiments/`: benchmark and comparison programs.
- `visualization/`: plot, animation and interactive-output generators.
- `tests/`: executable validation scenarios.
- `apps/`: small command-line demos.

Run source modules from the workspace root with `PYTHONPATH=src/task_planner`,
for example:

```bash
PYTHONPATH=src/task_planner python3 -m scripts.experiments.compare_all
PYTHONPATH=src/task_planner python3 -m scripts.visualization.gen_roadmap_png
```

The `../启动` launcher uses this form.  The protected packages
`ais_navigation` and `waterway_map` were intentionally left untouched.

## Recommended integrated workflow

`python3 -m scripts.apps.run_unified_planner` runs the unified domain model:
static Graph+ALNS allocation, versioned pickup/delivery plans, full road-node
sequence expansion, time-based execution, and event-triggered rolling
replanning.  See `../docs/UNIFIED_ARCHITECTURE.md`.
