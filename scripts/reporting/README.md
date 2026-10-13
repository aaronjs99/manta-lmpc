# Reporting

`json.py` owns shared JSON artifact serialization. Experiments pass completed
payloads to reporting instead of opening result files themselves.

`manta.py` writes prepared LMPC artifacts and delegates visualization:

- `metrics/manta/report.py` computes summary fields and plot-ready arrays.
- `save_manta_run_report` serializes that prepared report and delegates rendering
  to plotting functions; it preserves the existing JSON/CSV/PNG/GIF names.

Command-line entry points should execute runs and delegate artifact generation
here instead of duplicating metrics or serialization logic.
