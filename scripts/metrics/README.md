# Metrics

Offline scoring and acceptance criteria for manta and harbor rollouts.

- `core.py`, `evaluation.py`, `optimizer.py`: trajectory, safety, and solver summaries.
- `manta/report.py`: stable metric fields and plot-ready arrays for one LMPC result.
- `manta/benchmark.py`: benchmark record construction.
- `harbor/`: fault tracking, localization error, paired comparisons, bootstrap
  summaries, and frozen confirmation gates, grouped by study.
- `harbor/time_varying_fault_records.py`: schedule-aware trial-record assembly,
  causal event scoring, and compact artifact records.
- `harbor/time_varying_fault.py`: matched-seed summaries and bootstrap
  comparisons for scheduled temporary faults.

Metrics consume data; they do not import study launchers, execute experiments,
render plots, or write output files. Study labels and thresholds belong in
configuration. Plotting and reporting consume these computed scores.

Metrics use the first two state dimensions as position, so they work with the
active 7-state manta state and other position-leading state representations.

Current metrics:
- total position-error cost proxy
- swept minimum pairwise and obstacle distances between saved states
- collision interval count, including crossings missed by endpoint-only checks
- time-to-goal
- control effort when controls are provided
- rollout completion and safe-set admission validation
- first goal-tolerance hit step per agent and learning iteration
- shared padding of ragged per-agent histories
