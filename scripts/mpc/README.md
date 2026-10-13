# MPC

Controller builders for the project.

- `manta_lmpc.py`: active CasADi/IPOPT LMPC agent for the 7-state manta dynamics.
- `harbor/optimizer.py`: stable per-agent solve facade and status owner.
- `harbor/problem.py`: ordered CasADi variable declaration and NLP orchestration.
- `harbor/objective.py`: pure tracking and terminal objective terms.
- `harbor/constraints.py`: state, input, safety, dynamics, and terminal constraints.
- `harbor/solver.py`: numeric parameterization, IPOPT execution, and result mapping.
- `harbor/diagnostics.py`: best-effort residual diagnostics after solver failure.
- `harbor/result.py`: immutable result contract shared by the Harbor facade/solver.
- `harbor/controller.py`: distributed receding-horizon orchestration, composing
  guidance, observations, probes, solver retries, and telemetry.
- `harbor/retry.py`: ordered probe-removal, elastic, residual-projection, and
  fallback policy, including failure counters and diagnostics.
- `harbor/symbolic.py`: platform-independent symbolic optimization expressions.

Configuration dataclasses live in `scripts/configuration/`. Adaptation,
prediction, and admission algorithms live in `scripts/algorithms/`, not in the
solver implementation. Rollout execution lives in `scripts/simulation/`.
The Harbor optimizer facade keeps its existing `solve(...)` call interface;
the problem builder preserves variable declaration and call ordering while
delegating objective terms, constraint construction, numeric execution, and
failure diagnosis to their own modules. The numeric solver owns parameter
updates and execution, and diagnostics are collected only after a solver
`RuntimeError` before that original error is re-raised.

Harbor estimator, recovery, residual, and identification state is owned by
`scripts/algorithms/harbor/` components. The controller exposes a composed
`telemetry` snapshot with `estimator`, `identification`, and `retry` sections;
experiment and metric consumers read from that snapshot rather than from
controller-internal state.

LMPC horizons, costs, slack bounds, SVM distance thresholds, warm starts, solver
limits, and tolerances belong in `config/manta.yaml`.

Control warm starts are provided by the learning runner, which blends nominal
controls with stored controls from the current safe-set trajectory before each
IPOPT solve.

The default terminal constraint matches the learned convex-hull state in
position only. Full 7-state terminal matching can be enabled with
`terminal_position_only: false`, but it is much easier to make infeasible for
the manta oscillator states.
