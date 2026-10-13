# Simulation

Execution layer for manta and heterogeneous harbor workflows.

- `environment.py`: active variable-agent manta environment, accepted
  optimizer/safety-filter transition commits, and fixed-three compatibility wrapper.
- `scenarios.py`: scenario and obstacle dataclasses plus YAML-backed lookup helpers.
- `manta_autopilot.py`: APF rollout execution using the reusable APF algorithm.
- `manta_learning.py`: repeated manta LMPC execution and safe-set updates.
- `harbor/types.py`: platform, rollout, and controller interface contracts.
- `harbor/engine.py`: plant stepping, disturbances, and communication integration.
- `harbor/sensors.py`: seeded synthetic range sensing.
- `harbor/localization.py`, `harbor/localization_modes.py`: estimator integration.
- `harbor/learning.py`: admitted harbor MPC/LMPC rollout execution.

Policies and estimators belong in `scripts/algorithms/`, plant equations in
`scripts/dynamics/`, configuration in `scripts/configuration/`, and offline
scores in `scripts/metrics/`. This layer does not parse CLI arguments or plot.
Manta learning-loop coordination and per-iteration policy execution are in
`scripts/experiments/manta/`; accepted next states cross into the environment
after optimization and safety filtering without being reintegrated.

Scenario starts, goals, obstacles, and safety distances live in `config/manta.yaml`.
