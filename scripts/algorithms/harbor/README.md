# Harbor Algorithms

This package contains policy and estimation logic with no dependency on
experiment execution, simulation engines, plotting, CLI, or reporting.

- `observer.py`: `EffectivenessEstimator` owns actuator-effectiveness estimates
  and their histories; `HarborStateObserver` coordinates estimator updates.
- `recovery_state.py`: `EffectivenessRecoveryState` owns recovery episodes,
  arming, dwell, and recovery-offset telemetry.
- `residual_observer.py`: `ResidualEstimator` owns current/residual estimates,
  covariance, rejection counts, and histories.
- `probing.py`: `IdentificationProbe` owns excitation/information state and
  selects, accepts, or rejects constraint-aware probe pulses.
- `effectiveness.py`, `identification.py`, and `residual.py`: pure numerical
  helpers for the corresponding algorithms.
- `prediction.py`, `guidance.py`, and `guidance_control.py`: prediction and
  guidance policies independent of the physical model implementations.
- `admission.py`: safe-learning admission decisions over completed results.

Stateful components expose frozen telemetry snapshots. The contained arrays and
maps remain owned by their component; consumers must not mutate them. The MPC
controller coordinates the components and exposes their snapshots through its
composed `telemetry.estimator`, `telemetry.identification`, and
`telemetry.retry` fields. `mpc/harbor/retry.py` owns the ordered solve retry and
fallback policy.
