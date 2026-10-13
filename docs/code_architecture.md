# Code Architecture

Code is organized by responsibility, with manta and harbor components inside
each layer. Only `README.md` and `run.py` are files at the `scripts/` top level.

## Entry Points

Root `run.py` and `scripts/run.py` call `scripts.cli.dispatcher.main(argv)`.
The declarative command registry is lazy: global help does not import NumPy,
CasADi, or Matplotlib. Each experiment receives an explicit argument list;
dispatch does not modify `sys.argv`. No command starts a physical robot.

`manta` is the default; `lmpc` remains an alias. `baseline` enforces zero LMPC
iterations; `harbor-fault-noise-study` enforces observation noise, as before.
Unknown commands report a usage error. Individual `scripts/run_*.py` launchers
have been consolidated into the command dispatcher.

## Ownership

- **Contracts:** scenario/domain descriptions, agent specs, observations,
  measurements, rollout records, and controller-facing protocols. Contracts do
  not import concrete controllers, simulations, experiments, plotting, or reports.
- **Configuration:** YAML parsing, validated settings, profile resolution, paths,
  and study criteria. Harbor configuration returns model-independent specs.
- **Assembly:** converts validated settings and specs into dynamics models,
  agents, controllers, and simulation dependencies.
- **Dynamics:** physical equations, numerical/symbolic transitions, and factories;
  guidance policies live in `algorithms/harbor/guidance_control.py`.
- **Algorithms:** policies, adaptive estimation, recovery detection, residual
  estimation, identification probing, prediction, EKF/SLAM, information gain,
  safe sets, and admission. Stateful algorithms own their estimates, counters,
  and histories. No CLI, experiment, simulation, plotting, or reporting imports.
- **MPC:** optimizer construction, solver calls, and receding-horizon
  orchestration. `HarborSolveRetryPolicy` owns ordered retry/fallback behavior;
  `DistributedHarborMPC.telemetry` composes estimator, identification, and retry
  snapshots without duplicating their mutable state.
- **Communication:** synthetic message transmission and delivery.
- **Simulation:** plant stepping, sensing, message delivery, estimator interfaces,
  and trajectory recording. Plant truth remains separate from observations;
  experiment learning-loop coordination lives under `experiments/`.
- **Experiments:** development/holdout/confirmation orchestration and matched trials.
- **Metrics:** scoring, paired statistics and acceptance criteria; no experiment
  launcher dependencies, plotting or file output.
- **Plotting:** figures and animations from completed results.
- **Reporting:** JSON/CSV serialization and human-readable artifact assembly.

Type-only dependencies use postponed annotations and `TYPE_CHECKING`. Existing
configuration values, random seeds, equations and confirmation gates are retained.

## Migration Map

| Former location | Current owner |
| --- | --- |
| `scripts/config.py` | `configuration/manta.py`; shared defaults in `configuration/paths.py` |
| `scripts/run_*.py` | `cli/arguments/` + `experiments/`; scoring in `metrics/` |
| `scripts/learning/` | `algorithms/manta/`; plant stepping in `simulation/`, learning orchestration in `experiments/manta/` |
| `scripts/harbor/models.py` | `dynamics/harbor/`, split by model and shared mechanics |
| `scripts/harbor/mpc.py` | `mpc/harbor/`, `algorithms/harbor/`, `configuration/harbor_mpc.py` |
| `scripts/harbor/simulation.py` | `simulation/harbor/`, shared records in `contracts/`, and analysis in `metrics/harbor/` |
| `scripts/harbor/localization.py` | `algorithms/estimation/ekf.py`, `simulation/harbor/`, `configuration/localization.py` |
| `scripts/harbor/factor_graph.py` | `algorithms/estimation/factor_graph.py`; settings in `configuration/factor_graph.py` |
| `scripts/harbor/information.py` | `algorithms/estimation/information.py` |
| `scripts/harbor/experiments.py` | `experiments/harbor/execution/` |
| `scripts/harbor/plotting.py` | `plotting/harbor/`; numeric helpers in `metrics/harbor/estimation.py` |

Harbor plot families have separate modules for mismatch, actuator faults,
generalization, and prediction. The Harbor NLP coordinator declares variables
and preserves the historical constraint-row order while objective terms,
constraints, numerical execution, and failure diagnostics have distinct owners.
Internal imports use current owners; obsolete import shims are not retained.

Harbor learning state is divided among `EffectivenessEstimator`,
`EffectivenessRecoveryState`, `ResidualEstimator`, and `IdentificationProbe`.
The controller coordinates these components in the established update order;
trial assembly consumes their explicit telemetry snapshots. Solver retries are
separate from controller orchestration and preserve probe removal, dynamic
envelope retry, residual projection retry, orientation-triggered comparison,
and guidance fallback order.

Harbor trial execution uses immutable policy specifications and prepared
scenario records. Single-trial execution and result assembly are separate from
study orchestration. Metrics record construction is separate from aggregate
statistics and reporting.

Manta safe-set creation is separated into candidate scheduling, compact
delay/replay, and score/rank/select modules. `HarborProblemBuilder` coordinates
the separate objective and constraint components without changing symbolic
variable, objective, or constraint ordering.
Manta `runner.py` coordinates safe-set validation, learning admission, and
history updates; `iteration.py` executes one trial iteration. Accepted optimizer
and safety-filter state transitions are committed through the simulation
environment without reintegrating them.
Root command names and options remain supported. Default YAML
paths remain repository-relative; relative output paths remain relative to the
invoking working directory.

## Verification Boundary

Manta run persistence lives in `scripts/storage/`; the learning runner exposes
completed-iteration callbacks and accepts retained run results without doing
filesystem I/O. `metrics/manta/learning.py` validates archive shapes and
recomputes admission from trajectories. CLI orchestration owns output-directory
reservation, stop signals, archived-setting compatibility, and progress reports.
See [recoverable runs](manta_runs.md) for the supported workflow.

A structural refactor is not a new research result. Compare syntax, imports,
command options, representative numerical results and artifact generation without
overwriting `results/latest/`. Temporary verification material stays outside the
repository. Successful simulation does not establish physical readiness.
