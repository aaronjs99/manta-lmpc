# Application Code

`run.py` is the only launcher in this directory. Other code lives in packages by
responsibility. `scripts` is a Python namespace package with no `__init__.py`.

## Run

These entry points use the same dispatcher and accept the same commands:

```sh
python run.py --help
python scripts/run.py --help
python -m scripts.run --help
python scripts/run.py baseline
python scripts/run.py harbor --help
```

Inside this directory, use `python run.py --help`. Default YAML files resolve
relative to the repository. Relative output paths retain their existing
meaning: relative to the invoking working directory.

`sanity` is a zero-control simulation diagnostic.

## Responsibilities

| Package | Owns |
| --- | --- |
| `cli/` | Lazy command registry, dispatch, argument parsing |
| `configuration/` | Validated settings, YAML loaders, paths, study labels |
| `algorithms/` | APF, safe sets, adaptive estimation, EKF/SLAM, guidance |
| `dynamics/` | Numerical and symbolic plant models |
| `mpc/` | CasADi/IPOPT optimizer and controller implementations |
| `communication/` | Seeded message delivery, delay, dropout, TTL |
| `simulation/` | Plant stepping, accepted transition commits, synthetic sensing |
| `experiments/` | Study orchestration, learning-loop and matched-trial execution |
| `metrics/` | Scoring, paired statistics, confirmation gates |
| `plotting/` | Figures and animations from computed data |
| `reporting/` | Artifact serialization and presentation |
| `storage/` | Versioned, atomic Manta checkpoint persistence |

Algorithms do not depend on launchers, experiment execution, plotting, or
reporting. Metrics do not import experiment launchers. Plotting does not execute
studies. Shared helper implementations have a single owner.

## Adding a Study

1. Put orchestration in `experiments/manta/` or `experiments/harbor/`.
2. Put option parsing in the corresponding `cli/arguments/` package; accept `argv`.
3. Reuse algorithms, models, and execution primitives from their owning packages.
4. Keep scoring in `metrics/`, rendering in `plotting/`, and serialization in
   `reporting/`.
5. Register the command in `cli/registry.py`; do not add a top-level launcher.
6. Keep runtime settings in `config/` YAML. Preserve development, holdout, and
   confirmation seeds and thresholds.

See [the architecture guide](../docs/code_architecture.md) for the migration map.
Successful simulation does not establish physical readiness.
