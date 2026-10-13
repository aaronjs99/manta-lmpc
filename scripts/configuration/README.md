# Configuration

Validated settings and YAML loaders for both model families. `paths.py` owns the
repository root and default manta/harbor YAML paths. `manta.py` and `harbor.py`
parse scenario settings; `harbor_faults.py`, the study-specific
`harbor_*_criteria.py` modules, and `harbor_observation.py` own the
corresponding study and sensing settings. `harbor_criteria_io.py` shares the
criteria YAML-section reader without combining their contracts.
`assembly/` constructs runtime models and agents. Dedicated modules hold MPC,
APF, sensor, localization and factor-graph settings. `study_labels.py` owns
comparator names so metrics never depend on a launcher just to obtain labels.

Edit runtime values in the existing `config/` YAML files. Keep loaders free of
experiment execution and output generation.
