# Dynamics

Physical state transitions, separate from controller policies and experiments.

- `manta.py`: shared 7-state manta/CPG dynamics and numerical/symbolic RK4.
- `harbor/base.py`: platform interface and angle wrapping.
- `harbor/ground.py`: bicycle and dynamic skid-steer UGVs.
- `harbor/surface.py`: 3-DOF USV dynamics.
- `harbor/underwater.py`: 6-DOF ROV dynamics.
- `harbor/reduced.py`: retained reduced-model alternatives.
- `harbor/mechanics.py`, `harbor/symbolic.py`: shared marine mechanics.
- `harbor/factory.py`: named model construction.

The same constants and RK4 integration logic are used by APF initialization,
zero-control simulation, and CasADi LMPC optimization. Numeric constants should
be edited in `config/manta.yaml`, then loaded into `MantaDynamicsConfig` from
`scripts/configuration/manta_dynamics.py`. Harbor parameters remain in the
existing harbor YAML files; model fidelity is documented in `docs/harbor_dynamics.md`.
