# Algorithms

- `manta/`: APF control, safe sets, hyperplanes, priorities, recovery and safety.
- `harbor/`: coordinated guidance (`guidance_control.py`), adaptive estimation,
  prediction and admission policies. Platform model classes supply physical
  quantities and transitions; they do not select policy controls.
- `estimation/`: range EKF, fixed-lag SLAM and information-gain calculations.

These components do not parse commands, run studies, plot or write reports.
Solver/controller implementations live in `../mpc/`; execution loops live in
`../simulation/` and `../experiments/`.
