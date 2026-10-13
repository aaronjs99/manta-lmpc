# Plotting

Rendering only: figures and animations consume already-computed rollout data.

Manta plotting modules remain directly in this package. `harbor/` separates
trajectory animations, learning, mismatch, actuator faults, generalization,
prediction, recovery, uncertainty, and localization figures. Shared colors and
drawing primitives live in `harbor/common.py`.
Numerical summaries belong in `scripts/metrics/`, not in plotting modules.

Current outputs:
- 2D trajectories with start/goal markers.
- Pairwise distance time series with safety threshold.
- APF/LMPC learning progression plot.
- Cost-by-iteration plot.
- Optional final-iteration GIF with trajectory and wing-motion views.

Plotting uses a noninteractive Matplotlib backend so runs work from the terminal.
Shared primitives keep agent colors, obstacle layers, goal tolerances, workspace
limits, and output-directory handling consistent across static plots and GIFs.
