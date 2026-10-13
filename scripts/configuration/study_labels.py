"""Configuration / study labels."""

from __future__ import annotations

HARD_ENVELOPE_BASELINE = "Hard-envelope transient-offset RLS"
RETRY_ELASTIC_CANDIDATE = "Retry-elastic transient-offset RLS"
KINEMATIC_EWMA = "Kinematic-current EWMA transient-offset RLS"
KINEMATIC_RLS = "Kinematic-current RLS transient-offset RLS"
USV_NAME = "surface_vessel"


POLICIES = (
    "Direct position sensor",
    "Dead reckoning",
    "Known harbor map",
    "Joint landmark SLAM",
    "Robust fixed-lag SLAM",
    "Joint SLAM + belief retry",
)


CONTROLLER_LABELS = (
    "Fixed-covariance RLS",
    "Innovation-threshold RLS",
    "Recovery-prior threshold RLS",
    "Transient-offset threshold RLS",
    "Chi-square CUSUM RLS",
    "CUSUM-triggered probing RLS",
)
