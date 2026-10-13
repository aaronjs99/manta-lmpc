"""Available commands and their lazily imported experiment implementations."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Command:
    """An experiment module, its help summary, and enforced preset arguments."""

    module: str
    description: str
    preset: tuple[str, ...] = ()


COMMANDS = MappingProxyType(
    {
        "manta": Command(
            "scripts.experiments.manta.lmpc", "Manta APF/LMPC workflow (default)."
        ),
        "lmpc": Command("scripts.experiments.manta.lmpc", "Alias for manta."),
        "baseline": Command(
            "scripts.experiments.manta.lmpc",
            "Manta APF baseline without LMPC iterations.",
            ("--iterations", "0"),
        ),
        "sanity": Command(
            "scripts.experiments.manta.sanity",
            "Zero-control manta simulation diagnostics.",
        ),
        "sweep": Command(
            "scripts.experiments.manta.sweep", "Compact manta scenario benchmark."
        ),
        "harbor": Command(
            "scripts.experiments.harbor.communication", "Harbor communication ablation."
        ),
        "harbor-sweep": Command(
            "scripts.experiments.harbor.sweep", "Seeded network robustness sweep."
        ),
        "harbor-lmpc": Command(
            "scripts.experiments.harbor.lmpc", "Distributed harbor MPC/LMPC."
        ),
        "harbor-horizon-study": Command(
            "scripts.experiments.harbor.horizon_study",
            "Matched-horizon efficiency study.",
        ),
        "harbor-robustness": Command(
            "scripts.experiments.harbor.robustness",
            "Current and actuator mismatch study.",
        ),
        "harbor-fault-study": Command(
            "scripts.experiments.harbor.fault_study",
            "Per-channel actuator fault study.",
        ),
        "harbor-fault-generalization": Command(
            "scripts.experiments.harbor.fault_generalization",
            "Stratified hidden actuator faults.",
        ),
        "harbor-fault-noise-study": Command(
            "scripts.experiments.harbor.fault_generalization",
            "Hidden actuator faults with observation noise.",
            ("--with-observation-noise",),
        ),
        "harbor-prediction-study": Command(
            "scripts.experiments.harbor.prediction_study",
            "Peer-motion prediction ablation.",
        ),
        "harbor-time-varying-fault-study": Command(
            "scripts.experiments.harbor.time_varying_fault_study",
            "Scheduled actuator loss and recovery.",
        ),
        "harbor-temporary-fault-generalization": Command(
            "scripts.experiments.harbor.temporary_fault_generalization",
            "Temporary-fault development and confirmation.",
        ),
        "harbor-joint-uncertainty-study": Command(
            "scripts.experiments.harbor.joint_uncertainty_study",
            "Joint hidden-current and actuator uncertainty.",
        ),
        "harbor-projected-residual-study": Command(
            "scripts.experiments.harbor.projected_residual_study",
            "Controllability-projected current adaptation.",
        ),
        "harbor-dynamic-envelope-study": Command(
            "scripts.experiments.harbor.dynamic_envelope_study",
            "Dynamic-state envelope ablation.",
        ),
        "harbor-station-keeping-study": Command(
            "scripts.experiments.harbor.station_keeping_study",
            "Marine current estimation and station keeping.",
        ),
        "harbor-localization-study": Command(
            "scripts.experiments.harbor.localization_study",
            "Dead reckoning, range localization, and SLAM.",
        ),
        "harbor-joint-localization-study": Command(
            "scripts.experiments.harbor.joint_localization_study",
            "Estimated-state MPC under joint uncertainty.",
        ),
    }
)
