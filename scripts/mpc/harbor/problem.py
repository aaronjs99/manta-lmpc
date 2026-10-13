"""CasADi problem construction for the Harbor platform MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scripts.mpc.harbor.constraints import (
    HarborConstraintBuilder,
    dynamic_state_bounds,
)
from scripts.mpc.harbor.objective import pose_error_cost, tracking_cost

if TYPE_CHECKING:
    from typing import Any

    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.contracts.harbor import HarborAgent


@dataclass(frozen=True)
class HarborProblemVariables:
    """Symbolic variables and parameters in the order created by the NLP."""

    opti: Any
    states: Any
    controls: Any
    collision_slack: Any
    domain_buffer_slack: Any
    dynamic_state_slack: Any
    p_initial: Any
    p_goal: Any
    p_previous_control: Any
    p_position_drift: Any
    p_control_effectiveness: Any
    p_belief_collision_margin: Any
    p_identification_probe: Any
    p_identification_probe_mask: Any
    p_obstacles: list[Any]
    p_safe_pose: Any | None
    p_safe_costs: Any | None
    lambdas: Any | None
    terminal_slack: Any | None


class HarborProblemBuilder:
    """Build one platform's symbolic objective and constraints."""

    def __init__(
        self,
        *,
        agent: HarborAgent,
        other_agents: tuple[HarborAgent, ...],
        config: HarborMPCConfig,
        dt: float,
        learning: bool,
        ca: Any,
    ) -> None:
        self.agent = agent
        self.other_agents = other_agents
        self.config = config
        self.dt = dt
        self.learning = learning
        self.ca = ca

    def build(self) -> HarborProblemVariables:
        """Create the NLP, preserving historical variable/constraint ordering."""
        ca = self.ca
        model = self.agent.model
        cfg = self.config
        nx = model.state_dim
        nu = model.control_dim
        pose_dim = model.pose_dim
        horizon = cfg.prediction_horizon
        obstacle_count = len(self.other_agents)
        opti = ca.Opti()
        states = opti.variable(nx, horizon + 1)
        controls = opti.variable(nu, horizon)
        collision_slack = opti.variable(obstacle_count, horizon)
        domain_axes = 3 if model.kind == "rov" else 2
        domain_buffer_slack = opti.variable(2 * domain_axes, horizon)
        dynamic_bounds = dynamic_state_bounds(self.agent)
        dynamic_state_slack = opti.variable(2 * len(dynamic_bounds), horizon)
        p_initial = opti.parameter(nx, 1)
        p_goal = opti.parameter(pose_dim, 1)
        p_previous_control = opti.parameter(nu, 1)
        p_position_drift = opti.parameter(3, 1)
        p_control_effectiveness = opti.parameter(nu, 1)
        p_belief_collision_margin = opti.parameter()
        p_identification_probe = (
            opti.parameter(nu, 1) if cfg.active_identification else None
        )
        p_identification_probe_mask = (
            opti.parameter(nu, 1) if cfg.active_identification else None
        )
        p_obstacles = [opti.parameter(3, horizon) for _ in self.other_agents]

        constraints = HarborConstraintBuilder(
            agent=self.agent,
            other_agents=self.other_agents,
            config=cfg,
            dt=self.dt,
            ca=ca,
        )
        constraints.add_initial_and_envelope_constraints(
            opti=opti,
            states=states,
            controls=controls,
            initial=p_initial,
            collision_slack=collision_slack,
            domain_buffer_slack=domain_buffer_slack,
            dynamic_state_slack=dynamic_state_slack,
        )
        if cfg.active_identification:
            constraints.add_identification_probe_constraint(
                opti=opti,
                controls=controls,
                probe=p_identification_probe,
                probe_mask=p_identification_probe_mask,
            )

        cost = 0
        previous = p_previous_control
        control_scale = ca.DM(model.control_scale())
        for index in range(horizon):
            constraints.add_step_constraints(
                opti=opti,
                states=states,
                controls=controls,
                position_drift=p_position_drift,
                control_effectiveness=p_control_effectiveness,
                belief_collision_margin=p_belief_collision_margin,
                obstacle_parameters=p_obstacles,
                collision_slack=collision_slack,
                index=index,
            )
            cost += tracking_cost(ca, model, cfg, states[:, index + 1], p_goal)
            normalized_control = ca.rdivide(controls[:, index], control_scale)
            cost += cfg.control_weight * ca.dot(normalized_control, normalized_control)
            delta_control = ca.rdivide(controls[:, index] - previous, control_scale)
            cost += cfg.control_rate_weight * ca.dot(delta_control, delta_control)
            previous = controls[:, index]
        cost += cfg.collision_slack_weight * ca.sum1(ca.vec(collision_slack))
        cost += cfg.domain_boundary_slack_weight * ca.sumsqr(domain_buffer_slack)
        cost += cfg.dynamic_state_slack_weight * ca.sumsqr(dynamic_state_slack)
        terminal_pose = states[:pose_dim, horizon]
        p_safe_pose = p_safe_costs = lambdas = terminal_slack = None
        if self.learning:
            sample_count = cfg.terminal_samples
            terminal_dim = (
                (2 if model.pose_dim == 3 else 3)
                if cfg.terminal_position_only
                else pose_dim
            )
            p_safe_pose = opti.parameter(sample_count, terminal_dim)
            p_safe_costs = opti.parameter(sample_count, 1)
            lambdas = opti.variable(sample_count, 1)
            terminal_slack = opti.variable(terminal_dim, 1)
            constraints.add_terminal_constraints(
                opti=opti,
                safe_pose=p_safe_pose,
                lambdas=lambdas,
                terminal_slack=terminal_slack,
                terminal_pose=terminal_pose,
                terminal_dim=terminal_dim,
            )
            cost += cfg.terminal_slack_weight * ca.dot(terminal_slack, terminal_slack)
            cost += cfg.terminal_safe_cost_weight * ca.dot(p_safe_costs, lambdas)
            cost += cfg.terminal_goal_weight * pose_error_cost(
                ca, model, cfg, terminal_pose, p_goal
            )
        else:
            cost += cfg.terminal_goal_weight * pose_error_cost(
                ca, model, cfg, terminal_pose, p_goal
            )

        opti.minimize(cost)
        options = {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": cfg.ipopt_print_level,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.sb": "yes",
        }
        opti.solver("ipopt", options)
        return HarborProblemVariables(
            opti=opti,
            states=states,
            controls=controls,
            collision_slack=collision_slack,
            domain_buffer_slack=domain_buffer_slack,
            dynamic_state_slack=dynamic_state_slack,
            p_initial=p_initial,
            p_goal=p_goal,
            p_previous_control=p_previous_control,
            p_position_drift=p_position_drift,
            p_control_effectiveness=p_control_effectiveness,
            p_belief_collision_margin=p_belief_collision_margin,
            p_identification_probe=p_identification_probe,
            p_identification_probe_mask=p_identification_probe_mask,
            p_obstacles=p_obstacles,
            p_safe_pose=p_safe_pose,
            p_safe_costs=p_safe_costs,
            lambdas=lambdas,
            terminal_slack=terminal_slack,
        )
