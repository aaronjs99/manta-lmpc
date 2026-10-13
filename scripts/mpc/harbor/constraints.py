"""Ordered physical, safety, and terminal constraints for Harbor MPC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.mpc.harbor.symbolic import (
    _apply_symbolic_position_drift,
    _symbolic_position,
    _symbolic_step,
)

if TYPE_CHECKING:
    from scripts.configuration.harbor_mpc import HarborMPCConfig
    from scripts.contracts.harbor import HarborAgent


def dynamic_state_bounds(agent: HarborAgent) -> list[tuple[int, float, float]]:
    """Return nominal velocity and rate envelopes for one platform model."""
    model = agent.model
    if model.kind == "ugv":
        minimum_speed = (
            -model.max_reverse_speed
            if model.variant in {"kinematic_bicycle", "dynamic_skid_steer"}
            else 0.0
        )
        bounds = [(3, minimum_speed, model.max_speed)]
        if model.variant == "dynamic_skid_steer":
            bounds.append((4, -model.max_yaw_rate, model.max_yaw_rate))
        return bounds
    if model.kind == "usv":
        bounds = [(3, 0.0, model.max_speed)]
        if model.variant == "marine_3dof":
            bounds.extend(
                [
                    (4, -model.max_sway_speed, model.max_sway_speed),
                    (5, -model.max_yaw_rate, model.max_yaw_rate),
                ]
            )
        return bounds
    return [
        (6, -model.max_horizontal_speed, model.max_horizontal_speed),
        (7, -model.max_horizontal_speed, model.max_horizontal_speed),
        (8, -model.max_vertical_speed, model.max_vertical_speed),
        (9, -model.max_angular_rate, model.max_angular_rate),
        (10, -model.max_angular_rate, model.max_angular_rate),
        (11, -model.max_angular_rate, model.max_angular_rate),
    ]


class HarborConstraintBuilder:
    """Add constraints in the established NLP order."""

    def __init__(
        self,
        *,
        agent: HarborAgent,
        other_agents: tuple[HarborAgent, ...],
        config: HarborMPCConfig,
        dt: float,
        ca: Any,
    ) -> None:
        self.agent = agent
        self.other_agents = other_agents
        self.config = config
        self.dt = dt
        self.ca = ca

    def add_initial_and_envelope_constraints(
        self,
        *,
        opti: Any,
        states: Any,
        controls: Any,
        initial: Any,
        collision_slack: Any,
        domain_buffer_slack: Any,
        dynamic_state_slack: Any,
    ) -> None:
        opti.subject_to(states[:, 0] == initial)
        opti.subject_to(
            opti.bounded(0.0, collision_slack, self.config.collision_slack_bound)
        )
        opti.subject_to(
            opti.bounded(0.0, domain_buffer_slack, self.config.domain_boundary_margin)
        )
        opti.subject_to(
            opti.bounded(
                0.0, dynamic_state_slack, self.config.dynamic_state_slack_bound
            )
        )
        self._apply_bounds(
            opti, states, controls, domain_buffer_slack, dynamic_state_slack
        )

    def add_identification_probe_constraint(
        self, *, opti: Any, controls: Any, probe: Any, probe_mask: Any
    ) -> None:
        opti.subject_to(self.ca.times(probe_mask, controls[:, 0] - probe) == 0.0)

    def add_step_constraints(
        self,
        *,
        opti: Any,
        states: Any,
        controls: Any,
        position_drift: Any,
        control_effectiveness: Any,
        belief_collision_margin: Any,
        obstacle_parameters: list[Any],
        collision_slack: Any,
        index: int,
    ) -> None:
        ca = self.ca
        model = self.agent.model
        opti.subject_to(
            states[:, index + 1]
            == _apply_symbolic_position_drift(
                ca,
                model,
                _symbolic_step(
                    ca,
                    model,
                    states[:, index],
                    ca.times(control_effectiveness, controls[:, index]),
                    self.dt,
                ),
                position_drift,
                self.dt,
            )
        )
        own_position = _symbolic_position(ca, model, states[:, index + 1])
        for obstacle_index, other in enumerate(self.other_agents):
            relative = own_position - obstacle_parameters[obstacle_index][:, index]
            required = (
                self.agent.radius
                + other.radius
                + self.config.collision_buffer
                + belief_collision_margin
            )
            opti.subject_to(
                ca.dot(relative, relative) + collision_slack[obstacle_index, index]
                >= required * required
            )

    def add_terminal_constraints(
        self,
        *,
        opti: Any,
        safe_pose: Any,
        lambdas: Any,
        terminal_slack: Any,
        terminal_pose: Any,
        terminal_dim: int,
    ) -> None:
        opti.subject_to(opti.bounded(0.0, lambdas, 1.0))
        opti.subject_to(self.ca.sum1(lambdas) == 1.0)
        opti.subject_to(
            opti.bounded(
                -self.config.terminal_slack_bound,
                terminal_slack,
                self.config.terminal_slack_bound,
            )
        )
        safe_terminal = self.ca.mtimes(safe_pose.T, lambdas)
        opti.subject_to(terminal_pose[:terminal_dim] == safe_terminal + terminal_slack)

    def _apply_bounds(
        self,
        opti: Any,
        states: Any,
        controls: Any,
        domain_buffer_slack: Any,
        dynamic_state_slack: Any,
    ) -> None:
        model = self.agent.model
        domain = self.agent.domain
        margin = self.config.domain_boundary_margin
        predicted_states = states[:, 1:]
        domains = [domain.x_bounds, domain.y_bounds]
        if model.kind == "rov":
            domains.append(domain.z_bounds)
        for axis, bounds in enumerate(domains):
            values = predicted_states[axis, :]
            lower_slack = domain_buffer_slack[2 * axis, :]
            upper_slack = domain_buffer_slack[2 * axis + 1, :]
            opti.subject_to(opti.bounded(bounds[0], values, bounds[1]))
            opti.subject_to(values >= bounds[0] + margin - lower_slack)
            opti.subject_to(values <= bounds[1] - margin + upper_slack)
        for bound_index, (state_index, lower, upper) in enumerate(
            dynamic_state_bounds(self.agent)
        ):
            values = predicted_states[state_index, :]
            opti.subject_to(values >= lower - dynamic_state_slack[2 * bound_index, :])
            opti.subject_to(
                values <= upper + dynamic_state_slack[2 * bound_index + 1, :]
            )
        if model.kind == "ugv":
            if model.variant == "dynamic_skid_steer":
                opti.subject_to(
                    opti.bounded(
                        -model.max_side_force,
                        controls[0, :],
                        model.max_side_force,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -model.max_side_force,
                        controls[1, :],
                        model.max_side_force,
                    )
                )
            else:
                opti.subject_to(
                    opti.bounded(
                        -model.max_acceleration,
                        controls[0, :],
                        model.max_acceleration,
                    )
                )
                steering_bound = (
                    model.max_steering_angle
                    if model.variant == "kinematic_bicycle"
                    else model.max_yaw_rate
                )
                opti.subject_to(
                    opti.bounded(-steering_bound, controls[1, :], steering_bound)
                )
        elif model.kind == "usv":
            thrust_bound = (
                model.max_jet_thrust
                if model.variant == "marine_3dof"
                else model.max_thrust
            )
            opti.subject_to(opti.bounded(-thrust_bound, controls[0, :], thrust_bound))
            second_control_bound = (
                model.max_jet_thrust
                if model.variant == "marine_3dof"
                else model.max_yaw_rate
            )
            opti.subject_to(
                opti.bounded(
                    -second_control_bound, controls[1, :], second_control_bound
                )
            )
        else:
            for index, limit in enumerate(model.control_scale()):
                opti.subject_to(opti.bounded(-limit, controls[index, :], limit))
