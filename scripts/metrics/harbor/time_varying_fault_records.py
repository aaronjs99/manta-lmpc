"""Record assembly and event scoring for scheduled temporary-fault trials."""

from __future__ import annotations

import numpy as np


def _score_inflation_events(
    scheduled_steps: dict[str, list[int]],
    triggered_steps: dict[str, list[int]],
    window_steps: int,
) -> tuple[float, int, float | None]:
    """Score causal event recall, unmatched inflations, and detection delay."""
    event_count = 0
    matched_count = 0
    false_count = 0
    delays = []
    for name, events in scheduled_steps.items():
        triggers = list(triggered_steps[name])
        used = set()
        event_count += len(events)
        for event in events:
            match = next(
                (
                    (index, trigger)
                    for index, trigger in enumerate(triggers)
                    if index not in used and event <= trigger <= event + window_steps
                ),
                None,
            )
            if match is not None:
                index, trigger = match
                used.add(index)
                matched_count += 1
                delays.append(trigger - event)
        false_count += len(triggers) - len(used)
    recall = matched_count / event_count if event_count else 1.0
    return recall, false_count, float(np.mean(delays)) if delays else None


def build_time_varying_fault_records(
    cases, agents, simulation, event_detection_window_steps: int
) -> list[dict]:
    """Convert matched temporary-fault trials into schedule-aware records."""
    records = []
    for case in cases:
        fault_event_steps = {
            name: [event[0] for event in events]
            for name, events in case.disturbance.agent_control_effectiveness_schedule.items()
            if events
        }
        if set(fault_event_steps) != {agent.name for agent in agents} or any(
            len(events) != 2 for events in fault_event_steps.values()
        ):
            raise ValueError(
                "temporary fault evaluation requires one onset and recovery per agent"
            )
        hidden_schedule = {
            name: [
                {"step": int(step), "effectiveness": list(value)}
                for step, value in events
            ]
            for name, events in (
                case.disturbance.agent_control_effectiveness_schedule.items()
            )
        }
        marine_agents = [agent for agent in agents if agent.model.kind != "ugv"]
        hidden_water_current = {
            agent.name: case.disturbance.current(agent.model).tolist()
            for agent in marine_agents
        }
        for trial in case.trials:
            count = min(
                len(trial.effectiveness_history[agent.name]) for agent in agents
            )
            global_history = []
            platform_histories = {agent.name: [] for agent in agents}
            current_history = []
            control_current_history = []
            platform_current_histories = {agent.name: [] for agent in marine_agents}
            for step in range(count):
                errors = []
                for agent in agents:
                    estimate = trial.effectiveness_history[agent.name][step]
                    truth = trial.result.applied_effectiveness[agent.name][step]
                    platform_error = float(np.sqrt(np.mean((estimate - truth) ** 2)))
                    platform_histories[agent.name].append(platform_error)
                    errors.extend((estimate - truth).tolist())
                global_history.append(float(np.sqrt(np.mean(np.square(errors)))))
                current_errors = []
                for agent in marine_agents:
                    estimate = trial.residual_history[agent.name][step]
                    truth = case.disturbance.current(agent.model)
                    error = estimate - truth
                    current_errors.extend(error.tolist())
                    platform_current_histories[agent.name].append(
                        float(np.sqrt(np.mean(error * error)))
                    )
                current_history.append(
                    float(np.sqrt(np.mean(np.square(current_errors))))
                )
                control_current_errors = []
                for agent in marine_agents:
                    estimate = trial.control_residual_history[agent.name][step]
                    truth = case.disturbance.current(agent.model)
                    control_current_errors.extend((estimate - truth).tolist())
                control_current_history.append(
                    float(np.sqrt(np.mean(np.square(control_current_errors))))
                )
            platform_fault_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][
                            fault_event_steps[agent.name][0] : fault_event_steps[
                                agent.name
                            ][1]
                        ]
                    )
                )
                for agent in agents
            }
            platform_recovery_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][
                            fault_event_steps[agent.name][1] :
                        ]
                    )
                )
                for agent in agents
            }
            event_recall, false_inflations, mean_detection_delay = (
                _score_inflation_events(
                    fault_event_steps,
                    trial.effectiveness_change_steps_by_agent,
                    event_detection_window_steps,
                )
            )
            sustained_cost = sum(
                (
                    trial.result.first_goal_steps[agent.name]
                    if trial.result.first_goal_steps[agent.name] is not None
                    and trial.result.final_goal_errors[agent.name]
                    <= simulation.goal_tolerance
                    and trial.result.final_orientation_errors[agent.name]
                    <= simulation.orientation_tolerance
                    else simulation.horizon + 1
                )
                for agent in agents
            )
            records.append(
                {
                    "seed": case.seed,
                    "observation_seed": case.observation_seed,
                    "controller": trial.label,
                    "fault_event_steps": fault_event_steps,
                    "hidden_fault_schedule": hidden_schedule,
                    "hidden_water_current": hidden_water_current,
                    "tracking_rmse_history": global_history,
                    "current_rmse_history": current_history,
                    "control_current_rmse_history": control_current_history,
                    "residual_estimate_history": {
                        agent.name: trial.residual_history[agent.name][:count].tolist()
                        for agent in marine_agents
                    },
                    "control_residual_history": {
                        agent.name: trial.control_residual_history[agent.name][
                            :count
                        ].tolist()
                        for agent in marine_agents
                    },
                    "residual_projection_retry_count_by_agent": (
                        trial.residual_projection_retry_count_by_agent
                    ),
                    "effectiveness_estimate_history": {
                        agent.name: trial.effectiveness_history[agent.name][
                            :count
                        ].tolist()
                        for agent in agents
                    },
                    "raw_effectiveness_estimate_history": {
                        agent.name: trial.raw_effectiveness_history[agent.name][
                            :count
                        ].tolist()
                        for agent in agents
                    },
                    "applied_effectiveness_history": {
                        agent.name: trial.result.applied_effectiveness[agent.name][
                            :count
                        ].tolist()
                        for agent in agents
                    },
                    "information_std_history": {
                        agent.name: trial.information_std_history[agent.name][
                            :count
                        ].tolist()
                        for agent in agents
                    },
                    "excitation_history": {
                        agent.name: trial.excitation_history[agent.name][
                            :count
                        ].tolist()
                        for agent in agents
                    },
                    "platform_fault_rmse": platform_fault_rmse,
                    "platform_recovery_rmse": platform_recovery_rmse,
                    "fault_interval_rmse": float(
                        np.mean(list(platform_fault_rmse.values()))
                    ),
                    "recovery_rmse": float(
                        np.mean(list(platform_recovery_rmse.values()))
                    ),
                    "final_effectiveness_rmse": global_history[-1],
                    "current_rmse": float(np.mean(current_history)),
                    "final_current_rmse": current_history[-1],
                    "platform_current_rmse": {
                        name: float(np.mean(values))
                        for name, values in platform_current_histories.items()
                    },
                    "platform_final_current_rmse": {
                        name: values[-1]
                        for name, values in platform_current_histories.items()
                    },
                    "control_current_rmse": float(np.mean(control_current_history)),
                    "final_control_current_rmse": control_current_history[-1],
                    "change_steps_by_agent": trial.effectiveness_change_steps_by_agent,
                    "recovery_offset_steps_by_agent": (
                        trial.recovery_offset_steps_by_agent
                    ),
                    "recovery_offset_rejections_by_agent": (
                        trial.recovery_offset_rejections_by_agent
                    ),
                    "recovery_offset_unarmed_rejections_by_agent": (
                        trial.recovery_offset_unarmed_rejections_by_agent
                    ),
                    "recovery_offset_episode_limit_rejections_by_agent": (
                        trial.recovery_offset_episode_limit_rejections_by_agent
                    ),
                    "recovery_offset_dwell_rejections_by_agent": (
                        trial.recovery_offset_dwell_rejections_by_agent
                    ),
                    "valid": trial.valid,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "first_goal_steps": trial.result.first_goal_steps,
                    "final_goal_errors": trial.result.final_goal_errors,
                    "final_orientation_errors": (trial.result.final_orientation_errors),
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "max_collision_slack": trial.max_collision_slack,
                    "max_domain_buffer_slack": trial.max_domain_buffer_slack,
                    "max_dynamic_state_slack": trial.max_dynamic_state_slack,
                    "dynamic_state_retry_count_by_agent": (
                        trial.dynamic_state_retry_count_by_agent
                    ),
                    "residual_rejection_count_by_agent": (
                        trial.residual_rejection_count_by_agent
                    ),
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_steps_by_agent": (
                        trial.solver_failure_steps_by_agent
                    ),
                    "solver_failure_status_counts": (
                        trial.solver_failure_status_counts
                    ),
                    "sustained_completion_cost": sustained_cost,
                    "control_total_variation": {
                        agent.name: float(
                            np.sum(
                                np.abs(
                                    np.diff(trial.result.controls[agent.name], axis=0)
                                )
                            )
                        )
                        for agent in agents
                    },
                    "event_recall": event_recall,
                    "false_inflations": false_inflations,
                    "mean_detection_delay": mean_detection_delay,
                    "probe_count": sum(trial.probe_count_by_agent.values()),
                    "probe_count_by_agent": trial.probe_count_by_agent,
                }
            )
    return records


def compact_time_varying_fault_records(records: list[dict]) -> list[dict]:
    """Drop regenerable step histories from curated research artifacts."""
    return [
        {key: value for key, value in record.items() if not key.endswith("_history")}
        for record in records
    ]
