"""DAgger collection and retraining with explicit teacher interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from interrupt_nav.env import ContinuousNavEnv, EnvConfig
from interrupt_nav.model import NavigationPolicy, set_deterministic_seed
from interrupt_nav.oracle import AStarTeacher
from interrupt_nav.schema import Action, Episode, StepRecord, Trajectory, write_jsonl
from interrupt_nav.training import TrainConfig, train_behavior_cloning


@dataclass(slots=True)
class DAggerConfig:
    iterations: int = 3
    episodes_per_iteration: int = 4
    beta_start: float = 1.0
    beta_end: float = 0.1
    linear_intervention_threshold: float = 0.45
    angular_intervention_threshold: float = 1.0
    train_epochs: int = 3
    seed: int = 0

    def beta(self, iteration: int) -> float:
        if self.iterations <= 1:
            return self.beta_end
        ratio = iteration / (self.iterations - 1)
        return self.beta_start + ratio * (self.beta_end - self.beta_start)


def _requires_intervention(
    policy_action: Action, teacher_action: Action, config: DAggerConfig
) -> bool:
    return (
        abs(policy_action.linear - teacher_action.linear) > config.linear_intervention_threshold
        or abs(policy_action.angular - teacher_action.angular)
        > config.angular_intervention_threshold
        or policy_action.stop != teacher_action.stop
        or policy_action.ask
    )


def collect_dagger_trajectory(
    policy: NavigationPolicy,
    env: ContinuousNavEnv,
    episode: Episode,
    *,
    beta: float,
    rng: np.random.Generator,
    config: DAggerConfig,
    teacher: AStarTeacher | None = None,
) -> Trajectory:
    """Roll out the policy while recording teacher labels and takeover events."""
    teacher = teacher or AStarTeacher()
    observation, _ = env.reset(seed=episode.seed, episode=episode)
    corrections_by_step = {item.step: item for item in episode.corrections}
    hidden = None
    records: list[StepRecord] = []
    while True:
        correction = corrections_by_step.get(env.step_count)
        if correction is not None:
            env.apply_correction(correction)
            observation = env.observe()
        policy_action, hidden = policy.act(observation, hidden)
        teacher_action = teacher.act(env)
        intervention = bool(
            rng.random() < beta or _requires_intervention(policy_action, teacher_action, config)
        )
        executed = teacher_action if intervention else policy_action
        next_observation, reward, terminated, truncated, _ = env.step(executed)
        records.append(
            StepRecord(
                observation=observation,
                action=executed,
                teacher_action=teacher_action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                intervention=intervention,
                correction=correction,
            )
        )
        observation = next_observation
        if terminated or truncated:
            break
    return Trajectory(episode=episode, steps=records)


def run_dagger(
    policy: NavigationPolicy,
    episodes: list[Episode],
    *,
    initial_data: list[Trajectory] | None = None,
    config: DAggerConfig | None = None,
    env_config: EnvConfig | None = None,
    output_path: str | Path | None = None,
) -> tuple[NavigationPolicy, list[Trajectory], list[dict[str, float]]]:
    """Aggregate teacher-labeled rollouts and retrain after every iteration."""
    config = config or DAggerConfig()
    if not episodes:
        raise ValueError("DAgger requires at least one episode")
    set_deterministic_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    aggregate = list(initial_data or [])
    history: list[dict[str, float]] = []
    for iteration in range(config.iterations):
        beta = config.beta(iteration)
        collected: list[Trajectory] = []
        interventions = 0
        for index in range(config.episodes_per_iteration):
            source = episodes[index % len(episodes)]
            episode = Episode.from_dict(asdict(source))
            trajectory = collect_dagger_trajectory(
                policy,
                ContinuousNavEnv(env_config, seed=source.seed),
                episode,
                beta=beta,
                rng=rng,
                config=config,
            )
            interventions += sum(step.intervention for step in trajectory.steps)
            collected.append(trajectory)
        aggregate.extend(collected)
        policy, losses = train_behavior_cloning(
            aggregate,
            model=policy,
            config=TrainConfig(epochs=config.train_epochs, seed=config.seed + iteration),
        )
        history.append(
            {
                "iteration": float(iteration),
                "beta": beta,
                "trajectories": float(len(aggregate)),
                "interventions": float(interventions),
                "loss": losses[-1],
            }
        )
        if output_path is not None:
            write_jsonl(output_path, aggregate)
    return policy, aggregate, history
