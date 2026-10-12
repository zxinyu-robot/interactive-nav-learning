from __future__ import annotations

from dataclasses import asdict
from math import pi

import numpy as np
import torch

from interrupt_nav.dagger import DAggerConfig, run_dagger
from interrupt_nav.env import ContinuousNavEnv, EnvConfig
from interrupt_nav.metrics import EpisodeMetrics, aggregate_metrics
from interrupt_nav.model import NavigationPolicy, PolicyConfig
from interrupt_nav.oracle import generate_teacher_trajectory
from interrupt_nav.schema import (
    Action,
    Correction,
    Episode,
    Observation,
    StepRecord,
    Trajectory,
    read_jsonl,
    write_jsonl,
)
from interrupt_nav.training import TrainConfig, train_behavior_cloning


def episode(max_steps: int = 8) -> Episode:
    return Episode(
        episode_id="test",
        seed=4,
        start=(1.0, 1.0, 0.0),
        goal=(4.0, 1.0),
        obstacles=[],
        max_steps=max_steps,
        shortest_path=3.0,
    )


def test_environment_is_deterministic() -> None:
    config = EnvConfig(num_obstacles=3)
    env_a = ContinuousNavEnv(config, seed=12)
    env_b = ContinuousNavEnv(config, seed=12)
    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()
    assert asdict(obs_a) == asdict(obs_b)
    assert info_a["episode"].obstacles == info_b["episode"].obstacles
    for _ in range(3):
        obs_a, *_ = env_a.step(Action(0.4, 0.1))
        obs_b, *_ = env_b.step(Action(0.4, 0.1))
        assert asdict(obs_a) == asdict(obs_b)


def test_collision_and_max_step_termination() -> None:
    env = ContinuousNavEnv(EnvConfig(world_size=5, num_obstacles=0), seed=0)
    wall_episode = episode()
    wall_episode.start = (0.2, 2.0, pi)
    env.reset(episode=wall_episode)
    _, _, terminated, _, info = env.step(Action(1.0, 0.0))
    assert terminated and info["collision"] and info["safety_termination"]

    config = EnvConfig(world_size=5, num_obstacles=0, terminate_on_collision=False)
    env = ContinuousNavEnv(config)
    short = episode(max_steps=2)
    env.reset(episode=short)
    env.step(Action(0.0, 0.0))
    _, _, terminated, truncated, _ = env.step(Action(0.0, 0.0))
    assert not terminated and truncated


def test_corrections_take_effect() -> None:
    env = ContinuousNavEnv(EnvConfig(world_size=5, num_obstacles=0))
    env.reset(episode=episode())
    env.apply_correction(Correction("change_goal", "new", target=(1.0, 4.0)))
    assert tuple(env.goal) == (1.0, 4.0)
    assert env.observe().correction_token != 0
    env.apply_correction(Correction("forbid_direction", "no-left", direction="left"))
    _, _, _, _, info = env.step(Action(0.5, 1.0))
    assert info["applied_angular"] <= 0
    env.apply_correction(Correction("stop", "stop"))
    _, _, _, _, info = env.step(Action(1.0, 1.0))
    assert info["applied_linear"] == 0
    env.apply_correction(Correction("continue", "continue"))
    _, _, _, _, info = env.step(Action(0.2, 0.0))
    assert info["applied_linear"] > 0


def test_schema_jsonl_roundtrip(tmp_path) -> None:
    obs = Observation([1.0] * 12, [0.0] * 4, [0.5, 1.0, 0.0], 1, 0, 0)
    trace = Trajectory(
        episode(),
        [StepRecord(obs, Action(0.5, 0.1), -0.1, False, False)],
    )
    path = tmp_path / "trace.jsonl"
    write_jsonl(path, [trace])
    loaded = read_jsonl(path)
    assert loaded[0].to_dict() == trace.to_dict()


def test_model_output_ranges() -> None:
    model = NavigationPolicy(PolicyConfig(hidden_dim=16))
    output = model(
        torch.ones(2, 3, 12),
        torch.zeros(2, 3, 4),
        torch.zeros(2, 3, 3),
        torch.ones(2, 3, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.long),
    )
    assert output.linear.shape == (2, 3)
    assert torch.all((0 <= output.linear) & (output.linear <= 1.0))
    assert torch.all((-2.0 <= output.angular) & (output.angular <= 2.0))
    assert output.stop_logits.shape == (2, 3)


def test_short_training_and_checkpoint_roundtrip(tmp_path) -> None:
    env = ContinuousNavEnv(EnvConfig(world_size=5, num_obstacles=0, max_steps=6), seed=4)
    trace = generate_teacher_trajectory(env, episode(max_steps=6))
    model, losses = train_behavior_cloning(
        [trace],
        config=TrainConfig(epochs=2, learning_rate=0.005, seed=3),
    )
    assert len(losses) == 2 and np.isfinite(losses).all()
    path = tmp_path / "policy.pt"
    model.save_checkpoint(path, {"test": True})
    loaded, metadata = NavigationPolicy.load_checkpoint(path)
    assert metadata["test"]
    for left, right in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(left, right)


def test_dagger_aggregates_data() -> None:
    config = EnvConfig(world_size=5, num_obstacles=0, max_steps=5)
    env = ContinuousNavEnv(config, seed=4)
    source = episode(max_steps=5)
    initial = [generate_teacher_trajectory(env, source)]
    policy, _ = train_behavior_cloning(initial, config=TrainConfig(epochs=1))
    _, aggregate, history = run_dagger(
        policy,
        [source],
        initial_data=initial,
        env_config=config,
        config=DAggerConfig(
            iterations=1,
            episodes_per_iteration=1,
            beta_start=1.0,
            beta_end=1.0,
            train_epochs=1,
        ),
    )
    assert len(aggregate) == 2
    assert all(step.intervention for step in aggregate[-1].steps)
    assert history[0]["trajectories"] == 2


def test_metric_calculation() -> None:
    report = aggregate_metrics(
        [
            EpisodeMetrics(True, 5.0, 10.0, 1, 2, [3], 0.2, 1, 10),
            EpisodeMetrics(False, 5.0, 5.0, 0, 0, [1], 0.4, 0, 10),
        ]
    )
    assert report["success"] == 0.5
    assert report["spl"] == 0.25
    assert np.isclose(report["collisions_per_100m"], 100 / 15)
    assert report["interventions_per_episode"] == 1
    assert report["correction_response_steps"] == 2
    assert report["control_smoothness"] == 0.3
    assert report["ask_rate"] == 0.05
