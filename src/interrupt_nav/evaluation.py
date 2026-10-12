"""Planner-free learned-policy evaluation."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path

from interrupt_nav.env import ContinuousNavEnv, EnvConfig
from interrupt_nav.metrics import EpisodeMetrics, aggregate_metrics
from interrupt_nav.model import NavigationPolicy, set_deterministic_seed
from interrupt_nav.schema import Action, Correction, Episode


def _responded(correction: Correction, action: Action, previous: Action | None) -> bool:
    if correction.kind == "stop":
        return action.stop or (action.linear < 0.05 and abs(action.angular) < 0.05)
    if correction.kind == "continue":
        return not action.stop and action.linear > 0.05
    if correction.kind == "forbid_direction":
        if correction.direction == "forward":
            return action.linear < 0.05
        if correction.direction == "left":
            return action.angular <= 0
        return action.angular >= 0
    if previous is None:
        return True
    return hypot(action.linear - previous.linear, action.angular - previous.angular) > 0.05


def evaluate_episode(
    policy: NavigationPolicy,
    episode: Episode,
    *,
    env_config: EnvConfig | None = None,
    corrections: list[Correction] | None = None,
) -> EpisodeMetrics:
    env = ContinuousNavEnv(env_config, seed=episode.seed)
    observation, _ = env.reset(seed=episode.seed, episode=episode)
    schedule = episode.corrections if corrections is None else corrections
    corrections_by_step = {item.step: item for item in schedule}
    pending: list[tuple[Correction, int]] = []
    latencies: list[int] = []
    hidden = None
    previous_action: Action | None = None
    smoothness: list[float] = []
    asks = 0
    success = False
    while True:
        correction = corrections_by_step.get(env.step_count)
        if correction is not None:
            env.apply_correction(correction)
            observation = env.observe()
            pending.append((correction, env.step_count))
        action, hidden = policy.act(observation, hidden)
        asks += int(action.ask)
        if previous_action is not None:
            smoothness.append(
                hypot(
                    action.linear - previous_action.linear,
                    (action.angular - previous_action.angular) / env.config.max_angular_speed,
                )
            )
        unresolved: list[tuple[Correction, int]] = []
        for item, issued_at in pending:
            if _responded(item, action, previous_action):
                latencies.append(env.step_count - issued_at + 1)
            else:
                unresolved.append((item, issued_at))
        pending = unresolved
        observation, _, terminated, truncated, info = env.step(action)
        success = bool(info["success"])
        previous_action = action
        if terminated or truncated:
            break
    latencies.extend(env.step_count - issued_at + 1 for _, issued_at in pending)
    shortest = episode.shortest_path
    if shortest is None:
        shortest = hypot(episode.goal[0] - episode.start[0], episode.goal[1] - episode.start[1])
    return EpisodeMetrics(
        success=success,
        shortest_path=shortest,
        path_length=env.path_length,
        collisions=env.collision_count,
        interventions=0,
        correction_response_steps=latencies,
        control_smoothness=sum(smoothness) / max(1, len(smoothness)),
        asks=asks,
        steps=env.step_count,
    )


def evaluate_policy(
    policy: NavigationPolicy,
    *,
    seeds: list[int],
    env_config: EnvConfig | None = None,
    episodes: list[Episode] | None = None,
    corrections: dict[int, list[Correction]] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    set_deterministic_seed(min(seeds) if seeds else 0)
    if episodes is None:
        episodes = []
        for seed in seeds:
            env = ContinuousNavEnv(env_config, seed=seed)
            _, info = env.reset(seed=seed)
            episodes.append(info["episode"])  # type: ignore[arg-type]
    results = [
        evaluate_episode(
            policy,
            episode,
            env_config=env_config,
            corrections=(corrections or {}).get(episode.seed),
        )
        for episode in episodes
    ]
    report = aggregate_metrics(results)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
