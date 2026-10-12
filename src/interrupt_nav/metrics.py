"""Navigation metrics with explicit definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(slots=True)
class EpisodeMetrics:
    success: bool
    shortest_path: float
    path_length: float
    collisions: int
    interventions: int
    correction_response_steps: list[int]
    control_smoothness: float
    asks: int
    steps: int


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("metrics require at least one episode")
    success_rate = mean(float(item.success) for item in episodes)
    spl_values = [
        float(item.success) * item.shortest_path / max(item.shortest_path, item.path_length, 1e-9)
        for item in episodes
    ]
    total_distance = sum(item.path_length for item in episodes)
    total_collisions = sum(item.collisions for item in episodes)
    response_steps = [latency for item in episodes for latency in item.correction_response_steps]
    total_steps = sum(item.steps for item in episodes)
    return {
        "success": success_rate,
        "spl": mean(spl_values),
        "collisions_per_100m": (
            100.0 * total_collisions / total_distance if total_distance > 0 else 0.0
        ),
        "interventions_per_episode": mean(item.interventions for item in episodes),
        "correction_response_steps": mean(response_steps) if response_steps else None,
        "control_smoothness": mean(item.control_smoothness for item in episodes),
        "ask_rate": sum(item.asks for item in episodes) / max(1, total_steps),
        "episodes": len(episodes),
        "per_episode": [asdict(item) for item in episodes],
    }
