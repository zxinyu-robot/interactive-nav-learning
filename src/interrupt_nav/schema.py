"""Versioned, dependency-free trajectory schemas and JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
CorrectionKind = Literal["change_goal", "forbid_region", "forbid_direction", "stop", "continue"]


@dataclass(slots=True)
class Correction:
    kind: CorrectionKind
    token: str
    step: int = 0
    target: tuple[float, float] | None = None
    region: tuple[float, float, float, float] | None = None
    direction: str | None = None
    version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind == "change_goal" and self.target is None:
            raise ValueError("change_goal requires target")
        if self.kind == "forbid_region" and self.region is None:
            raise ValueError("forbid_region requires region")
        if self.kind == "forbid_direction" and self.direction not in {"left", "right", "forward"}:
            raise ValueError("forbid_direction requires left, right, or forward")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Correction:
        value = dict(data)
        if value.get("target") is not None:
            value["target"] = tuple(value["target"])
        if value.get("region") is not None:
            value["region"] = tuple(value["region"])
        return cls(**value)


@dataclass(slots=True)
class Action:
    linear: float
    angular: float
    stop: bool = False
    ask: bool = False
    version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(**data)


@dataclass(slots=True)
class Observation:
    distances: list[float]
    motion: list[float]
    goal_relative: list[float]
    language_token: int
    correction_token: int
    step: int
    version: str = SCHEMA_VERSION

    def vector(self) -> list[float]:
        """Return the numeric observation excluding categorical token ids."""
        return self.distances + self.motion + self.goal_relative

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**data)


@dataclass(slots=True)
class Episode:
    episode_id: str
    seed: int
    start: tuple[float, float, float]
    goal: tuple[float, float]
    obstacles: list[tuple[float, float, float, float]]
    max_steps: int
    language: str = "go to the goal"
    corrections: list[Correction] = field(default_factory=list)
    shortest_path: float | None = None
    version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        value = dict(data)
        value["start"] = tuple(value["start"])
        value["goal"] = tuple(value["goal"])
        value["obstacles"] = [tuple(item) for item in value["obstacles"]]
        value["corrections"] = [Correction.from_dict(item) for item in value.get("corrections", [])]
        return cls(**value)


@dataclass(slots=True)
class StepRecord:
    observation: Observation
    action: Action
    reward: float
    terminated: bool
    truncated: bool
    teacher_action: Action | None = None
    intervention: bool = False
    correction: Correction | None = None
    version: str = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        value = dict(data)
        value["observation"] = Observation.from_dict(value["observation"])
        value["action"] = Action.from_dict(value["action"])
        if value.get("teacher_action") is not None:
            value["teacher_action"] = Action.from_dict(value["teacher_action"])
        if value.get("correction") is not None:
            value["correction"] = Correction.from_dict(value["correction"])
        return cls(**value)


@dataclass(slots=True)
class Trajectory:
    episode: Episode
    steps: list[StepRecord]
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            episode=Episode.from_dict(data["episode"]),
            steps=[StepRecord.from_dict(item) for item in data["steps"]],
            version=data.get("version", SCHEMA_VERSION),
        )


def write_jsonl(path: str | Path, trajectories: list[Trajectory], append: bool = False) -> None:
    """Write one complete, versioned trajectory per JSONL line."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.to_dict(), separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Trajectory.from_dict(json.loads(line)) for line in handle if line.strip()]
