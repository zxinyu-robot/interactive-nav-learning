"""Deterministic CPU-only continuous differential-drive navigation environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from interrupt_nav.schema import Action, Correction, Episode, Observation

CORRECTION_TOKENS = {
    "none": 0,
    "change_goal": 1,
    "forbid_region": 2,
    "forbid_left": 3,
    "forbid_right": 4,
    "forbid_forward": 5,
    "stop": 6,
    "continue": 7,
}


@dataclass(slots=True)
class EnvConfig:
    world_size: float = 10.0
    num_obstacles: int = 5
    max_steps: int = 200
    dt: float = 0.2
    max_linear_speed: float = 1.0
    max_angular_speed: float = 2.0
    robot_radius: float = 0.18
    goal_tolerance: float = 0.35
    sensor_rays: int = 12
    sensor_range: float = 3.0
    terminate_on_collision: bool = True


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


class ContinuousNavEnv:
    """Small Gym-like environment with no Gym dependency.

    Collision termination is an environment safety facility, not learned behavior.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int = 0):
        self.config = config or EnvConfig()
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.pose = np.zeros(3, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.obstacles: list[tuple[float, float, float, float]] = []
        self.forbidden_regions: list[tuple[float, float, float, float]] = []
        self.forbidden_direction: str | None = None
        self.paused = False
        self.last_correction_token = 0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.step_count = 0
        self.path_length = 0.0
        self.collision_count = 0
        self.episode: Episode | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        episode: Episode | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        if episode is None:
            start = (1.0, 1.0, float(self.rng.uniform(-pi, pi)))
            goal = (self.config.world_size - 1.0, self.config.world_size - 1.0)
            obstacles = self._generate_obstacles(start[:2], goal)
            episode = Episode(
                episode_id=f"seed-{self._seed}",
                seed=self._seed,
                start=start,
                goal=goal,
                obstacles=obstacles,
                max_steps=self.config.max_steps,
                shortest_path=hypot(goal[0] - start[0], goal[1] - start[1]),
            )
        self.episode = episode
        self.pose[:] = episode.start
        self.goal[:] = episode.goal
        self.obstacles = list(episode.obstacles)
        self.forbidden_regions = []
        self.forbidden_direction = None
        self.paused = False
        self.last_correction_token = 0
        self.linear_velocity = self.angular_velocity = 0.0
        self.step_count = self.collision_count = 0
        self.path_length = 0.0
        return self._observation(), {"episode": episode}

    def _generate_obstacles(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float, float, float]]:
        obstacles: list[tuple[float, float, float, float]] = []
        attempts = 0
        while len(obstacles) < self.config.num_obstacles and attempts < 200:
            attempts += 1
            width, height = self.rng.uniform(0.5, 1.2, size=2)
            x = float(self.rng.uniform(0.6, self.config.world_size - width - 0.6))
            y = float(self.rng.uniform(0.6, self.config.world_size - height - 0.6))
            rect = (x, y, x + float(width), y + float(height))
            margin = self.config.robot_radius + 0.5
            if self._point_near_rect(start, rect, margin) or self._point_near_rect(
                goal, rect, margin
            ):
                continue
            obstacles.append(rect)
        return obstacles

    @staticmethod
    def _point_near_rect(
        point: tuple[float, float], rect: tuple[float, float, float, float], margin: float
    ) -> bool:
        return (
            rect[0] - margin <= point[0] <= rect[2] + margin
            and rect[1] - margin <= point[1] <= rect[3] + margin
        )

    def apply_correction(self, correction: Correction) -> None:
        """Apply a user correction immediately and expose its token in observations."""
        if correction.kind == "change_goal":
            assert correction.target is not None
            if not all(0.0 <= value <= self.config.world_size for value in correction.target):
                raise ValueError("corrected goal lies outside the world")
            self.goal[:] = correction.target
            self.last_correction_token = CORRECTION_TOKENS["change_goal"]
        elif correction.kind == "forbid_region":
            assert correction.region is not None
            x0, y0, x1, y1 = correction.region
            ordered_x = 0 <= x0 < x1 <= self.config.world_size
            ordered_y = 0 <= y0 < y1 <= self.config.world_size
            if not (ordered_x and ordered_y):
                raise ValueError("forbidden region must be ordered and inside the world")
            self.forbidden_regions.append(correction.region)
            self.last_correction_token = CORRECTION_TOKENS["forbid_region"]
        elif correction.kind == "forbid_direction":
            self.forbidden_direction = correction.direction
            self.last_correction_token = CORRECTION_TOKENS[f"forbid_{correction.direction}"]
        elif correction.kind == "stop":
            self.paused = True
            self.last_correction_token = CORRECTION_TOKENS["stop"]
        elif correction.kind == "continue":
            self.paused = False
            self.forbidden_direction = None
            self.last_correction_token = CORRECTION_TOKENS["continue"]
        if self.episode is not None and correction not in self.episode.corrections:
            self.episode.corrections.append(correction)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        linear = float(np.clip(action.linear, 0.0, self.config.max_linear_speed))
        angular = float(
            np.clip(action.angular, -self.config.max_angular_speed, self.config.max_angular_speed)
        )
        if self.paused or action.stop:
            linear = angular = 0.0
        if self.forbidden_direction == "forward":
            linear = 0.0
        elif self.forbidden_direction == "left":
            angular = min(0.0, angular)
        elif self.forbidden_direction == "right":
            angular = max(0.0, angular)

        old_pose = self.pose.copy()
        new_heading = wrap_angle(float(self.pose[2] + angular * self.config.dt))
        candidate = np.array(
            [
                self.pose[0] + linear * cos(new_heading) * self.config.dt,
                self.pose[1] + linear * sin(new_heading) * self.config.dt,
                new_heading,
            ]
        )
        collision = self._collides(float(candidate[0]), float(candidate[1]))
        if collision:
            self.collision_count += 1
            candidate[:2] = self.pose[:2]
            linear = 0.0
        self.pose[:] = candidate
        self.linear_velocity, self.angular_velocity = linear, angular
        self.path_length += hypot(self.pose[0] - old_pose[0], self.pose[1] - old_pose[1])
        self.step_count += 1
        success = self.distance_to_goal <= self.config.goal_tolerance
        terminated = success or (collision and self.config.terminate_on_collision)
        max_steps = self.episode.max_steps if self.episode else self.config.max_steps
        truncated = self.step_count >= max_steps and not terminated
        reward = (
            10.0 if success else -0.01 - 0.02 * self.distance_to_goal - (1.0 if collision else 0.0)
        )
        info: dict[str, object] = {
            "success": success,
            "collision": collision,
            "collision_count": self.collision_count,
            "path_length": self.path_length,
            "distance_to_goal": self.distance_to_goal,
            "applied_linear": linear,
            "applied_angular": angular,
            "ask": action.ask,
            "safety_termination": collision and self.config.terminate_on_collision,
        }
        return self._observation(), reward, terminated, truncated, info

    @property
    def distance_to_goal(self) -> float:
        return hypot(float(self.goal[0] - self.pose[0]), float(self.goal[1] - self.pose[1]))

    def observe(self) -> Observation:
        """Return the current observation without advancing simulation."""
        return self._observation()

    def _collides(self, x: float, y: float) -> bool:
        radius = self.config.robot_radius
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius > self.config.world_size
            or y + radius > self.config.world_size
        ):
            return True
        for x0, y0, x1, y1 in self.obstacles + self.forbidden_regions:
            nearest_x = min(max(x, x0), x1)
            nearest_y = min(max(y, y0), y1)
            if hypot(x - nearest_x, y - nearest_y) <= radius:
                return True
        return False

    def _observation(self) -> Observation:
        heading = float(self.pose[2])
        dx, dy = float(self.goal[0] - self.pose[0]), float(self.goal[1] - self.pose[1])
        bearing = wrap_angle(atan2(dy, dx) - heading)
        diagonal = self.config.world_size * 2**0.5
        language_token = 1 + int(dx < 0) + 2 * int(dy < 0)
        return Observation(
            distances=self._scan(),
            motion=[
                self.linear_velocity / self.config.max_linear_speed,
                self.angular_velocity / self.config.max_angular_speed,
                sin(heading),
                cos(heading),
            ],
            goal_relative=[
                min(1.0, hypot(dx, dy) / diagonal),
                cos(bearing),
                sin(bearing),
            ],
            language_token=language_token,
            correction_token=self.last_correction_token,
            step=self.step_count,
        )

    def _scan(self) -> list[float]:
        values: list[float] = []
        all_rects = self.obstacles + self.forbidden_regions
        for offset in np.linspace(-pi, pi, self.config.sensor_rays, endpoint=False):
            angle = float(self.pose[2] + offset)
            distance = self.config.sensor_range
            # Fixed increments keep scans deterministic and inexpensive.
            for candidate in np.linspace(0.05, self.config.sensor_range, 60):
                x = float(self.pose[0] + candidate * cos(angle))
                y = float(self.pose[1] + candidate * sin(angle))
                if not (0 <= x <= self.config.world_size and 0 <= y <= self.config.world_size):
                    distance = float(candidate)
                    break
                if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in all_rects):
                    distance = float(candidate)
                    break
            values.append(float(np.clip(distance / self.config.sensor_range, 0.0, 1.0)))
        return values
