"""A* teacher used only for offline labels and DAgger interventions.

This module must not be imported by runtime policy inference. The deployed
``NavigationPolicy.act`` path depends only on learned parameters and observations.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import atan2, cos, hypot, pi

from interrupt_nav.env import ContinuousNavEnv, wrap_angle
from interrupt_nav.schema import Action, Correction, Episode, StepRecord, Trajectory


@dataclass(slots=True)
class OracleConfig:
    grid_resolution: float = 0.25
    angular_gain: float = 2.5
    waypoint_distance: float = 0.5


class AStarTeacher:
    """Classical teacher. Never use this class in deployment policy inference."""

    def __init__(self, config: OracleConfig | None = None):
        self.config = config or OracleConfig()

    def act(self, env: ContinuousNavEnv) -> Action:
        if env.paused:
            return Action(0.0, 0.0, stop=True)
        if env.distance_to_goal <= env.config.goal_tolerance:
            return Action(0.0, 0.0, stop=True)
        path = self.plan(env)
        if len(path) < 2:
            return Action(0.0, 0.0, ask=True)
        lookahead = max(1, int(self.config.waypoint_distance / self.config.grid_resolution))
        waypoint = path[min(len(path) - 1, lookahead)]
        desired = atan2(waypoint[1] - env.pose[1], waypoint[0] - env.pose[0])
        error = wrap_angle(desired - float(env.pose[2]))
        angular = max(
            -env.config.max_angular_speed,
            min(env.config.max_angular_speed, self.config.angular_gain * error),
        )
        alignment = max(0.0, cos(error))
        clearance = min(env._scan())  # noqa: SLF001 - privileged teacher sees simulator geometry.
        linear = env.config.max_linear_speed * alignment * min(1.0, clearance * 2.0)
        if abs(error) > pi / 2:
            linear = 0.0
        return Action(linear=linear, angular=angular)

    def plan(self, env: ContinuousNavEnv) -> list[tuple[float, float]]:
        resolution = self.config.grid_resolution
        size = int(env.config.world_size / resolution) + 1

        def cell(point: tuple[float, float]) -> tuple[int, int]:
            return (
                max(0, min(size - 1, round(point[0] / resolution))),
                max(0, min(size - 1, round(point[1] / resolution))),
            )

        def point(node: tuple[int, int]) -> tuple[float, float]:
            return node[0] * resolution, node[1] * resolution

        blocked: set[tuple[int, int]] = set()
        margin = env.config.robot_radius
        for x0, y0, x1, y1 in env.obstacles + env.forbidden_regions:
            ix0, iy0 = cell((max(0.0, x0 - margin), max(0.0, y0 - margin)))
            ix1, iy1 = cell(
                (min(env.config.world_size, x1 + margin), min(env.config.world_size, y1 + margin))
            )
            blocked.update((x, y) for x in range(ix0, ix1 + 1) for y in range(iy0, iy1 + 1))
        start = cell((float(env.pose[0]), float(env.pose[1])))
        goal = cell((float(env.goal[0]), float(env.goal[1])))
        blocked.discard(start)
        blocked.discard(goal)
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        costs = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy in moves:
                neighbor = current[0] + dx, current[1] + dy
                if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                    continue
                if neighbor in blocked:
                    continue
                next_cost = costs[current] + hypot(dx, dy)
                if next_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = next_cost
                    parents[neighbor] = current
                    priority = next_cost + hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(frontier, (priority, neighbor))
        if goal not in costs:
            return []
        nodes = [goal]
        while nodes[-1] != start:
            nodes.append(parents[nodes[-1]])
        return [point(node) for node in reversed(nodes)]

    def shortest_path_length(self, env: ContinuousNavEnv) -> float:
        path = self.plan(env)
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:], strict=False))


def generate_teacher_trajectory(
    env: ContinuousNavEnv,
    episode: Episode,
    teacher: AStarTeacher | None = None,
    corrections: list[Correction] | None = None,
) -> Trajectory:
    teacher = teacher or AStarTeacher()
    observation, _ = env.reset(seed=episode.seed, episode=episode)
    schedule = episode.corrections if corrections is None else corrections
    by_step = {correction.step: correction for correction in schedule}
    records: list[StepRecord] = []
    while True:
        correction = by_step.get(env.step_count)
        if correction is not None:
            env.apply_correction(correction)
            observation = env.observe()
        action = teacher.act(env)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        records.append(
            StepRecord(
                observation=observation,
                action=action,
                teacher_action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                correction=correction,
            )
        )
        observation = next_observation
        if terminated or truncated:
            break
    return Trajectory(episode=episode, steps=records)
