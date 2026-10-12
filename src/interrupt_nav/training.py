"""Behavior-cloning training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from interrupt_nav.model import NavigationPolicy, PolicyConfig, set_deterministic_seed
from interrupt_nav.schema import Action, Trajectory, read_jsonl


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 0
    stop_loss_weight: float = 0.5
    ask_loss_weight: float = 0.25
    grad_clip: float = 1.0


def _trajectory_tensors(
    trajectory: Trajectory, device: torch.device
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
    if not trajectory.steps:
        raise ValueError("cannot train on an empty trajectory")
    observations = [step.observation for step in trajectory.steps]
    actions: list[Action] = [
        step.teacher_action if step.teacher_action is not None else step.action
        for step in trajectory.steps
    ]

    def float_tensor(rows: list[list[float]]) -> Tensor:
        return torch.tensor(rows, dtype=torch.float32, device=device).unsqueeze(0)

    inputs = (
        float_tensor([item.distances for item in observations]),
        float_tensor([item.motion for item in observations]),
        float_tensor([item.goal_relative for item in observations]),
        torch.tensor(
            [[item.language_token for item in observations]], dtype=torch.long, device=device
        ),
        torch.tensor(
            [[item.correction_token for item in observations]], dtype=torch.long, device=device
        ),
    )
    targets = (
        torch.tensor([[item.linear for item in actions]], dtype=torch.float32, device=device),
        torch.tensor([[item.angular for item in actions]], dtype=torch.float32, device=device),
        torch.tensor([[item.stop for item in actions]], dtype=torch.float32, device=device),
        torch.tensor([[item.ask for item in actions]], dtype=torch.float32, device=device),
    )
    return inputs, targets


def train_behavior_cloning(
    trajectories: list[Trajectory] | str | Path,
    *,
    model: NavigationPolicy | None = None,
    config: TrainConfig | None = None,
    device: str = "cpu",
) -> tuple[NavigationPolicy, list[float]]:
    """Train a recurrent policy on complete episodes in deterministic order."""
    config = config or TrainConfig()
    set_deterministic_seed(config.seed)
    if isinstance(trajectories, (str, Path)):
        trajectories = read_jsonl(trajectories)
    if not trajectories:
        raise ValueError("training requires at least one trajectory")
    if any(not trajectory.steps for trajectory in trajectories):
        raise ValueError("training trajectories must contain at least one step")
    if config.epochs < 1:
        raise ValueError("epochs must be at least one")
    target_device = torch.device(device)
    if model is None:
        sensor_dim = len(trajectories[0].steps[0].observation.distances)
        model = NavigationPolicy(PolicyConfig(sensor_dim=sensor_dim))
    model.to(target_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    binary_loss = nn.BCEWithLogitsLoss()
    history: list[float] = []
    for _ in range(config.epochs):
        total = 0.0
        for trajectory in trajectories:
            inputs, targets = _trajectory_tensors(trajectory, target_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(*inputs)
            loss = (
                nn.functional.mse_loss(output.linear, targets[0])
                + nn.functional.mse_loss(output.angular, targets[1])
                + config.stop_loss_weight * binary_loss(output.stop_logits, targets[2])
                + config.ask_loss_weight * binary_loss(output.ask_logits, targets[3])
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(trajectories))
    return model, history
