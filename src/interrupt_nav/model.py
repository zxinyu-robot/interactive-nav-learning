"""Recurrent behavior-cloning policy. No planner is used during inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from interrupt_nav.schema import Action, Observation


@dataclass(slots=True)
class PolicyConfig:
    sensor_dim: int = 12
    motion_dim: int = 4
    goal_dim: int = 3
    token_vocab: int = 16
    token_dim: int = 8
    hidden_dim: int = 64
    max_linear_speed: float = 1.0
    max_angular_speed: float = 2.0


@dataclass
class PolicyOutput:
    linear: Tensor
    angular: Tensor
    stop_logits: Tensor
    ask_logits: Tensor
    hidden: Tensor


class NavigationPolicy(nn.Module):
    """Sensor/language encoder, GRU memory, and mixed action heads."""

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.config.sensor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.motion_dim + self.config.goal_dim, 24),
            nn.ReLU(),
        )
        self.language_embedding = nn.Embedding(self.config.token_vocab, self.config.token_dim)
        self.correction_embedding = nn.Embedding(self.config.token_vocab, self.config.token_dim)
        gru_input = 32 + 24 + 2 * self.config.token_dim
        self.gru = nn.GRU(gru_input, self.config.hidden_dim, batch_first=True)
        self.linear_head = nn.Linear(self.config.hidden_dim, 1)
        self.angular_head = nn.Linear(self.config.hidden_dim, 1)
        self.stop_head = nn.Linear(self.config.hidden_dim, 1)
        self.ask_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(
        self,
        distances: Tensor,
        motion: Tensor,
        goal_relative: Tensor,
        language_token: Tensor,
        correction_token: Tensor,
        hidden: Tensor | None = None,
    ) -> PolicyOutput:
        sensor = self.sensor_encoder(distances.float())
        state = self.state_encoder(torch.cat((motion.float(), goal_relative.float()), dim=-1))
        language = self.language_embedding(language_token.long())
        correction = self.correction_embedding(correction_token.long())
        encoded = torch.cat((sensor, state, language, correction), dim=-1)
        memory, next_hidden = self.gru(encoded, hidden)
        linear = torch.sigmoid(self.linear_head(memory)) * self.config.max_linear_speed
        angular = torch.tanh(self.angular_head(memory)) * self.config.max_angular_speed
        return PolicyOutput(
            linear=linear.squeeze(-1),
            angular=angular.squeeze(-1),
            stop_logits=self.stop_head(memory).squeeze(-1),
            ask_logits=self.ask_head(memory).squeeze(-1),
            hidden=next_hidden,
        )

    @torch.no_grad()
    def act(
        self,
        observation: Observation,
        hidden: Tensor | None = None,
        *,
        stop_threshold: float = 0.5,
        ask_threshold: float = 0.5,
    ) -> tuple[Action, Tensor]:
        self.eval()
        device = next(self.parameters()).device

        def feature(values: list[float]) -> Tensor:
            return torch.tensor(values, dtype=torch.float32, device=device).view(1, 1, -1)

        output = self(
            feature(observation.distances),
            feature(observation.motion),
            feature(observation.goal_relative),
            torch.tensor([[observation.language_token]], device=device),
            torch.tensor([[observation.correction_token]], device=device),
            hidden,
        )
        action = Action(
            linear=float(output.linear.item()),
            angular=float(output.angular.item()),
            stop=bool(torch.sigmoid(output.stop_logits).item() >= stop_threshold),
            ask=bool(torch.sigmoid(output.ask_logits).item() >= ask_threshold),
        )
        return action, output.hidden

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            output,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> tuple[NavigationPolicy, dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(PolicyConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model, payload.get("metadata", {})


def set_deterministic_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
