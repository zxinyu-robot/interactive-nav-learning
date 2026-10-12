"""Command-line entry points for data generation, training, evaluation, and DAgger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from interrupt_nav.env import ContinuousNavEnv, EnvConfig
from interrupt_nav.evaluation import evaluate_policy
from interrupt_nav.model import NavigationPolicy, PolicyConfig, set_deterministic_seed
from interrupt_nav.schema import Correction, Episode, read_jsonl, write_jsonl
from interrupt_nav.training import TrainConfig, train_behavior_cloning


def _config(path: str | None) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def _episodes(seeds: list[int], env_config: EnvConfig) -> list[Episode]:
    episodes: list[Episode] = []
    for seed in seeds:
        env = ContinuousNavEnv(env_config, seed=seed)
        _, info = env.reset(seed=seed)
        episodes.append(info["episode"])  # type: ignore[arg-type]
    return episodes


def command_generate(args: argparse.Namespace) -> None:
    from interrupt_nav.oracle import AStarTeacher, generate_teacher_trajectory

    values = _config(args.config)
    env_config = EnvConfig(**values.get("env", {}))
    seeds = values.get("seeds", list(range(args.episodes)))
    trajectories = []
    for episode in _episodes(seeds[: args.episodes], env_config):
        episode.corrections = [
            Correction.from_dict(item) for item in values.get("teacher_corrections", [])
        ]
        env = ContinuousNavEnv(env_config, seed=episode.seed)
        teacher = AStarTeacher()
        env.reset(seed=episode.seed, episode=episode)
        episode.shortest_path = teacher.shortest_path_length(env)
        trajectories.append(generate_teacher_trajectory(env, episode, teacher))
    write_jsonl(args.output, trajectories)
    print(json.dumps({"trajectories": len(trajectories), "output": args.output}))


def command_train(args: argparse.Namespace) -> None:
    values = _config(args.config)
    trajectories = read_jsonl(args.data)
    if not trajectories or not trajectories[0].steps:
        raise ValueError("training data must contain at least one non-empty trajectory")
    train_config = TrainConfig(**values.get("train", {}))
    set_deterministic_seed(train_config.seed)
    model_values = dict(values.get("model", {}))
    model_values.setdefault("sensor_dim", len(trajectories[0].steps[0].observation.distances))
    model_config = PolicyConfig(**model_values)
    model = NavigationPolicy(model_config)
    model, losses = train_behavior_cloning(
        trajectories,
        model=model,
        config=train_config,
    )
    model.save_checkpoint(args.output, {"final_loss": losses[-1], "epochs": len(losses)})
    print(json.dumps({"checkpoint": args.output, "loss": losses[-1]}))


def command_evaluate(args: argparse.Namespace) -> None:
    values = _config(args.config)
    model, _ = NavigationPolicy.load_checkpoint(args.checkpoint)
    seeds = values.get("evaluation_seeds", values.get("seeds", [100, 101, 102]))
    env_config = EnvConfig(**values.get("env", {}))
    episodes = _episodes(seeds, env_config)
    for episode in episodes:
        episode.corrections = [
            Correction.from_dict(item) for item in values.get("teacher_corrections", [])
        ]
    report = evaluate_policy(
        model,
        seeds=seeds,
        episodes=episodes,
        env_config=env_config,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


def command_dagger(args: argparse.Namespace) -> None:
    from interrupt_nav.dagger import DAggerConfig, run_dagger
    from interrupt_nav.oracle import generate_teacher_trajectory

    values = _config(args.config)
    env_config = EnvConfig(**values.get("env", {}))
    dagger_config = DAggerConfig(**values.get("dagger", {}))
    episodes = _episodes(values.get("seeds", [0, 1]), env_config)
    for episode in episodes:
        episode.corrections = [
            Correction.from_dict(item) for item in values.get("teacher_corrections", [])
        ]
    if args.checkpoint:
        model, _ = NavigationPolicy.load_checkpoint(args.checkpoint)
        initial = read_jsonl(args.data) if args.data else []
    else:
        initial = [
            generate_teacher_trajectory(ContinuousNavEnv(env_config, seed=episode.seed), episode)
            for episode in episodes
        ]
        model, _ = train_behavior_cloning(
            initial,
            config=TrainConfig(epochs=max(1, dagger_config.train_epochs), seed=dagger_config.seed),
        )
    model, aggregate, history = run_dagger(
        model,
        episodes,
        initial_data=initial,
        config=dagger_config,
        env_config=env_config,
        output_path=args.output_data,
    )
    model.save_checkpoint(args.output_checkpoint, {"dagger_history": history})
    print(
        json.dumps(
            {
                "checkpoint": args.output_checkpoint,
                "trajectories": len(aggregate),
                "history": history,
            },
            indent=2,
        )
    )


def command_demo(args: argparse.Namespace) -> None:
    set_deterministic_seed(args.seed)
    model = (
        NavigationPolicy.load_checkpoint(args.checkpoint)[0]
        if args.checkpoint
        else NavigationPolicy()
    )
    env = ContinuousNavEnv(EnvConfig(max_steps=args.steps), seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    hidden = None
    events = []
    for step in range(args.steps):
        if step == args.correction_step:
            correction = Correction(
                kind="change_goal",
                token="new_goal",
                step=step,
                target=(1.0, env.config.world_size - 1.0),
            )
            env.apply_correction(correction)
            observation = env.observe()
            events.append({"step": step, "correction": correction.kind})
        action, hidden = model.act(observation, hidden)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(
        json.dumps(
            {
                "steps": env.step_count,
                "success": info["success"],
                "collisions": env.collision_count,
                "path_length": env.path_length,
                "events": events,
                "note": "collision termination is an environment safety facility",
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interrupt-nav")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate teacher-only JSONL data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--episodes", type=int, default=4)
    generate.add_argument("--config")
    generate.set_defaults(func=command_generate)

    train = subparsers.add_parser("train", help="train behavior cloning")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a learned checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config")
    evaluate.set_defaults(func=command_evaluate)

    dagger = subparsers.add_parser("dagger", help="collect interventions and retrain")
    dagger.add_argument("--checkpoint")
    dagger.add_argument("--data")
    dagger.add_argument("--output-data", required=True)
    dagger.add_argument("--output-checkpoint", required=True)
    dagger.add_argument("--config")
    dagger.set_defaults(func=command_dagger)

    demo = subparsers.add_parser("demo", help="run learned-policy correction demo")
    demo.add_argument("--checkpoint")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--steps", type=int, default=50)
    demo.add_argument("--correction-step", type=int, default=10)
    demo.set_defaults(func=command_demo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
