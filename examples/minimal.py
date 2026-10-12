"""Tiny end-to-end example; runs on CPU without ROS or a simulator."""

from interrupt_nav.env import ContinuousNavEnv, EnvConfig
from interrupt_nav.evaluation import evaluate_policy
from interrupt_nav.oracle import generate_teacher_trajectory
from interrupt_nav.training import TrainConfig, train_behavior_cloning

config = EnvConfig(world_size=6.0, num_obstacles=2, max_steps=40)
environment = ContinuousNavEnv(config, seed=0)
_, reset_info = environment.reset(seed=0)
trajectory = generate_teacher_trajectory(environment, reset_info["episode"])
policy, losses = train_behavior_cloning([trajectory], config=TrainConfig(epochs=2, seed=0))
report = evaluate_policy(policy, seeds=[10], env_config=config)
print({"loss": losses[-1], "success": report["success"], "spl": report["spl"]})
