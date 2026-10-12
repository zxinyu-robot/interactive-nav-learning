# Interruptible Interactive Navigation

**这是一个 research/learning repository，不是生产机器人栈，也不代表已具备真实世界自主能力。**

首版提供可在 CPU 上完整运行的闭环：确定性 2D 差速环境 → A* 教师数据 →
行为克隆策略 → 中途语言纠正与 `stop`/`ask` 动作 → DAgger 干预聚合 →
指标评测。核心代码不需要 ROS、Habitat、Isaac Sim、GPU、在线 LLM 或外部数据。

部署边界很重要：`NavigationPolicy.act()` 仅使用传感器、运动/目标状态、语言 token、
GRU 状态和学习参数。`oracle.py` 中的 A* **只允许**用于离线教师标签与 DAgger
采集，绝不进入策略运行时推理。碰撞终止由环境强制执行，是评测安全设施，并非学习能力。

## Quickstart

需要 Python 3.11+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 教师数据（A* 只在此阶段使用）
interrupt-nav generate --config configs/smoke.json \
  --episodes 2 --output artifacts/smoke.jsonl

# 行为克隆
interrupt-nav train --config configs/smoke.json \
  --data artifacts/smoke.jsonl --output checkpoints/bc.pt

# 无规划器的学习策略评测，生成 JSON 报告
interrupt-nav evaluate --config configs/smoke.json \
  --checkpoint checkpoints/bc.pt --output artifacts/report.json

# DAgger：策略 rollout、教师干预、聚合、再训练
interrupt-nav dagger --config configs/smoke.json \
  --checkpoint checkpoints/bc.pt --data artifacts/smoke.jsonl \
  --output-data artifacts/dagger.jsonl \
  --output-checkpoint checkpoints/dagger.pt

# 学习策略的中途目标纠正演示
interrupt-nav demo --checkpoint checkpoints/dagger.pt --steps 40
python examples/minimal.py
```

固定配置和种子时，环境、训练顺序和 CPU 结果可重复。不同 PyTorch/BLAS
版本间不承诺逐位一致。

## 架构

```text
src/interrupt_nav/
  schema.py       版本化 Episode/Observation/Correction/Action/Trajectory + JSONL
  env.py          连续 2D 差速环境、距离射线、障碍/碰撞/安全终止、纠正执行
  oracle.py       教师专用栅格 A* 和轨迹生成
  model.py        传感器编码 + token embedding + GRU + 混合动作头
  training.py     行为克隆、确定性训练
  dagger.py       beta schedule、干预采集、数据聚合、迭代训练
  evaluation.py   无规划器 rollout 和 JSON 报告
  metrics.py      success、SPL 等指标
  cli.py          generate/train/evaluate/dagger/demo
configs/          smoke 与 baseline JSON 配置
examples/         最小端到端示例
tests/            CPU 稳定测试
```

策略输入包括 `[0,1]` 归一化距离射线、线/角速度和朝向、目标相对距离/方位、
当前语言目标 token 及最近纠正 token。GRU 保留跨步记忆。输出为受限线速度、
受限角速度以及独立 `stop`/`ask` 二分类头；在线推理不由 LLM 直接生成速度。

支持的中途纠正：

- `change_goal`：立即替换目标；
- `forbid_region`：把指定矩形加入动态禁区和距离观测；
- `forbid_direction`：禁止 `left`、`right` 或 `forward`；
- `stop` / `continue`：暂停和恢复任务。

环境会强制执行禁区、方向和暂停约束。策略能否从 token 学会及时响应，应通过
`correction_response_steps` 评测，不能把环境约束算作学习成果。

## 数据格式

JSONL 每行是一个完整 `Trajectory`，包含 `version: "1.0"`、一个 `Episode`
以及有序 `StepRecord`。step 保存观测、实际动作、可选教师动作、纠正、奖励、
终止标志和干预标志。核心 schema 均为 dataclass，并由 `read_jsonl` /
`write_jsonl` roundtrip。示意：

```json
{"version":"1.0","episode":{"episode_id":"seed-0","seed":0,"start":[1,1,0],"goal":[5,5],"obstacles":[],"max_steps":30,"language":"go to the goal","corrections":[],"shortest_path":5.66,"version":"1.0"},"steps":[]}
```

## 评测定义

- `success`：到达当前目标的 episode 比例；
- `spl`：成功指示 × 最短路长度 / max(最短路长度, 实际路程)；
- `collisions_per_100m`：每 100 米碰撞数；
- `interventions_per_episode`：每 episode 教师接管次数；
- `correction_response_steps`：纠正发出到动作满足响应判据的步数；
- `control_smoothness`：相邻归一化控制量变化的均值（越低越平滑）；
- `ask_rate`：触发 ask 的步数比例。

未在 episode 结束前响应的纠正按剩余 horizon 计入延迟，避免静默忽略失败。
评测 CLI 输出机器可读 JSON，并包含逐 episode 明细。

## 开发

```bash
ruff check .
ruff format --check .
pytest
```

CI 在 Python 3.11 的 Ubuntu CPU runner 上执行同样检查。`.gitignore` 排除
checkpoint、运行产物、原始数据和 ONNX 文件。

## 已知限制与扩展点

- 语言目前是固定、版本化 token 词表，不含自由文本模型或在线 LLM；
- 2D 射线、运动学和 A* 教师是轻量研究替身，不建模真实传感器噪声、动力学或延迟；
- 首版 DAgger 使用动作偏差阈值和 beta 混合触发教师接管，不等同于人类监督；
- 未实现 sim-to-real、形式化安全保证、ROS2、Habitat、Isaac Sim 或 ONNX 导出。

后续可在环境/观测适配器边界接入 Habitat 或 Isaac，在策略 I/O 边界增加 ROS2
节点和 ONNX 导出；这些适配器应保持可选，且不得把规划器接入部署策略推理路径。
