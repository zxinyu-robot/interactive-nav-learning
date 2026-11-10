# Interruptible Interactive Navigation Learning

> 本仓库已并入 [embodied-learning-baselines-](https://github.com/zxinyu-robot/embodied-learning-baselines-) 的 `interactive_nav/`，此处只读归档，后续维护在新仓库进行。

## 项目状态

| 项目 | 状态 |
| --- | --- |
| 维护状态 | Archived |
| 后续维护 | [embodied-learning-baselines-/interactive_nav](https://github.com/zxinyu-robot/embodied-learning-baselines-/tree/main/interactive_nav) |
| 项目类型 | 交互式导航学习基线 |
| 验证方式 | 单元测试与合成环境 |

## 项目简介

本项目探索可中断交互式导航学习框架：人类可以在机器人自主导航过程中根据环境变化、任务歧义、导航风险或策略失效随时接管控制。系统记录介入时的环境状态、任务意图、机器人原有行为与人工修正动作，并将纠正行为转化为可积累、可检索和可复用的导航经验。

## 核心能力

- 人工接管与在线纠正信号记录。
- 行为克隆与 DAgger 学习基线。
- 介入状态、任务意图和动作差异分析。
- 合成环境中的可复现训练、评估和失败分类。

## 归档说明

该项目的实现已迁移到 `embodied-learning-baselines-/interactive_nav/`。请在目标仓库提交问题、改进和文档更新；本仓库不再接受独立功能开发。

## 许可证

以仓库中的 LICENSE 文件为准。
