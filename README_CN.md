# AscendingRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg) ![Status](https://img.shields.io/badge/status-in%20progress-orange.svg)

[English](README.md) | [中文](README_CN.md)

## 介绍

欢迎来到AscendingRL，这是一个致力于探索和推进强化学习（RL）领域的项目。本项目是一段全面的旅程，从使用Gymnasium（OpenAI Gym的更新版本）重现经典RL算法开始。

我们的目标是通过逐步实施来提供对RL算法底层原理和机制的清晰理解。此仓库不仅作为一个教程，还将作为一个实验和学习的活文档。

## 特点

- **重现**：经典RL算法的详细实现。
- **文档**：每个算法都有详尽的文档。
- **实验**：预配置的环境用于测试和调整算法。
- **社区**：这里是学习者和从业者可以交换想法和见解的地方。

## 开始

为了开始使用AscendingRL，你需要在系统上安装Python。我们建议使用Python 3.7或更高版本。

1. 克隆此仓库：
   ```bash
   git clone https://github.com/watoli/AscendingRL.git
   ```

2. 进入目录：
   ```bash
   cd AscendingRL
   ```

3. 安装依赖：
   ```bash
   pip install -r scripts/requirements.txt
   ```

## 覆盖的算法

- Q-Learning
- 深度Q网络（DQN）
- 策略梯度
- 行动者-评论家方法
- 近端策略优化（PPO）
- 以及更多...

每个算法都在Gymnasium环境下实现了示例。

## 资源

为了获取更多信息和资源，我们推荐以下网站：

- [强化学习环境升级 - 从gym到Gymnasium](https://blog.csdn.net/lusing/article/details/129272794)
- [深度强化学习：gymnasium下创建自己的环境（保姆式教程）](https://blog.csdn.net/qq_36592770/article/details/133325814)
- [RLChina 强化学习社区](http://rlchina.org/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [深度强化学习实验室](https://www.deeprlhub.com/)

这些网站提供了有价值的教程和社区支持，适合任何对RL和Gymnasium感兴趣的人员。

## 贡献

我们欢迎社区的贡献！如果你有关于新算法的想法或改进现有算法的意见，欢迎开立问题或提交拉取请求。

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

---

感谢您选择AscendingRL作为您的强化学习之旅的资源。希望您发现它有用并且具有启发性！

---

**免责声明**：这里实现的算法旨在用于教育目的。用户在实际场景中应用这些技术之前，应了解其理论基础。