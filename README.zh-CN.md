<h1 align="center">Pi05-VLABench</h1>

<p align="center">
  <strong>基于 π0.5 的 VLABench 模仿学习与强化学习</strong><br>
  专家示范 · LeRobot 数据集 · 多任务 SFT · PPO · 闭环评测
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;|&nbsp; <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/Turbo176/Pi05-VLABench/actions/workflows/verify-evidence.yml"><img src="https://github.com/Turbo176/Pi05-VLABench/actions/workflows/verify-evidence.yml/badge.svg" alt="实验记录校验"></a>
</p>

<p align="center">
  <a href="#demos">🎬 演示视频</a> ·
  <a href="#results">📊 评测结果</a> ·
  <a href="#training">📈 训练曲线</a> ·
  <a href="#data">🦾 任务与数据</a><br>
  <a href="#pipeline">🧩 技术流程</a> ·
  <a href="#engineering">🔍 工程验证</a> ·
  <a href="#quickstart">🚀 快速开始</a> ·
  <a href="#experiments">🧪 实验与复现</a> ·
  <a href="#credits">📚 引用与许可</a>
</p>

基于官方 VLABench 任务和控制器的机器人学习项目。已完成的水果选择流程包括：专家数据验证、共享水果与插花任务的 π0.5 SFT、自主闭环评测，以及从同一个 SFT checkpoint 出发的 PPO 后训练。

| 训练数据 | 共享策略 | 强化学习更新 | 评测 |
| :---: | :---: | :---: | :---: |
| **64 条 episode / 9,904 帧** | **水果 + 插花 SFT** | **224 个 actor 参数张量发生变化** | **40 个配对的留出水果场景** |

项目围绕官方机器人控制器完成数据验证、模型适配、SFT 到 RL 的接口对齐和配对评测。目前跑通的学习策略全流程是水果选择；插花与冰箱取饮料仍待完成。[系统设计与实现](docs/system-design.md)

<a id="demos"></a>

## 🎬 水果选择自主策略演示

三个视频使用同一个留出场景，随机种子为 6005，原始指令为 **“Put the orange into the plate_seen”**，即把橙子放入盘子。视频均来自重新运行的策略自主闭环执行。这一场景用于展示 SFT/PPO 的成功行为，整体结果包含全部 40 场。

| 已发布的 VLABench 参考模型 | 共享多任务 SFT180 | 从同一 SFT checkpoint 训练的 PPO |
| :---: | :---: | :---: |
| [![参考模型执行](media/policy/fruit-recovery-v2-base-6005.gif)](media/policy/fruit-recovery-v2-base-6005.mp4?raw=true) | [![SFT 达到官方成功条件](media/policy/fruit-recovery-v2-sft-6005.gif)](media/policy/fruit-recovery-v2-sft-6005.mp4?raw=true) | [![PPO 达到官方成功条件](media/policy/fruit-recovery-v2-ppo-6005.gif)](media/policy/fruit-recovery-v2-ppo-6005.mp4?raw=true) |
| 失败 · 600 个控制步 | 成功 · 133 个控制步 | 成功 · 145 个控制步 |

[视频一致性与选取记录](docs/experiments/2026-11-24/fruit-recovery-v2/evidence/video-verification.json) · [前期配对示例](docs/experiments/2026-11-22/fruit-completion-v1/README.md)

MP4 包含完整执行过程，保留 10 Hz 的原始时序，分辨率为 1024 × 1024；GIF 以正常速度、10 fps 预览最多 15 秒。每段录像的初始策略图像和完整动作序列哈希均与计分 episode 一致。这些视频展示一个场景，整体结果见下表。

<a id="results"></a>

## 📊 闭环评测结果

评测使用 40 个配对的留出水果场景，苹果、香蕉、橙子和梨各有 10 个自然初始化的目标。所有策略使用相同的初始化、策略相机、语言指令、官方 Franka 控制器和 600 个控制步预算。控制频率为 10 Hz，每次观测预测 10 个动作，执行其中 5 个后重新读取观测。先根据开发集冻结 PPO checkpoint，再查看参考模型与 SFT 的测试结果。

| 策略 | 官方判定成功数 | 成功率 | Wilson 95% 置信区间 |
| --- | ---: | ---: | ---: |
| 已发布的 VLABench 参考模型 | 20/40 | 50% | 35.2–64.8% |
| 共享多任务 SFT180 | 17/40 | 42.5% | 28.5–57.8% |
| SFT180 后的水果 PPO1 | 16/40 | 40% | 26.3–55.4% |

参考模型是上游已在 VLABench 上微调的 π0.5 模型。本轮验证了可执行的 SFT 到 PPO 流程，**尚未获得整体成功率提升**。相较 SFT180，PPO 新增 5 个成功场景，同时丢失 6 个。官方成功条件检查目标水果是否进入盘子区域，不要求松爪或静置稳定。[分水果结果、失败类型与评测协议](docs/experiments/2026-11-24/fruit-recovery-v2/README.md)

![成功率置信区间与各类水果结果](docs/experiments/2026-11-24/fruit-recovery-v2/results/success-rates.png)

保存的 PPO checkpoint 使用了 **6,897 个训练控制步**，包含 critic 预热，共执行 **24 次 actor 和 122 次 critic 优化器更新**。独立加载权重对比确认 **224 个 actor 参数张量发生变化**，包括动作专家 LoRA 与动作头。SFT180 共采样 1,440 个训练窗口，两者均为短实验，尚不能声称已训练收敛。[SFT 到 RL 的交接校验](docs/experiments/2026-11-24/fruit-recovery-v2/evidence/zero-update-handoff.json) · [Actor 参数更新审计](docs/experiments/2026-11-24/fruit-recovery-v2/results/saved-checkpoint-audit.json) · [全部配对场景结果](docs/experiments/2026-11-24/fruit-recovery-v2/results/paired-scenes.png)

<a id="training"></a>

## 📈 训练曲线

![完整 SFT、critic 预热与 PPO 训练记录](docs/experiments/2026-11-24/fruit-recovery-v2/results/training.png)

图中展示全部 180 次 SFT 更新、critic 预热、PPO 的策略与价值损失，以及采样前缀分数的更新保护指标，包含被回滚的更新。不同采样批次的损失存在波动，策略表现仍以闭环成功率为准。

[本次完整报告与开发集曲线](docs/experiments/2026-11-24/fruit-recovery-v2/README.md) · [前期运动分析](docs/experiments/2026-11-23/training-and-motion/README.md) · [奖励设计与参考文献](docs/fruit-reward-review.md)

<a id="data"></a>

## 🦾 任务技能与示范数据

官方专家覆盖三种操作技能：选择并放置水果、将花与花瓶对齐并插入，以及打开冰箱取出饮料。下面的视频均为在新 MuJoCo 仿真中执行保存动作得到的**专家动作回放**。上方的学习策略结果使用另一套位姿数据集，其统计见下文。

| 🍎 水果选择 | 🌷 插花 | 🧃 冰箱取饮料 |
| :---: | :---: | :---: |
| [![水果专家动作回放](media/demos/select-fruit-preview.gif)](media/demos/select-fruit-replay.mp4?raw=true) | [![插花专家动作回放](media/demos/insert-flower-preview.gif)](media/demos/insert-flower-replay.mp4?raw=true) | [![冰箱取饮料专家动作回放](media/demos/fridge-drink-preview.gif)](media/demos/fridge-drink-replay.mp4?raw=true) |
| [回放审计](docs/stage1-action-replay.json) | [回放审计](docs/stage2-insert-flower-replay.json) | [回放审计](docs/stage2-fridge-rgb-replay.json) |

![当前 SFT 数据划分与训练指令分布](docs/experiments/2026-11-23/project-overview/dataset-coverage.png)

当前共享 SFT 使用 47 条水果、17 条插花训练 episode，另有独立的 5 条验证 episode，共 744 帧。水果数据包含 2 条专家纠正轨迹。早期专家演示所用的关节空间数据集包含 60 条 episode、16,520 帧，不计入当前 SFT 数据总数。[当前数据卡](docs/system-design.md#data-card) · [采集与导出记录](docs/stage2.md)

<a id="pipeline"></a>

## 🧩 技术流程

```mermaid
flowchart LR
    E[官方专家] --> R[动作回放]
    R --> D[LeRobot 数据集]
    D --> S[共享多任务 SFT]
    S --> H[RL 零更新交接校验]
    H --> P[水果 PPO]
    S --> V[配对闭环评测]
    P --> V
    B[已发布的参考模型] --> V
```

- **数据验证。** 从记录的初始化重新执行保存的动作，检查官方成功条件，并按完整 episode 与初始化条件隔离数据划分。
- **策略接口。** 使用前方、右侧和腕部 RGB，以及原始语言指令。实测末端位姿与夹爪状态用于共享变换和动作解码。专家目标几何真值、奖励与成功标签不进入策略输入。
- **动作对齐。** SFT 与 RL 保持上游的状态条件设置、归一化、位姿解码、10 动作预测范围和每次执行 5 个控制步的规则。零更新交接校验比较预测、动作对数概率和一次完整闭环执行。
- **实验记录。** 保留 checkpoint 哈希、固定配置、已登记的源码哈希、原始结果、actor 梯度与参数变化记录。

位姿策略动作包含以米为单位的 XYZ、以弧度为单位的欧拉角和夹爪坐标，并通过官方控制器解码执行。已发布配置使用 `discrete_state_input=False`；状态用于位姿解码，没有额外加入新的模型状态输入。[实际策略输入输出与模型卡](docs/system-design.md#policy-and-control-contract)

<a id="engineering"></a>

## 🔍 工程验证

| 问题 | 已实现的检查 | 记录 |
| --- | --- | --- |
| 数据标签经过归一化与控制器解码后是否一致？ | 所有标签窗口完成往返变换检查；选定专家标签经同一控制器执行并达到官方成功条件。 | [变换与回放审计](docs/experiments/2026-11-22/pose-robustness-v2/training-contract-audit.json) |
| SFT 加载到 RL 后，尚未训练时行为是否变化？ | 3 个真实观测窗口完全一致；一次配对闭环执行重复成功。 | [零更新交接校验](docs/experiments/2026-11-24/fruit-recovery-v2/evidence/zero-update-handoff.json) |
| PPO 是否更新了 VLA 本身？ | 将保存的策略张量与准确的 SFT 父 checkpoint 比较，并单独检查策略与 critic 的优化器更新。 | [224 个 actor 张量的更新审计](docs/experiments/2026-11-24/fruit-recovery-v2/results/saved-checkpoint-audit.json) |
| 突然的动作变化出现在哪里？ | 对齐控制前的实测位姿与动作指令，使用四元数比较旋转；在一个动作块内部观察到 51.3 mm 的目标跳变。 | [运动分析；修复待验证](docs/experiments/2026-11-23/training-and-motion/README.md) |

项目复用上游模型、专家技能、任务与底层控制。仓库中的工作包括数据对齐、训练与闭环执行集成、验证和评测。[模块说明、配置、成本与待完成工作](docs/system-design.md)

<a id="quickstart"></a>

## 🚀 快速开始

### 无需 GPU 核验实验记录

在仓库根目录运行。Python 3.9+ 即可校验导出的记录并重新计算配对得分，无需安装额外依赖：

```bash
python scripts/review_fruit_experiment.py
```

该命令核验前期实验的 45 个保留文件、11 个源码与配置哈希，以及全部 57 条普通与语言评测 episode。本次 40 场对照另有只依赖 Python 标准库的校验，在仓库根目录的 Bash 中运行：

```bash
report=docs/experiments/2026-11-24/fruit-recovery-v2
python "$report/analyze.py" --verify-only --cases "$report/evidence/test-cases.jsonl" \
  --base "$report/evidence/test-reference.jsonl" --sft "$report/evidence/test-sft.jsonl" \
  --ppo "$report/evidence/test-ppo.jsonl" --selection "$report/evidence/final-selection.json" \
  --output "$report/results"
```

新命令会重新计算全部 **120 条原始测试结果**，核验配对接口、文件哈希、已保存的策略与 critic 更新记录，以及环境交互成本。它读取导出的实验记录；独立加载不随仓库提交的模型权重属于另一项检查。[复现命令](docs/experiments/2026-11-24/fruit-recovery-v2/commands.md)

### 运行官方专家示范

以下命令已在 WSL2 的 Ubuntu 22.04 中验证。按[安装指南](docs/stage1.md#setup-and-run)准备固定版本的仿真器、官方资产子集和独立 Python 环境，在仓库根目录运行，并使用新的输出目录。

```bash
bash scripts/run_wsl.sh probe --task select_fruit --seed 42
bash scripts/run_wsl.sh collect --task select_fruit --seed 42 --output runs/select_fruit-42
bash scripts/run_wsl.sh replay runs/select_fruit-42/episode.h5
```

`expert.mp4` 记录专家执行；`action_replay.mp4` 记录新仿真中执行保存动作的过程。模型训练与评测需要实验报告中单独说明的 GPU 环境。

<a id="experiments"></a>

## 🧪 实验与复现

| 记录 | 内容 |
| --- | --- |
| [40 场水果对照实验](docs/experiments/2026-11-24/fruit-recovery-v2/README.md) | 奖励对照、SFT180 → PPO、均衡配对测试、成本与原始记录 |
| [前期水果实验](docs/experiments/2026-11-22/fruit-completion-v1/README.md) | 原始 SFT90/PPO checkpoint、10 场测试与语言评测 |
| [训练与运动分析](docs/experiments/2026-11-23/training-and-motion/README.md) | 损失与梯度曲线、时间对齐的运动测量、可复现绘图命令 |
| [系统设计及模型卡与数据卡](docs/system-design.md) | 实现范围、具体接口、训练范围、成本与需求完成情况 |
| [CPU 校验与总览图表](docs/experiments/2026-11-23/project-overview/README.md) | 可执行完整性检查、原始结果汇总与数据覆盖情况 |
| [奖励设计分析](docs/fruit-reward-review.md) | 实际奖励、参考文献，以及尚未训练验证的分阶段奖励方案 |
| [专家数据与 LeRobot 导出](docs/stage2.md) | 采集、回放检查、数据集字段与划分审计 |
| [前期实验](docs/experiments/2026-11-22/pose-robustness-v2/README.md) | 早期接口、排查实验及其独立结果 |

[脚本入口说明](scripts/README.md)列出采集、转换、SFT、PPO、评测与分析的入口。`configs/` 保留固定的水果 PPO 和评测设置，以及独立参考模型评测器的默认控制设置。`media/` 存放三种专家动作回放与最终配对策略视频。模型权重、原始数据集、资产、缓存和凭据不提交 Git。

<a id="credits"></a>

## 📚 引用与许可

基于 VLABench、MuJoCo、LeRobot、OpenPI 和 RLinf 构建。项目代码使用 MIT 许可证；资产与模型 checkpoint 遵循各自的上游条款。[固定版本与引用](docs/upstream.md) · [MIT 许可证](LICENSE)
