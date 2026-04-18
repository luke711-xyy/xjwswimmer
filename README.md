# xjwswimmer

这个目录是一个独立的小型强化学习实验工程，用来训练一组二维连杆机器人形成类似纤毛/鞭毛的周期摆动，通过流体作用推动小球在 `x` 方向产生净位移。

当前版本的任务定义已经改成：

- 奖励按小球每一步真实 `x` 位移的有符号增量累计
- 不再使用固定 `target_x` 成功终止
- 不再惩罚 `y` 方向位移
- 相邻连杆内部夹角必须 `>= 60°`
- 自交叉或夹角超限时，状态回滚到上一帧并给予固定负奖励
- 训练输出自动保存到带时间戳的 `runs_cluster/` 目录
- TensorBoard 中记录环境统计均值和训练指标
- 训练时终端每 10 步打印一次小球实时位置

## 项目结构

```text
F:\fyp\xjw
├── Cluster_Env_ran.py
│   自定义 Gymnasium 环境，包含：
│   - 连杆机器人构型更新
│   - regularized stokeslet 流体计算
│   - 小球动力学
│   - 奖励函数
│   - 自交叉 / 最小夹角约束
│   - 渲染
│
├── Train_Cluster_ran.py
│   训练入口，包含：
│   - SAC 模型创建
│   - Monitor / Eval / Checkpoint / 自定义统计回调
│   - TensorBoard 日志
│   - 终端实时位置打印
│   - 自动创建时间戳训练目录和 run_config.json
│
├── Play.py
│   可视化 / 回放入口，默认：
│   - 自动读取最近一次训练目录
│   - 从 run_config.json 恢复环境参数
│   - 加载 best_model
│   - 生成 mp4 视频
│
├── runs_cluster/
│   训练后自动生成的运行目录，例如：
│   20260418_154500_sac_r3_l3/
│   ├── logs/
│   ├── models/
│   └── run_config.json
│
└── __pycache__/
    Python 自动生成的缓存目录，可忽略
```

## 这套代码相比原仓库额外需要的依赖

如果你之前只参考了原始仓库根目录下的 `requirements.txt`，那还不够。

这套 `F:\fyp\xjw` 脚本现在实际需要的核心依赖是：

- `python`
- `torch`
- `numpy`
- `gymnasium`
- `stable-baselines3`
- `pygame`
- `opencv-python`
- `tensorboard`

和原始仓库环境相比，最关键的差异是：

- 原仓库写的是 `gym`，这里实际用的是 `gymnasium`
- 这里必须安装 `stable-baselines3`
- 这里需要 `pygame` 做渲染
- 这里需要 `opencv-python` 导出视频
- 如果要看训练曲线，需要安装 `tensorboard`

## 从零开始创建 conda 环境

下面命令按 Windows + conda + CPU 训练写。

### 1. 创建虚拟环境

```powershell
conda create -n xjwswimmer python=3.10 -y
conda activate xjwswimmer
```

### 2. 安装 PyTorch

CPU 版本：

```powershell
pip install torch==2.1.0
```

如果你后面想切 CUDA 版本，再按你机器的 CUDA 版本改成官方对应安装命令。

### 3. 安装其余依赖

```powershell
pip install numpy==1.26.4
pip install gymnasium==0.29.1
pip install stable-baselines3==2.3.2
pip install pygame==2.5.2
pip install opencv-python==4.10.0.84
pip install tensorboard==2.16.2
```

### 4. 一次性安装写法

如果你想一次装完，也可以直接：

```powershell
pip install torch==2.1.0 numpy==1.26.4 gymnasium==0.29.1 stable-baselines3==2.3.2 pygame==2.5.2 opencv-python==4.10.0.84 tensorboard==2.16.2
```

## 进入项目目录

```powershell
cd /d F:\fyp\xjw
```

如果你在 PowerShell 里：

```powershell
Set-Location F:\fyp\xjw
```

## 开始训练

```powershell
python Train_Cluster_ran.py
```

训练开始后会自动创建：

```text
runs_cluster\YYYYMMDD_HHMMSS_sac_r3_l3\
```

其中包含：

- `models\best_model.zip`
- `models\check_model_XXXX_steps.zip`
- `models\final_model.zip`
- `logs\`
- `run_config.json`

训练过程中终端会每 10 步打印一次类似输出：

```text
step=12340 particle=(0.1820, 1.0940) dx=0.0017 invalid=None
```

## 查看 TensorBoard 曲线

在 `F:\fyp\xjw` 目录下执行：

```powershell
tensorboard --logdir .\runs_cluster
```

然后浏览器打开 TensorBoard 页面。

你可以重点看这些均值曲线：

- `rollout/ep_rew_mean`
- `rollout/ep_len_mean`
- `env/particle_x`
- `env/particle_y`
- `env/particle_dx_step`
- `env/particle_total_dx_episode`
- `env/invalid_state_rate`
- `env/min_internal_angle_deg`
- `env/self_intersection_rate`
- `env/min_angle_violation_rate`

## 可视化 / 回放

```powershell
python Play.py
```

默认行为：

- 自动扫描 `runs_cluster/`
- 选择最近一次训练目录
- 默认加载 `best_model`
- 使用该次训练保存的 `run_config.json` 恢复环境配置
- 导出视频文件

如果你想手动指定某次训练或某个模型，可以修改 `Play.py` 顶部：

```python
TARGET_RUN_NAME = "20260418_154500_sac_r3_l3"
TARGET_MODEL_NAME = "final_model"
```

或：

```python
TARGET_MODEL_NAME = "check_model_5000_steps"
```

## 说明

- 这套代码是 `F:\fyp\xjw` 下的独立实验脚本，不等同于原始仓库 README 里描述的完整层级强化学习主工程
- 当前默认配置为 `3` 个机器人、每个机器人 `3` 段连杆
- 当前算法为 `SAC`
- 当前默认设备为 CPU

