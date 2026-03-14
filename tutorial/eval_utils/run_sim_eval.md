# `run_sim_eval.py` — 仿真环境策略评估主脚本

## 文件概述

`run_sim_eval.py` 是 DreamZero 项目中执行**仿真策略评估**的主入口脚本。它基于 NVIDIA Isaac Lab 仿真框架构建评估环境，通过 WebSocket 客户端与远程策略服务器通信，驱动机器人在仿真场景中执行指定任务，并将评估过程录制为视频。该脚本整合了观测提取、图像预处理、动作块（action chunking）执行以及夹爪二值化等完整的评估管道。

## 使用方式

脚本文档注释中描述了完整的使用流程：

1. 下载仿真资源并解压到项目根目录
2. 在单独终端启动策略服务器（需设置 `XLA_PYTHON_CLIENT_MEM_FRACTION` 避免 JAX 占用全部 GPU 显存）
3. 运行评估脚本：`python run_eval.py --episodes 10 --headless`

## 关键代码解析

### 1. DreamZeroJointPosClient 类

这是整个评估管道的核心类，继承自 `sim_evals` 的 `InferenceClient` 抽象接口。

#### 初始化

```python
class DreamZeroJointPosClient(InferenceClient):
    def __init__(self,
                remote_host:str = "localhost",
                remote_port:int = 6000,
                open_loop_horizon:int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
```

关键参数：
- `remote_host` / `remote_port`：策略服务器地址，默认连接 `localhost:6000`
- `open_loop_horizon`：开环执行步数，默认为 8。即每次推理获取一个动作块后，连续执行 8 步才会再次查询策略服务器
- `session_id`：使用 UUID 为每个评估回合生成唯一标识，用于策略内部追踪历史

#### 观测提取

```python
def _extract_observation(self, obs_dict, *, save_to_disk=False):
    right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
    left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
    wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

    robot_state = obs_dict["policy"]
    joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
    gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()
```

该方法从 Isaac Lab 环境返回的观测字典中提取所需数据：
- **图像数据**：从 GPU tensor 转换为 numpy 数组，包括右侧外部相机、左侧外部相机和腕部相机
- **本体感知数据**：7 维关节位置和夹爪位置
- `[0]` 索引表示取第一个（也是唯一的）环境实例的数据
- `.clone().detach().cpu().numpy()` 是 PyTorch tensor 转 numpy 的标准安全操作链

#### 推理与动作块执行

```python
def infer(self, obs: dict, instruction: str) -> dict:
    curr_obs = self._extract_observation(obs)
    if (
        self.actions_from_chunk_completed == 0
        or self.actions_from_chunk_completed >= self.open_loop_horizon
    ):
        self.actions_from_chunk_completed = 0
        request_data = {
            "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
            "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
            "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
            "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
            "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
            "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
            "prompt": instruction,
            "session_id": self.session_id,
        }
        result = self.client.infer(request_data)
        actions = result["actions"] if isinstance(result, dict) else result
        self.pred_action_chunk = actions
```

这段代码实现了**动作块（action chunking）**机制，这是现代机器人策略中的重要技术：

1. **条件判断**：仅在以下两种情况下才向服务器请求新的推理：
   - 尚未执行过任何动作（`== 0`）
   - 当前动作块已执行完毕（`>= open_loop_horizon`）
2. **观测数据构造**：
   - 图像缩放至 `180x320`（使用 `resize_with_pad` 保持宽高比并填充）
   - 关节位置转为 `float64` 精度
   - 笛卡尔位置使用全零占位（因为使用的是关节位置控制模式）
   - 包含自然语言任务指令（`prompt`）和会话 ID
3. **结果验证**：断言检查动作块形状为 `(N, 8)`，即 7 个关节 + 1 个夹爪

#### 逐步动作执行与夹爪二值化

```python
action = self.pred_action_chunk[self.actions_from_chunk_completed]
self.actions_from_chunk_completed += 1

# binarize gripper action
if action[-1].item() > 0.5:
    action = np.concatenate([action[:-1], np.ones((1,))])
else:
    action = np.concatenate([action[:-1], np.zeros((1,))])
```

从动作块中按序取出当前步的动作，并将夹爪动作二值化：
- 模型输出的夹爪值是连续的，阈值 0.5 以上映射为 1（张开/闭合），以下映射为 0
- 这种二值化处理是因为实际夹爪控制通常只有开/关两个状态

### 2. 主函数与评估循环

```python
def main(
        episodes: int = 10,
        scene: int = 1,
        headless: bool = True,
        host: str = "localhost",
        port: int = 6000,
        ):
```

主函数参数通过 `tyro` 自动生成命令行接口：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `episodes` | 10 | 评估回合数 |
| `scene` | 1 | 场景编号（1-3） |
| `headless` | True | 是否无头模式运行（不显示渲染窗口） |
| `host` | localhost | 策略服务器地址 |
| `port` | 6000 | 策略服务器端口 |

#### Isaac Lab 环境初始化

```python
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

Isaac Lab 的启动必须在导入其他 IsaacLab 依赖模块之前完成。这里将启动代码放在函数内部（而非模块顶层），是为了避免与 `tyro` 的命令行参数解析冲突。

#### 场景与任务指令映射

```python
match scene:
    case 1:
        instruction = "put the cube in the bowl"
    case 2:
        instruction = "pick up the can and put it in the mug"
    case 3:
        instruction = "put the banana in the bin"
```

使用 Python 3.10 的 match 语法将场景编号映射为自然语言任务指令。目前支持三个预定义场景。

#### 评估主循环

```python
video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
video_dir.mkdir(parents=True, exist_ok=True)
video = []

with torch.no_grad():
    for ep in range(episodes):
        for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
            ret = client.infer(obs, instruction)
            if not headless:
                cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            video.append(ret["viz"])
            action = torch.tensor(ret["action"])[None]
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        client.reset()
        mediapy.write_video(video_dir / f"episode_{ep}.mp4", video, fps=15)
        video = []
```

评估循环的详细流程：

1. **视频存储目录**：按日期和时间自动创建目录结构 `runs/YYYY-MM-DD/HH-MM-SS/`
2. **推理与执行**：
   - 在 `torch.no_grad()` 上下文中运行，避免不必要的梯度计算
   - 每步调用 `client.infer()` 获取动作（内部自动处理动作块缓存）
   - 非无头模式下通过 OpenCV 实时显示相机视角
   - 动作转换为 PyTorch tensor 并添加 batch 维度 `[None]` 后传入环境
3. **终止判断**：当环境返回 `terminated` 或 `truncated` 信号时提前结束当前回合
4. **回合重置**：每个回合结束后调用 `client.reset()` 清除策略状态，并将视频帧保存为 MP4 文件（15 FPS）
5. **注意**：环境初始化时调用了两次 `env.reset()`，注释说明第二次渲染周期是为了正确加载材质

## 核心类/函数

| 类/函数 | 用途 |
|---------|------|
| `DreamZeroJointPosClient` | 推理客户端，封装观测提取、动作块管理、图像预处理和夹爪二值化 |
| `DreamZeroJointPosClient.__init__()` | 初始化 WebSocket 连接和动作块状态 |
| `DreamZeroJointPosClient.infer(obs, instruction)` | 执行推理，管理动作块缓存，返回动作和可视化图像 |
| `DreamZeroJointPosClient.reset()` | 重置内部状态（动作块计数器、会话 ID 等） |
| `DreamZeroJointPosClient.visualize(request)` | 返回模型视角的三相机拼接可视化图像 |
| `DreamZeroJointPosClient._extract_observation(obs_dict)` | 从 Isaac Lab 环境观测中提取图像和本体感知数据 |
| `main(episodes, scene, headless, host, port)` | 评估主函数，初始化环境并运行评估循环 |

## 与其他模块的关系

- **使用** `policy_client.WebsocketClientPolicy`：作为底层网络通信组件
- **继承** `sim_evals.inference.abstract_client.InferenceClient`：遵循仿真评估的抽象客户端接口
- **依赖** `sim_evals.environments`：注册 Isaac Lab 仿真环境（`"DROID"` 环境）
- **依赖** `isaaclab` / `isaaclab_tasks`：NVIDIA Isaac Lab 仿真框架，提供物理引擎和环境配置
- **使用** `openpi_client.image_tools`：图像缩放与填充工具
- **使用** `tyro`：自动将 `main()` 函数签名转换为命令行接口

## 总结

`run_sim_eval.py` 是 DreamZero 仿真评估的完整入口脚本，它将所有组件串联在一起：Isaac Lab 提供仿真环境，`WebsocketClientPolicy` 提供网络通信，`DreamZeroJointPosClient` 提供评估逻辑。核心设计亮点包括：

1. **动作块（Action Chunking）机制**：一次推理返回多步动作，减少推理频率，提高执行效率
2. **观测预处理管道**：从 GPU tensor 到 numpy，图像缩放填充，数据类型转换
3. **夹爪二值化**：将连续夹爪输出转为离散控制信号
4. **自动视频录制**：按时间戳组织的评估视频，便于事后分析
5. **延迟导入**：Isaac Lab 模块在函数内部导入，避免与命令行参数解析冲突
