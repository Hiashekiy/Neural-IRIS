# MPC 在线注入障碍实验实施文档

本文档对应实验脚本：

```text
scripts/test/mpc/test_dynamic_injected_obstacles.py
```

该实验用于验证 Neural-IRIS 生成的局部凸安全约束在闭环 MPC 轨迹规划中的可用性。实验过程中车辆沿全局参考路径行驶，不进行额外局部路径重规划；障碍物会在仿真运行过程中被在线注入到车辆前方路径附近，MPC 需要结合实时占据地图和 Neural-IRIS 半空间约束完成避障。

## 1. 实验目标

本实验主要回答以下问题：

1. Neural-IRIS 是否能够在闭环控制过程中，根据更新后的局部占据地图快速生成可用于 MPC 的局部凸安全区域。
2. MPC 是否能够直接使用 Neural-IRIS 输出的半空间约束，在不重新规划全局路径的情况下绕开在线注入障碍。
3. 在线批量生成预测时域安全约束的耗时是否满足 10 Hz 闭环控制频率。

需要注意，当前脚本中的“动态障碍”指的是仿真过程中在线新增到占据地图中的障碍物。障碍物注入后位置固定，并在后续控制周期持续存在；它并不是带速度状态和轨迹预测的移动障碍物。

## 2. 相关代码模块

| 模块 | 作用 |
| --- | --- |
| `scripts/test/mpc/test_dynamic_injected_obstacles.py` | 实验入口、地图加载、episode 循环、在线障碍注入、渲染与结果统计 |
| `src/planner/mpc_env.py` | MPC 闭环环境、随机参考路径生成、状态推进、碰撞检测、成功条件判断 |
| `src/planner/planner.py` | HPIPM 版本 MPC 求解器、Neural-IRIS 批量安全区域生成、半空间约束注入 |
| `src/planner/planner_osqp.py` | OSQP 版本 MPC 求解器，接口与核心逻辑相近 |
| `src/planner/car_model.py` | 五维车辆运动学模型与一阶线性化 |
| `config/config.json` | 地图、车辆参数、求解器后端、Neural-IRIS 接口配置 |

## 3. 实验环境与依赖

实验在项目根目录下运行：

```bash
cd d:\ProjectDirectory\Neural-IRIS
```

Python 依赖由项目 `requirements.txt` 管理。实验脚本还会使用以下主要库：

| 依赖 | 用途 |
| --- | --- |
| `numpy` | 数值计算 |
| `opencv-python` | 地图读取、障碍注入、车身碰撞检测 |
| `matplotlib` | 闭环过程可视化与视频帧捕获 |
| `scipy` | 半空间交点与凸包可视化 |
| `hpipm_python` | 默认 MPC 二次规划求解器 |
| Neural-IRIS Python 或 C++ 接口 | 批量生成局部安全半空间 |

默认配置文件中使用：

```json
{
  "planner_settings": {
    "solver": "hpipm",
    "neural_iris_interface": "cpp"
  }
}
```

因此实验默认优先使用 HPIPM 求解器和 Neural-IRIS C++ 后端。如果 C++ 后端不可用，代码会回退到 Python 后端。

## 4. 车辆与 MPC 参数

车辆参数来自 `config/config.json`：

| 参数 | 数值 | 说明 |
| --- | ---: | --- |
| `car_length_m` | 2.4 m | 车辆总长度 |
| `car_width_m` | 1.4 m | 车辆总宽度 |
| `wheelbase_m` | 1.4 m | 轴距 |
| `overhang_front_m` | 0.5 m | 前悬 |
| `safety_clearance_m` | 0.5 m | 额外安全裕度 |

MPC 采用五维车辆状态：

```text
x = [px, py, phi, v, delta]
```

控制输入为：

```text
u = [a, omega]
```

其中 `px, py` 为车辆平面位置，`phi` 为航向角，`v` 为速度，`delta` 为前轮转角，`a` 为加速度，`omega` 为前轮转角速度。

闭环控制与预测参数如下：

| 参数 | 数值 |
| --- | ---: |
| 控制周期 `dt` | 0.1 s |
| MPC 预测步长 `N` | 20 |
| 预测时长 | 2.0 s |
| episode 最大步数 | 1000 |
| 默认参考速度 | 8.0 m/s |

状态与控制边界如下：

| 变量 | 下界 | 上界 |
| --- | ---: | ---: |
| 位置 `px, py` | 地图下界 | 地图上界 |
| 速度 `v` | -10.0 m/s | 10.0 m/s |
| 前轮转角 `delta` | -0.6 rad | 0.6 rad |
| 加速度 `a` | -5.0 m/s^2 | 5.0 m/s^2 |
| 转角速度 `omega` | -0.6 rad/s | 0.6 rad/s |

## 5. 地图与参考路径生成

### 5.1 地图来源

当命令行参数 `--map random` 时，脚本从以下目录随机选择 `.map` 文件：

```text
data/street-map/val
```

`.map` 文件会被解析为二值占据地图：

| 栅格值 | 含义 |
| --- | --- |
| 0 | 自由空间 |
| 1 | 障碍物 |

环境内部使用的地图分辨率为：

```text
0.25 m/cell
```

### 5.2 参考路径生成

每个 episode reset 时，环境会随机生成一条全局参考路径：

1. 先对障碍物进行膨胀，膨胀边距为 2.0 m，用于避免起点和终点贴近障碍物。
2. 在膨胀后的自由空间中随机选取起点。
3. 使用 BFS 搜索可达区域，并随机选取距离起点至少 20.0 m 的终点。
4. 回溯得到栅格级路径。
5. 将路径转换到米制坐标，并按约 1.0 m 间距进行下采样。

该路径只作为全局参考路径使用。实验过程中即使新障碍物被注入到参考路径附近，也不会重新规划该全局路径。

## 6. 在线障碍注入机制

在线障碍注入发生在 episode 主循环中，触发条件为：

```text
step - last_obs_step >= 25
```

由于控制周期为 0.1 s，因此障碍物注入间隔为：

```text
25 * 0.1 s = 2.5 s
```

每次注入流程如下：

1. 获取车辆当前状态 `base_env.state`。
2. 在全局参考路径上寻找距离当前车辆位置最近的路径点。
3. 沿参考路径向前累计距离，随机选择 15.0 m 到 20.0 m 之间的前视距离。
4. 将该前视位置作为障碍物中心。
5. 随机采样圆形障碍物半径：

```text
r_px = randint(4, 9)
```

实际半径范围约为：

```text
4 * 0.25 m 到 8 * 0.25 m，即 1.0 m 到 2.0 m
```

6. 在占据地图 `base_env.map_obstacle` 中将圆形区域置为 1。

障碍物注入后不会被移除，后续 Neural-IRIS 局部地图裁剪、MPC 约束生成和碰撞检测都会使用更新后的占据地图。

## 7. Neural-IRIS 安全区域生成

每个控制周期中，MPC 会基于上一周期滚动得到的预测轨迹 `mpc_guess`，对预测时域内的状态位置批量生成局部安全区域。

### 7.1 局部地图裁剪

对于每个预测位置，系统从当前占据地图中裁剪以该位置为中心的局部区域：

| 参数 | 数值 |
| --- | ---: |
| 裁剪半径 | 10.0 m |
| 裁剪物理范围 | 20.0 m x 20.0 m |
| 网络输入尺寸 | 128 x 128 |

裁剪后的局部占据图会被重采样到 `128 x 128`，然后批量输入 Neural-IRIS。

### 7.2 半空间输出

Neural-IRIS 输出局部凸区域的半空间表示：

```text
A p <= b
```

其中：

| 符号 | 含义 |
| --- | --- |
| `p = [x, y]` | 车辆平面位置 |
| `A` | 半空间法向量矩阵 |
| `b` | 半空间边界向量 |

当前 MPC 固定每个阶段最多使用 12 个约束面：

```text
N_faces = 12
```

当 Neural-IRIS 输出的约束面数量超过 12 时，代码保留距离预测中心最近的 12 个约束面。

### 7.3 安全边界内缩

为了将车辆尺寸和额外安全裕度纳入约束，代码对半空间边界执行内缩：

```text
n_j^T p <= b_j - r_safe
```

其中：

```text
r_safe = car_width_m / 2 + safety_clearance_m
       = 1.4 / 2 + 0.5
       = 1.2 m
```

因此，MPC 约束限制的是车辆质心位置，但该质心约束已经考虑了车宽和安全裕度。

## 8. MPC 优化问题

每个控制周期中，MPC 求解一次二次规划问题。优化变量包括预测状态、控制输入以及安全约束松弛变量。

### 8.1 动力学约束

车辆连续模型为：

```text
dx/dt     = v cos(phi)
dy/dt     = v sin(phi)
dphi/dt   = v / L * tan(delta)
dv/dt     = a
ddelta/dt = omega
```

代码使用前向欧拉离散化，并在预测轨迹处进行一阶线性化：

```text
x_{k+1} = A_k x_k + B_k u_k + c_k
```

### 8.2 目标函数组成

目标函数主要由以下部分组成：

| 代价项 | 作用 |
| --- | --- |
| 横向误差代价 | 约束车辆贴近参考路径 |
| 纵向误差代价 | 约束车辆沿路径方向的推进误差 |
| 航向误差代价 | 约束车辆朝向接近参考路径切线 |
| 速度误差代价 | 约束车辆接近目标速度 |
| 前轮转角状态代价 | 抑制过大的绝对转角 |
| 加速度代价 | 抑制纵向急加速或急刹车 |
| 转角速度代价 | 抑制快速打方向 |
| 障碍物近邻斥力代价 | 在影响半径内增加远离障碍的趋势 |
| 松弛变量代价 | 强惩罚安全约束违反 |

默认脚本中的固定 action 为：

```text
[v_ref, w_obs, r_inf, w_lat, w_lon, w_steer]
= [8.0, 10.0, 1.0, 4.0, 5.0, 1.0]
```

但当前 `MPCEnv.step()` 实际读取方式为：

```text
v_ref       = action[0]
w_obs       = action[1]
R_influence = action[2]
w_lat       = action[3]
w_rate_steer= action[4]
```

也就是说，当前代码中第 6 个参数 `action[5]` 没有被使用，且脚本中名为 `w_lon` 的第 5 个参数实际被作为转角速度权重缩放使用。如果后续论文或实验表格需要精确列出权重，建议先统一脚本和环境中的 action 参数含义。

### 8.3 安全半空间约束

Neural-IRIS 生成的约束以一般线性约束形式注入 MPC：

```text
A_k [x_k, y_k]^T - s_k <= b_k
s_k >= 0
```

其中 `s_k` 为非负松弛变量。松弛变量用于避免极端情况下优化问题直接不可行，但其代价权重很大，因此求解器会优先寻找满足安全半空间的轨迹。

当前 HPIPM 实现中，半空间约束施加在 `k = 0 ... N-1` 的控制阶段。Neural-IRIS 会为 `N+1` 个预测状态生成区域，其中前 `N` 个区域用于约束注入，终端区域主要用于矩阵构造一致性和可视化扩展。

## 9. 实验运行命令

### 9.1 默认运行

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py
```

默认行为：

| 参数 | 默认值 |
| --- | --- |
| 地图 | `random` |
| episode 数量 | 5 |
| 渲染 | 开启 |
| Neural-IRIS 后端 | `cpp` |
| 参考速度 | 8.0 m/s |

### 9.2 关闭渲染进行批量测试

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --no-render --episodes 100
```

该模式适合统计成功率、平均步数、平均速度和规划耗时。

### 9.3 指定 Neural-IRIS 后端

使用 C++ 后端：

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --backend cpp
```

使用 Python 后端：

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --backend python
```

如果指定 C++ 后端但本地不可用，脚本会打印提示并回退到 Python 后端。

### 9.4 保存视频

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --save-video
```

视频保存目录：

```text
logs/videos
```

文件名格式：

```text
episode_<ep_idx>.mp4
```

### 9.5 调整控制参数

示例：

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py \
  --episodes 20 \
  --v-ref 6.0 \
  --w-obs 10.0 \
  --r-inf 1.0 \
  --w-lat 4.0 \
  --w-lon 5.0 \
  --w-steer 1.0
```

注意：如第 8.2 节所述，当前环境代码没有使用第 6 个 action 参数，因此 `--w-steer` 当前不会真正影响 MPC。

## 10. 可视化内容

开启渲染时，窗口中显示以下内容：

| 元素 | 含义 |
| --- | --- |
| 灰度背景 | 当前占据地图 |
| 绿色虚线 | 全局参考路径 |
| 绿色三角 | 起点 |
| 红色星形 | 终点 |
| 彩色实线 | 车辆已执行轨迹，颜色表示速度 |
| 青色多边形 | 当前车辆车身 |
| 黑色小矩形 | 车辆轮胎 |
| 红色箭头 | 当前航向 |
| 粉色曲线 | MPC 当前预测轨迹 |
| 浅红色多边形 | Neural-IRIS 生成的安全凸区域 |
| 蓝色虚线椭圆 | Neural-IRIS 椭圆预测结果 |

脚本中保留了 `CBF ACTIVE/Inactive` 的显示文字，但当前规划流程中 `u_nom` 和 `u_safe` 实际相同，并没有额外 CBF 控制器介入。因此论文图注和实验说明中不建议强调 CBF。

## 11. 结果统计

脚本每个 episode 结束后记录：

| 字段 | 含义 |
| --- | --- |
| `success` | 是否到达目标 |
| `steps` | episode 执行步数 |
| `avg_speed` | 平均速度 |
| `traj_x`, `traj_y` | 执行轨迹 |
| `velocities` | 每步速度 |
| `map` | episode 初始地图副本 |
| `path` | episode 初始参考路径 |
| `start`, `target` | 起点和终点 |
| `map_name` | 地图文件名 |

脚本最终打印：

```text
success  : 成功数量 / episode 数量
avg_steps: 平均步数
avg_speed: 平均速度
```

### 11.1 成功条件

当车辆当前位置与目标点距离小于 3.0 m 时，记为成功：

```text
dist_to_goal < 3.0
```

### 11.2 失败条件

发生以下情况之一时 episode 结束：

1. 车辆车身多边形与占据地图障碍物重叠，判定为碰撞。
2. 执行步数达到 `max_steps = 1000`，判定为超时截断。

### 11.3 建议补充统计指标

当前脚本默认只输出成功率、平均步数和平均速度。如果论文需要报告“碰撞率、障碍规避率、任务推进比例”，建议在脚本中额外统计：

| 指标 | 建议定义 |
| --- | --- |
| 碰撞率 | 碰撞 episode 数 / 总 episode 数 |
| 障碍规避率 | 未碰撞 episode 数 / 总 episode 数 |
| 任务推进比例 | episode 结束时沿参考路径的累计进度 / 参考路径总长度 |
| 平均 Neural-IRIS 耗时 | 每步 `nn_inference_time + polygon_generation_time` 的均值 |
| 平均 MPC 求解耗时 | 每步 `solve_time` 的均值 |
| 平均规划总耗时 | 每步 `total_time` 的均值 |

其中 `time_stats` 字段已经在 `infos` 中返回，包含：

```text
nn_inference_time
polygon_generation_time
matrix_build_time
solve_time
total_time
cpp_batch_call_time
python_postprocess_time
total_corridor_time
```

## 12. 推荐实验流程

### 12.1 单 episode 调试

用于确认环境、后端和可视化正常：

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --episodes 1
```

观察内容：

1. 地图是否正常加载。
2. 车辆是否沿绿色参考路径启动。
3. 控制 2.5 s 后是否出现在线注入障碍日志。
4. 粉色 MPC 预测轨迹是否随障碍发生偏移。
5. 浅红色安全凸区域是否跟随预测时域更新。

### 12.2 小批量功能验证

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --episodes 10 --no-render
```

用于确认无渲染情况下脚本稳定运行，并检查每步时间统计是否合理。

### 12.3 大批量统计实验

```bash
python scripts/test/mpc/test_dynamic_injected_obstacles.py --episodes 100 --no-render --backend cpp
```

用于论文表格统计。建议固定随机种子或保存每个 episode 的地图名、起终点、注入障碍参数和结果，保证统计结果可追溯。

## 13. 对照实验设计建议

论文中可设置以下对照：

| 组别 | 设置 | 目的 |
| --- | --- | --- |
| Baseline MPC | 不注入 Neural-IRIS 半空间约束 | 验证仅跟踪参考路径时无法处理在线障碍 |
| Neural-IRIS + MPC | 使用当前脚本默认设置 | 验证局部安全约束对避障能力的提升 |
| Python 后端 | `--backend python` | 对比 Python 推理接口耗时 |
| C++ 后端 | `--backend cpp` | 验证在线批量推理效率 |
| 不同参考速度 | 修改 `--v-ref` | 分析速度对成功率和碰撞率的影响 |

如果需要实现严格的 Baseline MPC，需要在 planner 中关闭或替换 Neural-IRIS 半空间约束。目前该脚本默认始终调用 Neural-IRIS 生成安全走廊。

## 14. 已知实现细节与注意事项

1. 当前脚本名称包含 `dynamic`，但障碍物注入后不移动。若论文强调“运动障碍物”，应使用或扩展 `test_moving_obstacles.py`。
2. 当前 action 参数命名与 `MPCEnv.step()` 的读取方式不完全一致，第 6 个 action 参数未使用。
3. 当前统计输出较简单，论文中的碰撞率、规避率、任务推进比例需要额外统计或由外部批处理脚本汇总。
4. 可视化中的 CBF 状态文字是遗留显示项，当前控制流程没有额外 CBF 修正。
5. Neural-IRIS 半空间约束使用松弛变量，因此极端情况下车辆可能轻微违反局部安全区域；这类违反会受到很高代价惩罚。
6. 障碍物会累积写入 `base_env.map_obstacle`，但返回结果中的 `map` 字段保存的是 episode 初始地图副本，不包含后续注入的全部障碍。

## 15. 论文记录建议

正式汇报实验结果时，建议至少记录以下内容：

1. 地图集合路径与地图数量。
2. episode 数量。
3. 随机种子或地图抽样方式。
4. 在线障碍注入间隔、前视距离范围和半径范围。
5. 车辆尺寸、安全边界内缩距离。
6. MPC 控制周期、预测时域、状态/控制边界。
7. Neural-IRIS 输入 patch 尺寸、局部裁剪范围、最大半空间面数。
8. 成功率、碰撞率、障碍规避率、任务推进比例。
9. Neural-IRIS 推理耗时、MPC 求解耗时、单步总规划耗时。
10. 失败案例分类，例如狭窄通道、注入过近、可行空间突然收缩、速度过高等。
