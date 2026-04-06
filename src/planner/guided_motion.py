import math
import numpy as np

def get_guide_point_at_s(target_s, path_points, cumulative_dists, total_len):
    """
    [核心函数] 根据里程 s 获取路径上的坐标 (x, y) 和切线角度 angle
    这一步替代了原来的 find_nearest_point_on_path
    """
    # 1. 限制 s 范围，防止越界
    target_s = np.clip(target_s, 0.0, total_len)
    
    # 2. 二分查找 s 所在的线段索引
    # cumulative_dists[i] 是第 i 个点的里程
    # searchsorted 找到满足 cumulative_dists[i-1] <= s < cumulative_dists[i] 的 i
    idx = np.searchsorted(cumulative_dists, target_s) - 1
    idx = max(0, min(idx, len(path_points) - 2))
    
    # 3. 获取线段起终点信息
    s_start = cumulative_dists[idx]
    s_end = cumulative_dists[idx+1]
    seg_len = s_end - s_start
    
    p_start = np.array(path_points[idx])
    p_end = np.array(path_points[idx+1])
    
    # 4. 线性插值计算坐标
    if seg_len < 1e-6:
        # 防止除以零（两点重合的情况）
        pos = p_start
        # 角度沿用上一段或默认为0，这里简单处理
        delta = p_end - p_start # 此时delta接近0，角度可能不准，但很少见
    else:
        ratio = (target_s - s_start) / seg_len
        pos = p_start + (p_end - p_start) * ratio
    
    # 5. 计算切线角度
    delta = p_end - p_start
    angle = math.atan2(delta[1], delta[0])
    
    return pos, angle

def simulate_guided_motion(current_s, path_points, cumulative_dists, velocity, sample_time, num_steps):
    """
    基于里程 s 的参考轨迹生成 (彻底解决转角卡死问题)
    
    :param current_s: 当前车辆在路径上的进度 s (由 Environment 维护)
    :param path_points: 路径点列表
    :param cumulative_dists: 路径点的累积里程数组
    :param velocity: 期望速度 (v_ref)
    :param sample_time: dt
    :param num_steps: 预测步数 (Horizon)
    """
    positions = []
    angles = []
    
    # 获取总长度
    total_len = cumulative_dists[-1]
    
    # 生成未来 N+1 个点 (包含当前时刻 k=0)
    # s_k = s_current + v * k * dt
    for k in range(num_steps + 1):
        future_s = current_s + velocity * (k * sample_time)
        
        # 如果超出终点，就停在终点
        if future_s > total_len:
            future_s = total_len
            
        pos, ang = get_guide_point_at_s(future_s, path_points, cumulative_dists, total_len)
        
        positions.append(pos)
        angles.append(ang)
        
    return positions, angles