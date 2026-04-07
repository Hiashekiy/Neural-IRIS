import numpy as np
from collections import deque

def find_random_path_bfs(binary_map, min_dist=20.0, margin=2):
    """
    基于广度优先搜索 (BFS) 生成两点之间的随机路径。
    binary_map: 二值化地图, 1 表示障碍物, 0 表示空闲区域
    min_dist: 起点和终点之间的最小距离要求 (以像素为单位)
    margin: 膨胀边距，防止贴近障碍物
    """
    h, w = binary_map.shape
    
    # 1. 膨胀地图，保证起始位置绝对安全
    dilated_map = binary_map.copy()
    if margin > 0:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2+1, margin*2+1))
        # 因为1是障碍物，对1做膨胀就是让障碍区变大
        dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
        
    # 找到所有安全的候选点
    free_ys, free_xs = np.where(dilated_map == 0)
    if len(free_ys) < 2:
        return None  # 找不到足够的安全点
        
    free_points = list(zip(free_xs, free_ys))
    
    # 重试几次，直到找到包含合理距离终点的起点
    for _ in range(10):
        # 随机选择一个起点
        start_idx = np.random.randint(len(free_points))
        start_pt = free_points[start_idx]
        
        # 2. 从起点开始运行 BFS，找到所有可达点和路径
        queue = deque([start_pt])
        visited = {start_pt: start_pt}  # 记录上一个节点用于回溯
        
        # 能够达到最小距离要求的终点集合
        valid_targets = []
        
        while queue:
            curr = queue.popleft()
            cx, cy = curr
            
            # 使用欧氏距离判断是否满足最小距离要求
            d = np.hypot(cx - start_pt[0], cy - start_pt[1])
            if d >= min_dist:
                valid_targets.append(curr)
                
            # 探索4连通邻居
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx_pt, ny_pt = cx + dx, cy + dy
                neighbor = (nx_pt, ny_pt)
                
                # 检查边界和安全性、是否已访问
                if 0 <= nx_pt < w and 0 <= ny_pt < h:
                    if dilated_map[ny_pt, nx_pt] == 0 and neighbor not in visited:
                        visited[neighbor] = curr
                        queue.append(neighbor)
                        
        if valid_targets:
            # 在满足距离要求的节点中随机挑选一个作为终点
            target_pt = valid_targets[np.random.randint(len(valid_targets))]
            
            # 3. 回溯生成路径
            path = []
            curr = target_pt
            while curr != start_pt:
                path.append(curr)
                curr = visited[curr]
            path.append(start_pt)
            path.reverse()
            
            # 由于BFS生成的路径是像素级的，点非常密集，可以进行下采样或使用全部点
            return tuple(start_pt), tuple(target_pt), path
            
    return None
