import numpy as np
import math

class DynamicModel:
    def __init__(self, sampling_time, wheelbase=2.0):
        self.dt = sampling_time
        self.L = wheelbase 
        
        # 5维状态，2维控制
        self.nx = 5 # [x, y, phi, v, psi]
        self.nu = 2 # [u1, u2] (加速度, 转向角速度)

    def sim_forward_step(self, x_n, u_n):
        """
        非线性模型的前向欧拉积分推演 (5D)
        """
        x, y, phi, v, psi = x_n[0], x_n[1], x_n[2], x_n[3], x_n[4]
        u1, u2 = u_n[0], u_n[1]

        x_next = x + v * math.cos(phi) * self.dt
        y_next = y + v * math.sin(phi) * self.dt
        phi_next = phi + (v / self.L) * math.tan(psi) * self.dt
        v_next = v + u1 * self.dt
        psi_next = psi + u2 * self.dt

        return np.array([x_next, y_next, phi_next, v_next, psi_next])

    def get_linearized_matrices(self, x_ref, u_ref):
        """
        在参考点 (x_ref, u_ref) 处计算一阶泰勒展开的线性化矩阵 Ad, Bd, cd
        使得 x_{k+1} ≈ Ad * x_k + Bd * u_k + cd
        """
        phi = x_ref[2]
        v = x_ref[3]
        psi = x_ref[4]
        
        # 避免 psi 接近 90 度时 cos(psi) 为 0 导致奇异
        if abs(math.cos(psi)) < 1e-3:
            psi = math.copysign(math.pi/2 - 1e-3, psi)

        # 1. 计算连续时间的雅可比矩阵 Ac = df/dx, Bc = df/du
        Ac = np.zeros((self.nx, self.nx))
        # 偏导数 (关于 phi)
        Ac[0, 2] = -v * math.sin(phi)
        Ac[1, 2] = v * math.cos(phi)
        
        # 偏导数 (关于 v)
        Ac[0, 3] = math.cos(phi)
        Ac[1, 3] = math.sin(phi)
        Ac[2, 3] = math.tan(psi) / self.L
        
        # 偏导数 (关于 psi)
        Ac[2, 4] = v / (self.L * (math.cos(psi) ** 2))

        Bc = np.zeros((self.nx, self.nu))
        # 偏导数 (关于 u1)
        Bc[3, 0] = 1.0
        # 偏导数 (关于 u2)
        Bc[4, 1] = 1.0 

        # 2. 前向欧拉离散化：Ad = I + Ac * dt, Bd = Bc * dt
        Ad = np.eye(self.nx) + Ac * self.dt
        Bd = Bc * self.dt

        # 3. 计算仿射常数项 cd
        # x_dot_ref 是参考点处的非线性导数
        x_dot_ref = np.array([
            v * math.cos(phi),
            v * math.sin(phi),
            (v / self.L) * math.tan(psi),
            u_ref[0],
            u_ref[1] 
        ])
        
        # cd = (x_dot_ref - Ac * x_ref - Bc * u_ref) * dt
        cd = (x_dot_ref - Ac @ x_ref - Bc @ u_ref) * self.dt

        return Ad, Bd, cd