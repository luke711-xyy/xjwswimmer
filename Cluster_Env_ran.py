import math
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch

torch.set_printoptions(threshold=torch.inf)


class Cluster_Env_NRS_torch_C(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        control_period: float = 0.02,
        link_num: int = 3,
        num_robots: int = 3,
        N_per_Seg: int = 3,
        Q_per_Seg: int = 6,
        epsilon: float = 0.02,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 4000,
        reward_scale: float = 2000.0,
        invalid_penalty: float = -60.0,
        min_internal_angle_deg: float = 60.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_robots = num_robots
        self.link_num = link_num
        self.robot_spacing = 0.5

        print(
            f"Initialize environment with {self.num_robots} robots and one particle; "
            "objective is net positive x transport."
        )

        self.length = 1.0
        self.seg_length = float(self.length) / float(self.link_num)

        self.nodes_per_robot = 1 + N_per_Seg * self.link_num
        self.fg_traction_node_num = self.nodes_per_robot * self.num_robots

        self.q_nodes_per_robot = 1 + Q_per_Seg * self.link_num
        self.fg_stokeslet_node_num = self.q_nodes_per_robot * self.num_robots

        self.eps = epsilon
        self.CONTROL_PERIOD = control_period
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.invalid_penalty = invalid_penalty
        self.min_internal_angle_deg_limit = min_internal_angle_deg

        self.current_step = 0

        self.gnd_length = 8.0
        self.gnd_traction_node_num = int(1 + self.gnd_length * N_per_Seg * self.link_num)
        self.gnd_stokeslet_node_num = int(1 + self.gnd_length * Q_per_Seg * self.link_num)

        self.traction_node_num = self.gnd_traction_node_num + self.fg_traction_node_num
        self.stokeslet_node_num = self.gnd_stokeslet_node_num + self.fg_stokeslet_node_num

        total_dof = self.link_num * self.num_robots
        self.theta = torch.zeros(total_dof, dtype=torch.float32)
        self.theta_last = self.theta.clone()
        self.beta = torch.zeros(total_dof, dtype=torch.float32)
        self.beta_last = self.beta.clone()
        self.pos_x = torch.zeros((self.num_robots, self.link_num + 1, 2), dtype=torch.float32)
        self.pos_x_last = self.pos_x.clone()

        self.particle_pos = torch.zeros(2, dtype=torch.float32)
        self.particle_pos_last = torch.zeros(2, dtype=torch.float32)
        self.initial_particle_pos = torch.zeros(2, dtype=torch.float32)

        self.Q_Matrix = torch.zeros((self.traction_node_num * 3, total_dof), dtype=torch.float32)
        self.N_Matrix = torch.zeros(
            (self.stokeslet_node_num * 3, self.traction_node_num * 3), dtype=torch.float32
        )
        self.A_Matrix = torch.zeros(
            (self.traction_node_num * 3, self.traction_node_num * 3), dtype=torch.float32
        )
        self.w2f_Matrix = torch.zeros((self.traction_node_num * 3, total_dof), dtype=torch.float32)

        self.traction_nodes = torch.zeros(self.traction_node_num * 3)
        self.stokeslet_nodes = torch.zeros(self.stokeslet_node_num * 3)
        self.vel_vector = torch.zeros(total_dof, dtype=torch.float32)

        self.grid_dens = 10
        self.window_dpi = 300
        self.set_grid_field(self.grid_dens)
        self.particle_path = []

        self.invalid_state = False
        self.invalid_reason: Optional[str] = None
        self.min_internal_angle_deg = 180.0
        self.cross = 0

        self.action_space = gym.spaces.Box(
            low=-math.pi / 3,
            high=math.pi / 3,
            shape=(total_dof,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dof * 2 + 2,),
            dtype=np.float32,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        total_dof = self.link_num * self.num_robots
        self.theta = torch.zeros(total_dof, dtype=torch.float32)
        self.theta_last = self.theta.clone()
        self.beta = torch.zeros(total_dof, dtype=torch.float32)
        self.beta_last = self.beta.clone()
        self.pos_x = torch.zeros((self.num_robots, self.link_num + 1, 2), dtype=torch.float32)
        self.pos_x_last = self.pos_x.clone()

        self.Calculate_Robot_Config()

        self.particle_pos = torch.tensor([0.0, 1.1], dtype=torch.float32)
        self.particle_pos_last = self.particle_pos.clone()
        self.initial_particle_pos = self.particle_pos.clone()

        self.current_step = 0
        self.invalid_state = False
        self.invalid_reason = None
        self.min_internal_angle_deg = 180.0
        self.cross = 0
        self.vel_vector.zero_()

        self.particle_path = [self.particle_pos.clone()]
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs_theta = self.theta.numpy().astype(np.float32)
        obs_theta_vel = (self.theta - self.theta_last).numpy().astype(np.float32)
        obs_particle = self.particle_pos.numpy().astype(np.float32)
        return np.concatenate((obs_theta, obs_theta_vel, obs_particle))

    def step(self, action):
        self.theta_last = self.theta.clone()
        self.particle_pos_last = self.particle_pos.clone()

        omega = torch.as_tensor(np.asarray(action).reshape(-1), dtype=torch.float32)
        self.Step_Action(omega)

        if not self.invalid_state:
            particle_point = torch.tensor(
                [self.particle_pos[0], self.particle_pos[1], 0.0],
                dtype=torch.float32,
            )
            stokes_mat = self.regularize_stokeslet(particle_point)
            force_vec = torch.matmul(self.N_Matrix, torch.matmul(self.w2f_Matrix, self.vel_vector))
            vel_at_particle = torch.matmul(stokes_mat, force_vec)
            vel_at_particle = torch.clamp(vel_at_particle, -2.0, 2.0)

            self.particle_pos[0] += vel_at_particle[0] * self.CONTROL_PERIOD
            self.particle_pos[1] += vel_at_particle[1] * self.CONTROL_PERIOD

        self.particle_path.append(self.particle_pos.clone())

        particle_dx = float((self.particle_pos[0] - self.particle_pos_last[0]).item())
        particle_total_dx = float((self.particle_pos[0] - self.initial_particle_pos[0]).item())

        if self.invalid_state:
            reward = float(self.invalid_penalty)
        else:
            reward = float(self.reward_scale * particle_dx)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        info: Dict[str, Any] = {
            "particle_x": float(self.particle_pos[0].item()),
            "particle_y": float(self.particle_pos[1].item()),
            "particle_dx_step": particle_dx,
            "particle_total_dx": particle_total_dx,
            "invalid_state": bool(self.invalid_state),
            "invalid_reason": self.invalid_reason,
            "min_internal_angle_deg": float(self.min_internal_angle_deg),
        }
        if terminated or truncated:
            info["episode_particle_total_dx"] = particle_total_dx

        return self._get_obs(), reward, terminated, truncated, info

    def Check_Topology(self) -> bool:
        for i in range(self.num_robots):
            for j in range(self.link_num):
                for m in range(self.num_robots):
                    for n in range(self.link_num):
                        if not ((i == m) and (abs(j - n) < 1)):
                            pos_A = self.pos_x[i][j]
                            pos_B = self.pos_x[i][j + 1]
                            pos_C = self.pos_x[m][n]
                            pos_D = self.pos_x[m][n + 1]
                            AB = pos_B - pos_A
                            AC = pos_C - pos_A
                            AD = pos_D - pos_A
                            CA = pos_A - pos_C
                            CD = pos_D - pos_C
                            CB = pos_B - pos_C
                            cross_1 = (AC[0] * AB[1] - AC[1] * AB[0]) * (
                                AD[0] * AB[1] - AD[1] * AB[0]
                            )
                            cross_2 = (CA[0] * CD[1] - CA[1] * CD[0]) * (
                                CB[0] * CD[1] - CB[1] * CD[0]
                            )
                            if (cross_1 < -1e-8) and (cross_2 < -1e-8):
                                return True
        return False

    def _compute_min_internal_angle_deg(self) -> float:
        min_angle = 180.0
        for r in range(self.num_robots):
            for joint_idx in range(1, self.link_num):
                prev_vec = self.pos_x[r][joint_idx - 1] - self.pos_x[r][joint_idx]
                next_vec = self.pos_x[r][joint_idx + 1] - self.pos_x[r][joint_idx]

                prev_norm = torch.linalg.norm(prev_vec)
                next_norm = torch.linalg.norm(next_vec)
                if prev_norm <= 1e-8 or next_norm <= 1e-8:
                    continue

                cos_angle = torch.dot(prev_vec, next_vec) / (prev_norm * next_norm)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle_deg = float(torch.rad2deg(torch.acos(cos_angle)).item())
                min_angle = min(min_angle, angle_deg)
        return min_angle

    def Calculate_Robot_Config(self):
        prev_theta = self.theta.clone()
        prev_pos_x = self.pos_x.clone()
        prev_particle_pos = self.particle_pos.clone()

        center_offset = (self.num_robots - 1) * self.robot_spacing / 2.0
        for r in range(self.num_robots):
            start_idx = r * self.link_num
            self.theta[start_idx] = self.beta[start_idx]
            for i in range(1, self.link_num):
                self.theta[start_idx + i] = self.theta[start_idx + i - 1] + self.beta[start_idx + i]

            base_x = r * self.robot_spacing - center_offset
            self.pos_x[r][0][0] = base_x
            self.pos_x[r][0][1] = 0.0
            for j in range(1, self.link_num + 1):
                angle = self.theta[start_idx + j - 1]
                self.pos_x[r][j][0] = self.pos_x[r][j - 1][0] + self.seg_length * torch.cos(angle)
                self.pos_x[r][j][1] = self.pos_x[r][j - 1][1] + self.seg_length * torch.sin(angle)

        self.cross = 0
        self.invalid_state = False
        self.invalid_reason = None

        min_angle = self._compute_min_internal_angle_deg()
        self.min_internal_angle_deg = min_angle

        if self.Check_Topology():
            self.invalid_state = True
            self.invalid_reason = "self_intersection"
            self.cross = 1
        elif min_angle < self.min_internal_angle_deg_limit:
            self.invalid_state = True
            self.invalid_reason = "min_angle_violation"

        if self.invalid_state:
            self.theta = prev_theta
            self.beta = self.beta_last.clone()
            self.pos_x = prev_pos_x
            self.particle_pos = prev_particle_pos

        self.update_nodes_geometry()

    def update_nodes_geometry(self):
        N_per_Seg = (self.nodes_per_robot - 1) // self.link_num
        all_nodes = []
        for r in range(self.num_robots):
            nodes = torch.tensor(
                [[self.pos_x[r][0][0], self.pos_x[r][0][1], 0.0]],
                dtype=torch.float32,
            )
            inter_nodes = torch.zeros((3, N_per_Seg + 1), dtype=torch.float32)
            for i in range(self.link_num):
                p_start, p_end = self.pos_x[r][i], self.pos_x[r][i + 1]
                inter_nodes[0] = torch.linspace(p_start[0], p_end[0], N_per_Seg + 1)
                inter_nodes[1] = torch.linspace(p_start[1], p_end[1], N_per_Seg + 1)
                inter_nodes[2] = 0.0
                nodes = torch.cat((nodes, inter_nodes[:, 1:].T), dim=0)
            all_nodes.append(nodes)

        half_num = (self.gnd_traction_node_num - 1) // 2
        gnd_nodes = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        inter_nodes_l = torch.zeros((3, half_num + 1), dtype=torch.float32)
        inter_nodes_l[0] = torch.linspace(0.0, -self.gnd_length / 2, half_num + 1)
        inter_nodes_r = torch.zeros((3, half_num + 1), dtype=torch.float32)
        inter_nodes_r[0] = torch.linspace(0.0, self.gnd_length / 2, half_num + 1)
        gnd_nodes = torch.cat((gnd_nodes, inter_nodes_l[:, 1:].T), dim=0)
        gnd_nodes = torch.cat((gnd_nodes, inter_nodes_r[:, 1:].T), dim=0)
        total_nodes = torch.cat(all_nodes + [gnd_nodes], dim=0)
        self.traction_nodes = torch.flatten(total_nodes.T)

        Q_per_Seg = (self.q_nodes_per_robot - 1) // self.link_num
        all_q_nodes = []
        for r in range(self.num_robots):
            nodes = torch.tensor(
                [[self.pos_x[r][0][0], self.pos_x[r][0][1], 0.0]],
                dtype=torch.float32,
            )
            inter_nodes = torch.zeros((3, Q_per_Seg + 1), dtype=torch.float32)
            for i in range(self.link_num):
                p_start, p_end = self.pos_x[r][i], self.pos_x[r][i + 1]
                inter_nodes[0] = torch.linspace(p_start[0], p_end[0], Q_per_Seg + 1)
                inter_nodes[1] = torch.linspace(p_start[1], p_end[1], Q_per_Seg + 1)
                inter_nodes[2] = 0.0
                nodes = torch.cat((nodes, inter_nodes[:, 1:].T), dim=0)
            all_q_nodes.append(nodes)

        half_num_q = (self.gnd_stokeslet_node_num - 1) // 2
        gnd_nodes_q = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        inter_nodes_l = torch.zeros((3, half_num_q + 1), dtype=torch.float32)
        inter_nodes_l[0] = torch.linspace(0.0, -self.gnd_length / 2, half_num_q + 1)
        inter_nodes_r = torch.zeros((3, half_num_q + 1), dtype=torch.float32)
        inter_nodes_r[0] = torch.linspace(0.0, self.gnd_length / 2, half_num_q + 1)
        gnd_nodes_q = torch.cat((gnd_nodes_q, inter_nodes_l[:, 1:].T), dim=0)
        gnd_nodes_q = torch.cat((gnd_nodes_q, inter_nodes_r[:, 1:].T), dim=0)
        total_q_nodes = torch.cat(all_q_nodes + [gnd_nodes_q], dim=0)
        self.stokeslet_nodes = torch.flatten(total_q_nodes.T)

    def Step_Action(self, omega: torch.Tensor):
        self.Update_Global_Values(omega)
        if self.invalid_state:
            self.vel_vector.zero_()
            return

        self.update_Q()
        self.update_N()
        self.update_A()

        n = self.A_Matrix.size(0)
        A_safe = self.A_Matrix + torch.eye(n, dtype=torch.float32) * 1e-8
        self.w2f_Matrix = torch.matmul(torch.inverse(A_safe), self.Q_Matrix)

        vel_vec_list = []
        for r in range(self.num_robots):
            start = r * self.link_num
            end = start + self.link_num
            vel_vec_list.append(torch.cumsum(omega[start:end], dim=0))
        self.vel_vector = torch.cat(vel_vec_list, dim=0)

    def Update_Global_Values(self, omega: torch.Tensor):
        self.beta_last = self.beta.clone()
        for r in range(self.num_robots):
            for i in range(self.link_num):
                idx = r * self.link_num + i
                self.beta[idx] += omega[idx] * self.CONTROL_PERIOD
        self.Calculate_Robot_Config()

    def update_Q(self):
        self.Q_Matrix.zero_()
        N, N_fg_total = self.traction_node_num, self.fg_traction_node_num
        N_nodes_per_robot = self.nodes_per_robot
        N_per_Seg = (N_nodes_per_robot - 1) // self.link_num
        s = self.seg_length / N_per_Seg
        for i in range(3 * N):
            is_robot_x = i < N_fg_total
            is_robot_y = N <= i < (N + N_fg_total)
            if not (is_robot_x or is_robot_y):
                continue
            if is_robot_x:
                node_idx = i
                robot_idx = node_idx // N_nodes_per_robot
                local_node_idx = node_idx % N_nodes_per_robot
                cur_link_index = max(0, (local_node_idx - 1) // N_per_Seg)
                cur_link_delta = ((local_node_idx - 1) % N_per_Seg) * s if local_node_idx > 0 else 0.0
                theta_start_idx = robot_idx * self.link_num
                for j in range(self.link_num):
                    target_col = theta_start_idx + j
                    if j < cur_link_index:
                        self.Q_Matrix[i][target_col] = -self.seg_length * torch.sin(
                            self.theta[theta_start_idx + j]
                        )
                    elif j == cur_link_index and local_node_idx > 0:
                        self.Q_Matrix[i][target_col] = -cur_link_delta * torch.sin(
                            self.theta[theta_start_idx + j]
                        )
            elif is_robot_y:
                node_idx = i - N
                robot_idx = node_idx // N_nodes_per_robot
                local_node_idx = node_idx % N_nodes_per_robot
                cur_link_index = max(0, (local_node_idx - 1) // N_per_Seg)
                cur_link_delta = ((local_node_idx - 1) % N_per_Seg) * s if local_node_idx > 0 else 0.0
                theta_start_idx = robot_idx * self.link_num
                for j in range(self.link_num):
                    target_col = theta_start_idx + j
                    if j < cur_link_index:
                        self.Q_Matrix[i][target_col] = self.seg_length * torch.cos(
                            self.theta[theta_start_idx + j]
                        )
                    elif j == cur_link_index and local_node_idx > 0:
                        self.Q_Matrix[i][target_col] = cur_link_delta * torch.cos(
                            self.theta[theta_start_idx + j]
                        )

    def update_N(self, block_nodes: int = 0):
        Q, N = self.stokeslet_node_num, self.traction_node_num
        if block_nodes == 0:
            block_nodes = Q
        Q_x, Q_y, Q_z = self.stokeslet_nodes[:Q], self.stokeslet_nodes[Q : 2 * Q], self.stokeslet_nodes[2 * Q :]
        T_x, T_y, T_z = self.traction_nodes[:N], self.traction_nodes[N : 2 * N], self.traction_nodes[2 * N :]
        nMin = torch.zeros(Q, dtype=torch.long)
        for iMin in range(0, Q, block_nodes):
            iMax = min(iMin + block_nodes, Q)
            dis_x = Q_x[iMin:iMax, None] - T_x[None, :]
            dis_y = Q_y[iMin:iMax, None] - T_y[None, :]
            dis_z = Q_z[iMin:iMax, None] - T_z[None, :]
            distsq = dis_x.pow(2) + dis_y.pow(2) + dis_z.pow(2)
            _, nMin_block = distsq.min(dim=1)
            nMin[iMin:iMax] = nMin_block
        indices = torch.stack([torch.arange(Q), nMin])
        NClosest = torch.sparse_coo_tensor(indices, torch.ones(Q), size=(Q, N)).to_dense()
        self.N_Matrix = torch.kron(torch.eye(3), NClosest)

    def update_A(self):
        M, Q = self.traction_node_num, self.stokeslet_node_num
        r_x = self.traction_nodes[:M].unsqueeze(1) - self.stokeslet_nodes[:Q].unsqueeze(0)
        r_y = self.traction_nodes[M : 2 * M].unsqueeze(1) - self.stokeslet_nodes[Q : 2 * Q].unsqueeze(0)
        r_z = self.traction_nodes[2 * M :].unsqueeze(1) - self.stokeslet_nodes[2 * Q :].unsqueeze(0)
        r2 = r_x.pow(2) + r_y.pow(2) + r_z.pow(2)
        inv_r3 = torch.reciprocal((torch.sqrt(r2 + self.eps**2)) ** 3)
        factor = (r2 + 1.0 * (self.eps**2)) * inv_r3
        isotropic = torch.kron(torch.eye(3), factor)
        dyadic = torch.stack(
            [
                torch.stack([r_x * r_x, r_x * r_y, r_x * r_z], dim=1),
                torch.stack([r_y * r_x, r_y * r_y, r_y * r_z], dim=1),
                torch.stack([r_z * r_x, r_z * r_y, r_z * r_z], dim=1),
            ],
            dim=0,
        ).view(3 * M, 3 * Q)
        dyadic = dyadic * torch.kron(torch.ones(3, 3), inv_r3)
        self.A_Matrix = torch.matmul((1 / (8.0 * torch.pi)) * (isotropic + dyadic), self.N_Matrix)

    def regularize_stokeslet(self, x_field: torch.Tensor) -> torch.Tensor:
        x_field = x_field.reshape(-1)
        M, Q = int(len(x_field) // 3), self.stokeslet_node_num
        r_x = x_field[:M].unsqueeze(1) - self.stokeslet_nodes[:Q].unsqueeze(0)
        r_y = x_field[M : 2 * M].unsqueeze(1) - self.stokeslet_nodes[Q : 2 * Q].unsqueeze(0)
        r_z = x_field[2 * M : 3 * M].unsqueeze(1) - self.stokeslet_nodes[2 * Q : 3 * Q].unsqueeze(0)
        r2 = r_x.pow(2) + r_y.pow(2) + r_z.pow(2)
        inv_r3 = torch.reciprocal((torch.sqrt(r2 + self.eps**2)) ** 3)
        factor = (r2 + 2.0 * (self.eps**2)) * inv_r3
        isotropic = torch.kron(torch.eye(3), factor)
        dyadic = torch.stack(
            [
                torch.stack([r_x * r_x, r_x * r_y, r_x * r_z], dim=1),
                torch.stack([r_y * r_x, r_y * r_y, r_y * r_z], dim=1),
                torch.stack([r_z * r_x, r_z * r_y, r_z * r_z], dim=1),
            ],
            dim=0,
        ).view(3 * M, 3 * Q)
        dyadic = dyadic * torch.kron(torch.ones(3, 3), inv_r3)
        return (1 / (8.0 * torch.pi)) * (isotropic + dyadic)

    def set_grid_field(self, point_density: int):
        x_num, y_num = int(self.gnd_length * point_density), int(2.0 * point_density)
        x_values = torch.linspace(-self.gnd_length / 2, self.gnd_length / 2, steps=x_num)
        y_values = torch.linspace(0.0, 2.0, steps=y_num)
        x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
        self.field_grid = torch.cat(
            (x_grid.flatten(), y_grid.flatten(), torch.zeros_like(x_grid.flatten())),
            dim=0,
        )

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((int(self.window_dpi * self.gnd_length), int(self.window_dpi * 2.0)))
        canvas.fill((255, 255, 255))
        CX, CY, R = int(self.window_dpi * self.gnd_length) // 2, int(self.window_dpi * 2.0), self.window_dpi
        for r in range(self.num_robots):
            for i in range(1, self.link_num + 1):
                pygame.draw.line(
                    canvas,
                    (0, 0, 0),
                    (CX + float(self.pos_x[r][i - 1][0]) * R, CY - float(self.pos_x[r][i - 1][1]) * R),
                    (CX + float(self.pos_x[r][i][0]) * R, CY - float(self.pos_x[r][i][1]) * R),
                    width=4,
                )
        pygame.draw.line(canvas, (0, 0, 0), (CX - 4 * R, CY), (CX + 4 * R, CY), width=4)
        Stokeslet = self.regularize_stokeslet(self.field_grid)
        f_vel = torch.matmul(Stokeslet, torch.matmul(self.N_Matrix, torch.matmul(self.w2f_Matrix, self.vel_vector)))
        N = int(self.gnd_length * 2 * self.grid_dens**2)
        for i in range(0, N, 4):
            pygame.draw.line(
                canvas,
                (150, 150, 150),
                (CX + float(self.field_grid[i]) * R, CY - float(self.field_grid[N + i]) * R),
                (
                    CX + float(self.field_grid[i] + f_vel[i] * 0.2) * R,
                    CY - float(self.field_grid[N + i] + f_vel[N + i] * 0.2) * R,
                ),
                width=1,
            )
        px, py = CX + float(self.particle_pos[0]) * R, CY - float(self.particle_pos[1]) * R
        pygame.draw.circle(canvas, (0, 0, 255), (int(px), int(py)), 8)
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        pass
