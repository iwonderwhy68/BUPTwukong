# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:34:52 2024

@author: admin
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# 计算每个区域的目标无人机数量，确保每个区域至少有min_uavs
def calculate_target_uavs(regions, total_uavs):
    """
    根据区域的优先级计算每个区域的目标无人机数量，确保每个区域至少有min_uavs。
    """
    total_min_uavs = sum(region['min_uavs'] for region in regions)
    if total_min_uavs > total_uavs:
        raise ValueError("总无人机数量不足以满足所有区域的最少保障需求")
    
    remaining_uavs = total_uavs - total_min_uavs
    total_priority = sum(region['priority'] for region in regions)
    target_uavs = [region['min_uavs'] for region in regions]
    
    # 分配剩余无人机根据优先级
    for idx, region in enumerate(regions):
        extra = int((region['priority'] / total_priority) * remaining_uavs)
        target_uavs[idx] += extra
    
    # 处理由于整数除法导致的剩余无人机
    assigned = sum(target_uavs)
    remaining = total_uavs - assigned
    while remaining > 0:
        for idx, region in enumerate(regions):
            if remaining == 0:
                break
            target_uavs[idx] += 1
            remaining -= 1
    
    return target_uavs

# 定义环境
class UAVEnvironment:
    def __init__(self, M, N, regions, grid_size=0.5, max_steps=50, target_uavs=None):
        """
        M: 无人机数量
        N: 区域数量
        regions: 各区域的定义，列表中每个元素为 {'x_center': x, 'y_center': y, 'priority': p, 'radius': r, 'min_uavs': m}
        grid_size: 网格大小，用于离散化状态空间
        max_steps: 每个周期的最大步数
        target_uavs: 各区域的目标无人机数量
        """
        self.M = M
        self.N = N
        self.regions = regions
        self.grid_size = grid_size
        self.max_steps = max_steps
        if target_uavs is None:
            self.target_uavs = [region['min_uavs'] for region in regions]  # 默认每个区域目标为min_uavs
        else:
            self.target_uavs = target_uavs
        self.reset()
    
    def reset(self):
        """
        随机初始化无人机的位置，返回无人机的当前位置列表
        """
        self.uav_positions = [(
            random.uniform(0, 10),
            random.uniform(0, 10)
        ) for _ in range(self.M)]
        self.uav_positions[0] = (2.2,8)
        self.uav_positions[1] = (2,8.2)
        self.uav_positions[2] = (2.2,8.2)
        self.uav_positions[3] = (1.8,8)
        self.uav_positions[4] = (2,8)
        self.current_step = 0
        self.done = False
        self.trajectories = [[] for _ in range(self.M)]  # 记录每架无人机的轨迹
        for i in range(self.M):
            self.trajectories[i].append(self.uav_positions[i])
        return self.get_states()
    
    def get_states(self):
        """
        将连续位置映射到离散状态
        """
        states = []
        for pos in self.uav_positions:
            x, y = pos
            state_x = min(int(x // self.grid_size), 19)  # 确保索引不超过19
            state_y = min(int(y // self.grid_size), 19)
            state = state_x * 20 + state_y  # 状态编号从0到399
            states.append(state)
        return states
    
    def step(self, actions):
        """
        执行动作，返回下一个状态、奖励和是否完成
        actions: 无人机的动作列表，每个动作为0=上, 1=下, 2=左, 3=右, 4=左上, 5=右上, 6=左下, 7=右下
        """
        rewards = np.zeros(self.M)
        region_counts = {i:0 for i in range(self.N)}  # 记录每个区域内的无人机数量
        
        # 定义动作对应的移动向量
        action_moves = {
            0: (0, 0.5),    # 上
            1: (0, -0.5),   # 下
            2: (-0.5, 0),   # 左
            3: (0.5, 0),    # 右
            4: (-0.5, 0.5), # 左上
            5: (0.5, 0.5),  # 右上
            6: (-0.5, -0.5),# 左下
            7: (0.5, -0.5)  # 右下
        }
        
        # 执行所有动作，更新位置
        for i, action in enumerate(actions):
            dx, dy = action_moves.get(action, (0, 0))
            x, y = self.uav_positions[i]
            x_new = min(max(x + dx, 0), 10)
            y_new = min(max(y + dy, 0), 10)
            self.uav_positions[i] = (x_new, y_new)
            self.trajectories[i].append((x_new, y_new))
        
        # 统计每个区域内的无人机数量
        for pos in self.uav_positions:
            for idx, region in enumerate(self.regions):
                xc, yc, radius = region['x_center'], region['y_center'], region['radius']
                distance = np.sqrt((pos[0] - xc)**2 + (pos[1] - yc)**2)
                if distance <= radius:
                    region_counts[idx] += 1
                    break  # 只计入一个区域
        
        # 计算奖励
        for i, pos in enumerate(self.uav_positions):
            reward = 0
            in_region = False
            for idx, region in enumerate(self.regions):
                xc, yc, radius = region['x_center'], region['y_center'], region['radius']
                priority = region['priority']
                distance = np.sqrt((pos[0] - xc)**2 + (pos[1] - yc)**2)
                if distance <= radius:
                    in_region = True
                    # 奖励基于区域优先级
                    reward += priority
                    # 额外奖励：如果区域内无人机数量未超过目标数量
                    if region_counts[idx] <= self.target_uavs[idx]:
                        reward += 1  # 提供额外奖励，鼓励满足目标
                    else:
                        reward -= 0.5 * (region_counts[idx] - self.target_uavs[idx])  # 奖励递减，避免过度集中
                    break
            if not in_region:
                reward = -0.1  # 不在任何区域时的小惩罚
            rewards[i] = reward
        
        # 全局惩罚：如果某些区域未达到最小保障数量
        total_missing = 0
        for idx, region in enumerate(self.regions):
            missing = max(0, region['min_uavs'] - region_counts[idx])
            total_missing += missing
        
        if total_missing > 0:
            penalty_per_uav = 0.1 * total_missing / self.M  # 分摊惩罚
            rewards -= penalty_per_uav  # 对所有无人机施加惩罚
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
        return self.get_states(), rewards, self.done

# 定义智能体
class UAVAgent:
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        self.state_size = 400  # 20x20网格
        self.action_size = 8  # 上, 下, 左, 右, 左上, 右上, 左下, 右下
        self.q_table = np.zeros((self.state_size, self.action_size))  # Q表
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.15  # 探索与利用的平衡
    
    def choose_action(self, state):
        """
        根据当前状态选择动作，使用ε-贪婪策略
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用
    
    def learn(self, state, action, reward, next_state):
        """
        更新Q表
        """
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (target - predict)

# 训练函数
def train(env, agents, episodes):
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                action = agent.choose_action(state[i])
                actions.append(action)
            next_state, rewards, done = env.step(actions)
            for i, agent in enumerate(agents):
                agent.learn(state[i], actions[i], rewards[i], next_state[i])
                total_reward += rewards[i]
            state = next_state
        average_reward = total_reward / env.M
        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode}/{episodes} - Total Reward: {total_reward:.2f} - Average Reward per UAV: {average_reward:.2f}")
    return env.trajectories

# 可视化函数
def visualize(env, trajectories):
    """
    可视化无人机的飞行轨迹
    env: 环境对象，包含区域坐标
    trajectories: 无人机轨迹列表，每个元素是一个无人机的(x, y)序列
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制区域
    for idx, region in enumerate(env.regions):
        xc, yc, priority, radius = region['x_center'], region['y_center'], region['priority'], region['radius']
        circle = plt.Circle((xc, yc), radius, color='red', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.text(xc, yc, f'R{idx}\nP:{priority}\nMin:{region["min_uavs"]}', 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=12, color='white')
    
    # 绘制无人机轨迹
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H', '+']
    for i, traj in enumerate(trajectories):
        traj_coords = traj
        xs, ys = zip(*traj_coords)
        plt.plot(xs, ys, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f'UAV {i+1}', linewidth=2)
        # 标记起点
        plt.scatter(xs[0], ys[0], marker='X', color='black', s=100, edgecolors='k', zorder=7)
        plt.text(xs[0], ys[0], f'S{i+1}', horizontalalignment='right', verticalalignment='bottom', fontsize=10, color='black', zorder=7)
        # 标记终点
        plt.scatter(xs[-1], ys[-1], marker='*', color='gold', s=200, edgecolors='k', zorder=7)
        plt.text(xs[-1], ys[-1], f'E{i+1}', horizontalalignment='left', verticalalignment='bottom', fontsize=10, color='black', zorder=7)
    
    plt.title('无人机飞行轨迹')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 定义区域: {'x_center': x, 'y_center': y, 'priority': p, 'radius': r, 'min_uavs': m}
    regions = [
        {'x_center': 2, 'y_center': 2, 'priority': 3, 'radius': 1.5, 'min_uavs': 1},   # 区域0
        {'x_center': 8, 'y_center': 2, 'priority': 10, 'radius': 1.5, 'min_uavs': 1},   # 区域1
        {'x_center': 8, 'y_center': 8, 'priority': 3, 'radius': 1.5, 'min_uavs': 1},   # 区域2
    ]
    N = len(regions)  # 区域数量
    M = 18  # 无人机数量
    grid_size = 0.5  # 网格大小
    max_steps = 80  # 每个周期的最大步数
    
    # 计算目标无人机数量
    target_uavs = calculate_target_uavs(regions, M)
    print(f"目标无人机数量: {target_uavs}")
    
    # 初始化环境和智能体
    env = UAVEnvironment(M=M, N=N, regions=regions, grid_size=grid_size, max_steps=max_steps, target_uavs=target_uavs)
    agents = [UAVAgent(env, agent_id=i) for i in range(M)]
    
    # 训练智能体
    episodes = 14000
    trajectories = train(env, agents, episodes)
    
    # 可视化训练完成后的飞行轨迹
    visualize(env, trajectories)
