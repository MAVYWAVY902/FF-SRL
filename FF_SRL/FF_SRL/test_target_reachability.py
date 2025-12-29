"""测试目标是否在RCM约束下可达."""
import numpy as np
import torch
from rl_env import FFSRLEnv

print("=" * 70)
print("目标可达性测试")
print("=" * 70)

env = FFSRLEnv(num_envs=1, device="cuda:0", action_strength=0.4, max_episode_steps=200)
obs, info = env.reset()

initial_pos = obs[:3].copy()
target_pos = obs[3:6].copy()
delta = target_pos - initial_pos
distance = np.linalg.norm(delta)

print(f"\n【初始状态】")
print(f"  初始位置: [{initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f}]")
print(f"  目标位置: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
print(f"  距离: {distance:.4f}")
print(f"  方向向量: [{delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}]")

print(f"\n【测试1：直线接近】")
print(f"  策略：每步沿目标方向移动，持续50步")

violation_count = 0
distances = [distance]
positions = [initial_pos.copy()]

for step in range(50):
    current_pos = obs[:3]
    target_pos = obs[3:6]
    direction = target_pos - current_pos
    dist = np.linalg.norm(direction)
    
    if dist > 0.01:
        action = direction / dist  # 归一化方向
    else:
        action = np.array([0.0, 0.0, 0.0])
        print(f"\n  ✓✓✓ 第{step+1}步到达目标！")
        break
    
    obs_new, reward, done, truncated, info = env.step(action)
    new_dist = np.linalg.norm(obs_new[:3] - obs_new[3:6])
    
    distances.append(new_dist)
    positions.append(obs_new[:3].copy())
    
    # 检查工作空间违规
    pos = obs_new[:3]
    in_workspace = (
        env.workspace_low[0] <= pos[0] <= env.workspace_high[0] and
        env.workspace_low[1] <= pos[1] <= env.workspace_high[1] and
        env.workspace_low[2] <= pos[2] <= env.workspace_high[2]
    )
    
    if not in_workspace:
        violation_count += 1
    
    # 每10步打印一次
    if (step + 1) % 10 == 0:
        print(f"  第{step+1}步: 距离={new_dist:.4f}, 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
              f"{'违规' if not in_workspace else '合法'}")
    
    obs = obs_new
    
    if done or truncated:
        print(f"\n  Episode在第{step+1}步结束")
        break

print(f"\n【结果分析】")
print(f"  初始距离: {distances[0]:.4f}")
print(f"  最终距离: {distances[-1]:.4f}")
print(f"  最小距离: {min(distances):.4f}")
print(f"  改善: {distances[0] - distances[-1]:.4f} ({(distances[0]-distances[-1])/distances[0]*100:.1f}%)")
print(f"  工作空间违规: {violation_count}/50步 ({violation_count/50*100:.1f}%)")

if distances[-1] < 0.3:
    print(f"\n  ✓✓✓ 成功！目标可达")
elif distances[-1] < distances[0]:
    print(f"\n  ⚠️  部分成功：能靠近但未达到目标")
    print(f"      可能原因：")
    print(f"        1. action_strength太小（当前0.4）")
    print(f"        2. RCM约束限制了最后的接近")
    print(f"        3. 物理碰撞阻止进一步移动")
else:
    print(f"\n  ✗✗✗ 失败！无法靠近目标")
    print(f"      问题：直线移动策略反而让距离增加")

if violation_count > 25:
    print(f"\n  ⚠️  严重工作空间违规（{violation_count}/50）")
    print(f"      说明：目标可能在工作空间外或RCM不可达区域")

print(f"\n【测试2：检查目标是否在工作空间内】")
print(f"  工作空间X: [{env.workspace_low[0]:.1f}, {env.workspace_high[0]:.1f}]")
print(f"  工作空间Y: [{env.workspace_low[1]:.1f}, {env.workspace_high[1]:.1f}]")
print(f"  工作空间Z: [{env.workspace_low[2]:.1f}, {env.workspace_high[2]:.1f}]")
print(f"\n  目标位置: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")

target_in_workspace = (
    env.workspace_low[0] <= target_pos[0] <= env.workspace_high[0] and
    env.workspace_low[1] <= target_pos[1] <= env.workspace_high[1] and
    env.workspace_low[2] <= target_pos[2] <= env.workspace_high[2]
)

if target_in_workspace:
    print(f"  ✓ 目标在工作空间内")
else:
    print(f"  ✗ 目标在工作空间外！！！")
    print(f"    这是根本性错误 - 目标根本不可达")

print("\n" + "=" * 70)
