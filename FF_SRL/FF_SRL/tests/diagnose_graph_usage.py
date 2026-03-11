"""
深度诊断：检查 CUDA Graph 是否真的在使用
"""
import torch
import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np
import FF_SRL as dk

wp.init()

from pxr import Usd

device = "cuda:0"
num_envs = 1

# 初始化模型
print("初始化模型...")
stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
sim_model = dk.SimModelDO(
    stage, num_envs, device,
    globalKsDrag=1.0,
    simFrameRate=60,
    simConstraintsSteps=1,
    simSubsteps=10,
    globalLaparoscopeDragLookupRadius=1.0,
    globalKsDistance=1.0,
    globalKsVolume=1.0
)

sim_integrator = dk.SimIntegratorDO(device)

print("\n" + "="*70)
print("测试 1：无 Graph（直接调用）")
print("="*70)

# 预热
for _ in range(5):
    if sim_model.reduce:
        sim_integrator.stepModelReduce(sim_model)
    else:
        sim_integrator.stepModel(sim_model)

wp.synchronize()
times = []
for i in range(50):
    start = time.perf_counter()
    if sim_model.reduce:
        sim_integrator.stepModelReduce(sim_model)
    else:
        sim_integrator.stepModel(sim_model)
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"平均时间（无 Graph）: {np.mean(times):.3f} ms")
print(f"中位数: {np.median(times):.3f} ms")

print("\n" + "="*70)
print("测试 2：有 Graph（使用 capture）")
print("="*70)

# 创建 graph
wp.capture_begin()
if sim_model.reduce:
    sim_integrator.stepModelReduce(sim_model)
else:
    sim_integrator.stepModel(sim_model)
sim_graph = wp.capture_end()

# 预热
for _ in range(5):
    wp.capture_launch(sim_graph)

wp.synchronize()
times = []
for i in range(50):
    start = time.perf_counter()
    wp.capture_launch(sim_graph)
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"平均时间（有 Graph）: {np.mean(times):.3f} ms")
print(f"中位数: {np.median(times):.3f} ms")

print("\n" + "="*70)
print("测试 3：模拟交互式代码（带工具控制）")
print("="*70)

# 重置 graph
sim_graph = None

# 模拟交互式循环
times = []
for i in range(50):
    # 模拟 apply_actions (零动作)
    actions = torch.zeros(3, dtype=torch.float32, device=device)
    if torch.any(actions != 0.0):  # 永远不会执行
        pass
    
    start = time.perf_counter()
    
    # 模拟 step() 函数逻辑
    if sim_graph is None:
        wp.capture_begin()
        if sim_model.reduce:
            sim_integrator.stepModelReduce(sim_model)
        else:
            sim_integrator.stepModel(sim_model)
        sim_graph = wp.capture_end()
    else:
        wp.capture_launch(sim_graph)
    
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"平均时间（交互式循环）: {np.mean(times):.3f} ms")
print(f"中位数: {np.median(times):.3f} ms")
print(f"第1帧（创建 Graph）: {times[0]:.3f} ms")
print(f"第2帧（使用 Graph）: {times[1]:.3f} ms")

print("\n" + "="*70)
print("诊断完成")
print("="*70)
print("\n如果'交互式循环'的时间接近 1-5ms，说明 Graph 工作正常")
print("如果仍然是 1000+ms，说明有其他同步点")
