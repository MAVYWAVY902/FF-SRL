"""
详细性能分析：找出 665ms 的真正来源
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

sim_bvh = dk.SimBVH(
    device, sim_model,
    binsNumber=8, load=True, save=False,
    path="../scenes/liverRetractionTexture.npz"
)

renderer = dk.render.WarpRaycastRendererDO(
    device=device,
    simModel=sim_model,
    resolution=512,
    cameraPos=[2.0, 22.0, 40.0],
    cameraRot=[0.0, 0.0, 0.0],
    lightPos=[0.0, 15.0, 80.0],
    lightIntensity=0.03,
    mode="gpu",
    horizontalAperture=45,
    verticalAperture=45
)

sim_integrator = dk.SimIntegratorDO(device)

print("\n" + "="*70)
print("分步测试：找出每个操作的真实开销")
print("="*70)

# 1. 测试物理计算
print("\n[1] 物理计算（stepModelReduce）")
sim_graph = None
times = []
for i in range(20):
    wp.synchronize()
    start = time.perf_counter()
    
    if sim_graph is None:
        wp.capture_begin()
        sim_integrator.stepModelReduce(sim_model)
        sim_graph = wp.capture_end()
    else:
        wp.capture_launch(sim_graph)
    
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"  平均: {np.mean(times[1:]):.3f} ms")  # 跳过第1帧（创建 graph）

# 2. 测试 BVH 更新
print("\n[2] BVH 更新（refitBVH with Graph）")
times = []
for i in range(20):
    wp.synchronize()
    start = time.perf_counter()
    
    sim_bvh.refitBVH(useGraph=True)
    
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"  平均: {np.mean(times[1:]):.3f} ms")

# 3. 测试渲染（不显示）
print("\n[3] 渲染计算（renderNew，跳过显示）")
times = []
for i in range(20):
    wp.synchronize()
    start = time.perf_counter()
    
    renderer.renderNew(sim_bvh, sim_model)
    
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"  平均: {np.mean(times[1:]):.3f} ms")

# 4. 完整循环测试（模拟交互式代码）
print("\n[4] 完整循环（物理 + BVH每5帧 + 渲染每5帧）")
times = []
sim_graph = None
bvh_counter = 0
render_counter = 0

for i in range(50):
    wp.synchronize()
    start = time.perf_counter()
    
    # 物理
    if sim_graph is None:
        wp.capture_begin()
        sim_integrator.stepModelReduce(sim_model)
        sim_graph = wp.capture_end()
    else:
        wp.capture_launch(sim_graph)
    
    # BVH（每5帧）
    bvh_counter += 1
    if bvh_counter % 5 == 0:
        sim_bvh.refitBVH(useGraph=True)
    
    # 渲染（每5帧）
    render_counter += 1
    if render_counter % 5 == 0:
        renderer.renderNew(sim_bvh, sim_model)
    
    wp.synchronize()
    times.append((time.perf_counter() - start) * 1000)

print(f"  平均: {np.mean(times[1:]):.3f} ms")
print(f"  中位数: {np.median(times[1:]):.3f} ms")
print(f"  最小值: {np.min(times[1:]):.3f} ms")
print(f"  最大值: {np.max(times[1:]):.3f} ms")

# 5. 测试 mode="gpu" 窗口开销
print("\n[5] 测试 GPU 窗口显示开销")
print("  检查 renderer.window 是否存在...")
if hasattr(renderer, 'window') and renderer.window is not None:
    print(f"  ✅ GPU 窗口已创建（这可能是性能瓶颈）")
    print(f"  窗口类型: {type(renderer.window)}")
else:
    print(f"  ❌ 没有 GPU 窗口")

print("\n" + "="*70)
print("诊断完成")
print("="*70)
print("\n如果'完整循环'时间接近用户看到的 665ms，说明测量准确")
print("如果远小于 665ms，说明有其他隐藏开销")
