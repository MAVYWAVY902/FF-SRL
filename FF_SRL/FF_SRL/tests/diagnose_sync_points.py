"""
诊断脚本：检测 CUDA Graph 和同步点问题

对比三种执行模式：
1. 无 Graph + 有 Sync（最慢）
2. 有 Graph + 有 Sync（中等）
3. 有 Graph + 无 Sync（最快）
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

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")


def benchmark_mode(mode_name, use_graph, use_sync, iterations=100):
    """
    测试不同执行模式的性能
    
    Args:
        mode_name: 模式名称
        use_graph: 是否使用 CUDA Graph
        use_sync: 是否在计时时调用 synchronize
        iterations: 测试迭代次数
    """
    print(f"\n{'='*70}")
    print(f"测试模式: {mode_name}")
    print(f"CUDA Graph: {'✅' if use_graph else '❌'} | Synchronize: {'✅' if use_sync else '❌'}")
    print(f"{'='*70}")
    
    device = "cuda:0"
    num_envs = 1
    
    # 初始化模型
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
    
    # 工具位置初始化
    initial_tool_pos = torch.tensor([0.0, 23.0, -2.0], dtype=torch.float, device=device)
    current_pos = sim_model.getLaparoscopePositionsTensor()
    shift = initial_tool_pos - current_pos
    sim_model.applyCartesianActions(wp.from_torch(shift[0], dtype=wp.float32))
    
    # CUDA Graph (如果启用)
    sim_graph = None
    if use_graph:
        wp.capture_begin()
        if sim_model.reduce:
            sim_integrator.stepModelReduce(sim_model)
        else:
            sim_integrator.stepModel(sim_model)
        sim_graph = wp.capture_end()
    
    # 预热
    print("预热中...")
    for _ in range(10):
        if use_graph:
            wp.capture_launch(sim_graph)
        else:
            if sim_model.reduce:
                sim_integrator.stepModelReduce(sim_model)
            else:
                sim_integrator.stepModel(sim_model)
        if use_sync:
            wp.synchronize()
    
    # 正式测试
    print(f"运行 {iterations} 次迭代...")
    times = []
    
    for i in range(iterations):
        if use_sync:
            wp.synchronize()  # 确保上一次操作完成
        
        start = time.perf_counter()
        
        if use_graph:
            wp.capture_launch(sim_graph)
        else:
            if sim_model.reduce:
                sim_integrator.stepModelReduce(sim_model)
            else:
                sim_integrator.stepModel(sim_model)
        
        if use_sync:
            wp.synchronize()  # 等待当前操作完成
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    # 统计结果
    times = np.array(times)
    print(f"\n结果统计:")
    print(f"  平均时间: {np.mean(times):.3f} ms")
    print(f"  中位数:   {np.median(times):.3f} ms")
    print(f"  最小值:   {np.min(times):.3f} ms")
    print(f"  最大值:   {np.max(times):.3f} ms")
    print(f"  标准差:   {np.std(times):.3f} ms")
    
    return np.mean(times)


def check_sync_in_integrator():
    """检查 integrator 内部是否有隐藏的同步点"""
    print(f"\n{'='*70}")
    print("检查 SimIntegratorDO 源码中的同步点")
    print(f"{'='*70}\n")
    
    integrator_file = "../integrator.py"
    if os.path.exists(integrator_file):
        with open(integrator_file, 'r') as f:
            lines = f.readlines()
        
        sync_points = []
        for i, line in enumerate(lines, 1):
            if 'synchronize' in line.lower() or 'to_torch' in line or 'from_torch' in line:
                sync_points.append((i, line.strip()))
        
        if sync_points:
            print(f"❌ 发现 {len(sync_points)} 个潜在同步点:")
            for line_num, line_content in sync_points:
                print(f"  行 {line_num}: {line_content}")
        else:
            print("✅ 未发现明显的同步点")
    else:
        print(f"⚠️  找不到文件: {integrator_file}")


def check_sync_in_tool_controller():
    """检查 tool_controller 中的同步点"""
    print(f"\n{'='*70}")
    print("检查 ToolController 中的同步点")
    print(f"{'='*70}\n")
    
    controller_file = "../tool_controller.py"
    if os.path.exists(controller_file):
        with open(controller_file, 'r') as f:
            lines = f.readlines()
        
        sync_points = []
        for i, line in enumerate(lines, 1):
            if 'synchronize' in line.lower() or 'to_torch' in line or 'from_torch' in line or '.numpy()' in line:
                sync_points.append((i, line.strip()))
        
        if sync_points:
            print(f"❌ 发现 {len(sync_points)} 个潜在同步点:")
            for line_num, line_content in sync_points:
                print(f"  行 {line_num}: {line_content}")
        else:
            print("✅ 未发现明显的同步点")
    else:
        print(f"⚠️  找不到文件: {controller_file}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUDA Graph 和同步点诊断工具")
    print("="*70)
    
    # 测试四种模式
    results = {}
    
    # 模式 1: 无 Graph + 有 Sync（用户看到的"优化版"性能）
    results['No Graph + Sync'] = benchmark_mode(
        "无 CUDA Graph + 有 Synchronize",
        use_graph=False,
        use_sync=True,
        iterations=100
    )
    
    # 模式 2: 有 Graph + 有 Sync
    results['Graph + Sync'] = benchmark_mode(
        "有 CUDA Graph + 有 Synchronize",
        use_graph=True,
        use_sync=True,
        iterations=100
    )
    
    # 模式 3: 有 Graph + 无 Sync（理论最快）
    results['Graph + No Sync'] = benchmark_mode(
        "有 CUDA Graph + 无 Synchronize",
        use_graph=True,
        use_sync=False,
        iterations=100
    )
    
    # 模式 4: 无 Graph + 无 Sync
    results['No Graph + No Sync'] = benchmark_mode(
        "无 CUDA Graph + 无 Synchronize",
        use_graph=False,
        use_sync=False,
        iterations=100
    )
    
    # 性能对比
    print(f"\n{'='*70}")
    print("性能对比总结")
    print(f"{'='*70}")
    for mode, avg_time in results.items():
        speedup = results['No Graph + Sync'] / avg_time
        print(f"{mode:25s}: {avg_time:7.3f} ms  (加速 {speedup:.2f}x)")
    
    print(f"\n{'='*70}")
    print("关键发现:")
    print(f"{'='*70}")
    
    # 计算 Graph 的加速比
    graph_speedup = results['No Graph + Sync'] / results['Graph + Sync']
    print(f"1. CUDA Graph 加速比: {graph_speedup:.2f}x")
    
    # 计算 Sync 的开销
    sync_overhead_graph = results['Graph + Sync'] / results['Graph + No Sync']
    sync_overhead_no_graph = results['No Graph + Sync'] / results['No Graph + No Sync']
    print(f"2. Synchronize 开销 (有 Graph): {sync_overhead_graph:.2f}x")
    print(f"3. Synchronize 开销 (无 Graph): {sync_overhead_no_graph:.2f}x")
    
    # 检查源码中的同步点
    check_sync_in_integrator()
    check_sync_in_tool_controller()
    
    print(f"\n{'='*70}")
    print("诊断完成")
    print(f"{'='*70}\n")
