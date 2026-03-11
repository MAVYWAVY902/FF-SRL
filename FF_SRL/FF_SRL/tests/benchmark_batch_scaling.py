"""
Benchmark: Warp batch scaling vs C++ baseline
Tests num_envs = [1, 8, 32, 128, 512] to find break-even point
"""

import warp as wp
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

import FF_SRL as dk

try:
    from pxr import Usd
except:
    print("Warning: pxr not available, using placeholder")

wp.init()

def benchmark_warp_scaling(num_envs_list=[1], num_steps=50):
    """
    Benchmark Warp performance across different batch sizes
    
    Returns:
        results: dict with timing data
    """
    results = {
        'num_envs': [],
        'total_time_ms': [],
        'time_per_env_ms': [],
        'envs_per_sec': [],
        'gpu_utilization': []
    }
    
    for num_envs in num_envs_list:
        print(f"\n{'='*60}")
        print(f"Testing Warp with {num_envs} parallel environments...")
        print(f"{'='*60}")
        
        # Initialize environment
        device = "cuda:0"
        fps = 60
        dt = 1.0 / fps
        sim_substeps = 10
        constraint_steps = 1
        
        try:
            print(f"Loading USD scene for {num_envs} environments...")
            stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
            
            print("Creating simulation model...")
            sim_model = dk.SimModelDO(
                stage, num_envs, device,
                globalKsDrag=1.0,
                simFrameRate=fps,
                simConstraintsSteps=constraint_steps,
                simSubsteps=sim_substeps,
                globalLaparoscopeDragLookupRadius=1.0,
                globalKsDistance=1.0,
                globalKsVolume=1.0
            )
            
            print("Creating integrator...")
            integrator = dk.SimIntegratorDO(device)
            
            # Dummy action (no tool movement)
            dummy_action = torch.zeros((num_envs, 6), device=device)
            
            # Warmup
            print("Warmup (10 steps)...")
            for _ in range(10):
                integrator.stepModel(sim_model)
            
            # Actual benchmark
            print(f"Running benchmark ({num_steps} steps)...")
            wp.synchronize()  # Ensure GPU is done
            start = time.perf_counter()
            
            for step in range(num_steps):
                integrator.stepModel(sim_model)
            
            wp.synchronize()  # Wait for all GPU work
            end = time.perf_counter()
            
            total_time = (end - start) * 1000  # ms
            time_per_step = total_time / num_steps
            time_per_env = time_per_step / num_envs
            envs_per_sec = (num_envs * num_steps) / (end - start)
            
            results['num_envs'].append(num_envs)
            results['total_time_ms'].append(time_per_step)
            results['time_per_env_ms'].append(time_per_env)
            results['envs_per_sec'].append(envs_per_sec)
            
            print(f"\n{'='*60}")
            print(f"Results for {num_envs} environments:")
            print(f"{'='*60}")
            print(f"  Total time per step: {time_per_step:.2f} ms")
            print(f"  Time per environment: {time_per_env:.2f} ms")
            print(f"  Throughput: {envs_per_sec:.1f} envs/sec")
            print(f"  GPU efficiency estimate: {100 * num_envs / 1024:.1f}%")
            
            # Cleanup
            del sim_model
            del integrator
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {num_envs} environments: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return results


def plot_scaling_results(results, cpp_baseline_ms=5.0):
    """
    Visualize batch scaling efficiency
    
    Args:
        results: Output from benchmark_warp_scaling
        cpp_baseline_ms: C++ single-env time (ms)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    num_envs = np.array(results['num_envs'])
    
    # Plot 1: Time per environment (should decrease with batching)
    ax = axes[0, 0]
    ax.plot(num_envs, results['time_per_env_ms'], 'bo-', linewidth=2, label='Warp GPU')
    ax.axhline(cpp_baseline_ms, color='r', linestyle='--', linewidth=2, label='C++ CPU (single)')
    ax.axhline(cpp_baseline_ms / 8, color='g', linestyle='--', linewidth=2, label='C++ CPU (8-core)')
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Time per Environment (ms)', fontsize=12)
    ax.set_title('Per-Environment Latency vs Batch Size', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput (envs/sec)
    ax = axes[0, 1]
    ax.plot(num_envs, results['envs_per_sec'], 'go-', linewidth=2, label='Warp GPU')
    cpp_throughput = 1000 / cpp_baseline_ms
    cpp_parallel_throughput = 8 * 1000 / cpp_baseline_ms
    ax.axhline(cpp_throughput, color='r', linestyle='--', linewidth=2, label=f'C++ single ({cpp_throughput:.0f}/s)')
    ax.axhline(cpp_parallel_throughput, color='g', linestyle='--', linewidth=2, label=f'C++ 8-core ({cpp_parallel_throughput:.0f}/s)')
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Throughput (envs/sec)', fontsize=12)
    ax.set_title('Throughput vs Batch Size', fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parallel efficiency (ideal = 1.0)
    ax = axes[1, 0]
    ideal_time = results['total_time_ms'][0]  # Time for 1 env
    actual_times = np.array(results['total_time_ms'])
    parallel_efficiency = ideal_time / (actual_times / num_envs)
    ax.plot(num_envs, parallel_efficiency, 'mo-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax.set_xlabel('Number of Parallel Environments', fontsize=12)
    ax.set_ylabel('Parallel Efficiency', fontsize=12)
    ax.set_title('GPU Parallelization Efficiency', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim([0, 1.5])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Break-even analysis
    ax = axes[1, 1]
    warp_total_time = np.array(results['total_time_ms'])
    cpp_single_total = cpp_baseline_ms * num_envs
    cpp_parallel_total = (cpp_baseline_ms * num_envs) / 8
    
    ax.plot(num_envs, warp_total_time, 'bo-', linewidth=2, label='Warp GPU')
    ax.plot(num_envs, cpp_single_total, 'r--', linewidth=2, label='C++ single-thread')
    ax.plot(num_envs, cpp_parallel_total, 'g--', linewidth=2, label='C++ 8-core')
    ax.set_xlabel('Number of Environments', fontsize=12)
    ax.set_ylabel('Total Time per Step (ms)', fontsize=12)
    ax.set_title('Break-Even Point Analysis', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent.parent / 'output' / 'batch_scaling_benchmark.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SCALING ANALYSIS SUMMARY")
    print("="*60)
    
    # Find break-even point
    cpp_8core_time_per_env = cpp_baseline_ms / 8
    for i, n in enumerate(num_envs):
        warp_per_env = results['time_per_env_ms'][i]
        if warp_per_env < cpp_8core_time_per_env:
            print(f"\n✅ GPU becomes faster than C++ 8-core at {n}+ environments")
            print(f"   Warp: {warp_per_env:.2f} ms/env")
            print(f"   C++:  {cpp_8core_time_per_env:.2f} ms/env")
            break
    else:
        print(f"\n❌ GPU never beats C++ 8-core in tested range")
        print(f"   Best Warp: {min(results['time_per_env_ms']):.2f} ms/env @ {num_envs[np.argmin(results['time_per_env_ms'])]} envs")
        print(f"   C++ 8-core: {cpp_8core_time_per_env:.2f} ms/env (constant)")
        print(f"\n   Recommendation: Need {int(num_envs[-1] * 2)}+ environments for GPU advantage")
    
    print("="*60)


if __name__ == "__main__":
    print("Warp Batch Scaling Benchmark")
    print("="*60)
    print("This benchmark will help determine:")
    print("  1. GPU parallelization efficiency")
    print("  2. Break-even point vs C++ baseline")
    print("  3. Optimal batch size for your hardware")
    print("="*60)
    
    # Start with single environment to establish baseline
    print("\n✅ Single environment test successful!")
    print("Now testing batch scaling: [1, 2, 4, 8, 16]\n")
    
    # Run benchmark with multiple batch sizes
    results = benchmark_warp_scaling(
        num_envs_list=[1, 2, 4, 8, 16],  # Progressive batch sizes
        num_steps=50
    )
    
    if len(results['num_envs']) > 0:
        # Plot results
        plot_scaling_results(results, cpp_baseline_ms=5.0)
        
        print("\n" + "="*60)
        print("✅ Benchmark completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the plot in FF_SRL/output/batch_scaling_benchmark.png")
        print("2. To test more environments, edit this file and change:")
        print("   num_envs_list=[1, 2, 4, 8]")
        print("3. Compare with your C++ baseline (currently set to 5ms)")
        print("="*60)
    else:
        print("\n❌ Benchmark failed. Check errors above.")
