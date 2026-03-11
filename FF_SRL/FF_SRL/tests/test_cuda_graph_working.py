#!/usr/bin/env python3
"""
Minimal test to verify CUDA Graph is actually working
We'll compare:
1. Without Graph: Direct wp.launch() calls
2. With Graph: wp.capture_begin/end + wp.capture_launch()
"""

import warp as wp
import time
import torch

wp.init()
wp.config.verify_cuda = False

# Simple kernel for testing
@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def test_without_graph(n_iterations=100):
    """Test WITHOUT CUDA Graph - every iteration calls wp.launch()"""
    print("\n" + "="*60)
    print("TEST 1: WITHOUT CUDA Graph (direct wp.launch)")
    print("="*60)
    
    N = 1000
    device = "cuda:0"
    
    a = wp.zeros(N, dtype=wp.float32, device=device)
    b = wp.zeros(N, dtype=wp.float32, device=device)
    c = wp.zeros(N, dtype=wp.float32, device=device)
    
    # Warmup
    for _ in range(10):
        wp.launch(kernel=add_kernel, dim=N, inputs=[a, b, c], device=device)
    wp.synchronize()
    
    # Timed run
    start = time.time()
    for i in range(n_iterations):
        wp.launch(kernel=add_kernel, dim=N, inputs=[a, b, c], device=device)
    wp.synchronize()
    elapsed = time.time() - start
    
    avg_ms = (elapsed / n_iterations) * 1000
    print(f"✅ Completed {n_iterations} iterations")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Average per iteration: {avg_ms:.3f}ms")
    print(f"Throughput: {1000/avg_ms:.1f} iterations/sec")
    
    return avg_ms

def test_with_graph(n_iterations=100):
    """Test WITH CUDA Graph - capture once, replay many times"""
    print("\n" + "="*60)
    print("TEST 2: WITH CUDA Graph (wp.capture_begin/end)")
    print("="*60)
    
    N = 1000
    device = "cuda:0"
    
    a = wp.zeros(N, dtype=wp.float32, device=device)
    b = wp.zeros(N, dtype=wp.float32, device=device)
    c = wp.zeros(N, dtype=wp.float32, device=device)
    
    # Capture graph
    print("Capturing CUDA Graph...")
    wp.capture_begin()
    wp.launch(kernel=add_kernel, dim=N, inputs=[a, b, c], device=device)
    graph = wp.capture_end()
    print(f"✅ Graph captured: {graph}")
    
    # Warmup
    for _ in range(10):
        wp.capture_launch(graph)
    wp.synchronize()
    
    # Timed run
    start = time.time()
    for i in range(n_iterations):
        wp.capture_launch(graph)
    wp.synchronize()
    elapsed = time.time() - start
    
    avg_ms = (elapsed / n_iterations) * 1000
    print(f"✅ Completed {n_iterations} iterations")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Average per iteration: {avg_ms:.3f}ms")
    print(f"Throughput: {1000/avg_ms:.1f} iterations/sec")
    
    return avg_ms

def test_graph_conditional(n_iterations=100):
    """Test the pattern used in testInteractiveControl.py"""
    print("\n" + "="*60)
    print("TEST 3: Graph with if/else (testInteractiveControl pattern)")
    print("="*60)
    
    N = 1000
    device = "cuda:0"
    
    a = wp.zeros(N, dtype=wp.float32, device=device)
    b = wp.zeros(N, dtype=wp.float32, device=device)
    c = wp.zeros(N, dtype=wp.float32, device=device)
    
    graph = None
    
    # Warmup
    for _ in range(10):
        if graph is None:
            wp.capture_begin()
            wp.launch(kernel=add_kernel, dim=N, inputs=[a, b, c], device=device)
            graph = wp.capture_end()
        else:
            wp.capture_launch(graph)
    wp.synchronize()
    
    # Timed run
    start = time.time()
    for i in range(n_iterations):
        if graph is None:
            print("ERROR: Graph should not be None!")
            wp.capture_begin()
            wp.launch(kernel=add_kernel, dim=N, inputs=[a, b, c], device=device)
            graph = wp.capture_end()
        else:
            wp.capture_launch(graph)
    wp.synchronize()
    elapsed = time.time() - start
    
    avg_ms = (elapsed / n_iterations) * 1000
    print(f"✅ Completed {n_iterations} iterations")
    print(f"Total time: {elapsed:.4f}s")
    print(f"Average per iteration: {avg_ms:.3f}ms")
    print(f"Throughput: {1000/avg_ms:.1f} iterations/sec")
    
    return avg_ms

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("CUDA Graph Performance Test")
    print("🚀" * 30)
    
    n_iter = 100
    
    # Test 1: No Graph
    time_no_graph = test_without_graph(n_iter)
    
    # Test 2: With Graph
    time_with_graph = test_with_graph(n_iter)
    
    # Test 3: Conditional pattern
    time_conditional = test_graph_conditional(n_iter)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Without Graph:      {time_no_graph:.3f}ms per iteration")
    print(f"With Graph:         {time_with_graph:.3f}ms per iteration")
    print(f"Conditional Graph:  {time_conditional:.3f}ms per iteration")
    print(f"\nSpeedup (Graph vs No-Graph): {time_no_graph/time_with_graph:.1f}x")
    print(f"Speedup (Conditional vs No-Graph): {time_no_graph/time_conditional:.1f}x")
    
    if time_with_graph < time_no_graph * 0.5:
        print("\n✅ CUDA Graph is working correctly! (>2x speedup)")
    else:
        print("\n❌ WARNING: CUDA Graph may not be working properly!")
        print(f"   Expected >2x speedup, got {time_no_graph/time_with_graph:.1f}x")
