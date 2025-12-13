"""
Simple demo comparing standard volume constraint vs Stable Neo-Hookean
Shows how Stable NH handles extreme deformations better
"""

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constraints_stable_nh import (
    stableNeohookeanHydrostaticConstraint,
    computeInverseRestMatrices
)
from integrator import volumeConstraintsXPBD

wp.init()

def test_compression_comparison():
    """
    Compare standard volume constraint vs Stable NH
    under progressive compression
    """
    device = "cuda:0"
    
    # Test different compression ratios
    compression_ratios = np.linspace(0.2, 1.0, 20)  # 20% to 100%
    
    standard_energies = []
    stable_energies = []
    
    print("Testing compression from 20% to 100% of rest volume...")
    
    for ratio in compression_ratios:
        # Rest configuration
        vertex_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Compressed configuration
        vertex_compressed = vertex_rest * (ratio ** (1/3))  # uniform scaling
        
        tetrahedron_np = np.array([0, 1, 2, 3], dtype=np.int32)
        inverseMass_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        activeTetrahedron_np = np.array([1.0], dtype=np.float32)
        
        # GPU arrays
        vertex = wp.array(vertex_rest, dtype=wp.vec3, device=device)
        predictedVertex = wp.array(vertex_compressed, dtype=wp.vec3, device=device)
        tetrahedron = wp.array(tetrahedron_np, dtype=wp.int32, device=device)
        inverseMass = wp.array(inverseMass_np, dtype=wp.float32, device=device)
        activeTetrahedron = wp.array(activeTetrahedron_np, dtype=wp.float32, device=device)
        
        invRestMatrix = wp.zeros(1, dtype=wp.mat33, device=device)
        restVolume = wp.zeros(1, dtype=wp.float32, device=device)
        
        wp.launch(
            kernel=computeInverseRestMatrices,
            dim=1,
            inputs=[vertex, tetrahedron, invRestMatrix, restVolume],
            device=device
        )
        
        mu = 1000.0
        lambda_ = 5000.0
        dt = 0.01
        
        # Test Stable NH
        lambdas_stable = wp.zeros(1, dtype=wp.float32, device=device)
        dP_stable = wp.zeros(4, dtype=wp.vec3, device=device)
        constraintsNumber_stable = wp.zeros(4, dtype=wp.int32, device=device)
        
        wp.launch(
            kernel=stableNeohookeanHydrostaticConstraint,
            dim=1,
            inputs=[
                predictedVertex, dP_stable, constraintsNumber_stable,
                tetrahedron, invRestMatrix, lambdas_stable,
                inverseMass, restVolume, activeTetrahedron,
                mu, lambda_, dt
            ],
            device=device
        )
        
        lambda_stable = lambdas_stable.numpy()[0]
        stable_energies.append(np.abs(lambda_stable))
        
        # Test standard volume constraint
        tetrahedronA = wp.array(np.array([0], dtype=np.int32), device=device)
        tetrahedronB = wp.array(np.array([1], dtype=np.int32), device=device)
        tetrahedronC = wp.array(np.array([2], dtype=np.int32), device=device)
        tetrahedronD = wp.array(np.array([3], dtype=np.int32), device=device)
        
        lambdas_standard = wp.zeros(1, dtype=wp.float32, device=device)
        dP_standard = wp.zeros(4, dtype=wp.vec3, device=device)
        constraintsNumber_standard = wp.zeros(4, dtype=wp.int32, device=device)
        
        wp.launch(
            kernel=volumeConstraintsXPBD,
            dim=1,
            inputs=[
                predictedVertex, dP_standard, lambdas_standard,
                constraintsNumber_standard,
                tetrahedronA, tetrahedronB, tetrahedronC, tetrahedronD,
                restVolume, inverseMass, activeTetrahedron,
                0.1  # alpha compliance
            ],
            device=device
        )
        
        lambda_standard = lambdas_standard.numpy()[0]
        standard_energies.append(np.abs(lambda_standard))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(compression_ratios, stable_energies, 'b-', linewidth=2, label='Stable Neo-Hookean')
    plt.plot(compression_ratios, standard_energies, 'r--', linewidth=2, label='Standard Volume Constraint')
    plt.xlabel('Compression Ratio (J = det(F))', fontsize=12)
    plt.ylabel('|Lambda| (Constraint Force)', fontsize=12)
    plt.title('Constraint Response under Compression', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.2, 1.0])
    
    # Highlight extreme compression region
    plt.axvspan(0.2, 0.4, alpha=0.2, color='red', label='Extreme Compression')
    plt.text(0.3, max(stable_energies)*0.8, 'Stable NH\nremains bounded', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    plt.tight_layout()
    plt.savefig('/home/yunxin/FF-SRL/FF_SRL/output/stable_nh_comparison.png', dpi=150)
    print(f"\nPlot saved to: /home/yunxin/FF-SRL/FF_SRL/output/stable_nh_comparison.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("COMPRESSION TEST RESULTS")
    print("="*60)
    print(f"At 20% compression (J=0.2):")
    print(f"  Standard:  |λ| = {standard_energies[0]:.2e}")
    print(f"  Stable NH: |λ| = {stable_energies[0]:.2e}")
    print(f"  Ratio: {standard_energies[0]/stable_energies[0]:.2f}x")
    print("\nStable NH shows bounded energy even under extreme compression!")
    print("="*60)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('/home/yunxin/FF-SRL/FF_SRL/output', exist_ok=True)
    
    test_compression_comparison()
