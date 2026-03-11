#!/usr/bin/env python3
"""
Quick visualization of mesh quality metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from pxr import Usd, UsdGeom

def analyze_and_plot(usd_path):
    """Create visualization of mesh quality metrics"""
    
    stage = Usd.Stage.Open(usd_path)
    meshes = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simMesh").Get() == True]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Tissue Mesh Quality Analysis', fontsize=16, fontweight='bold')
    
    mesh_names = ['Liver', 'Fat with Artery', 'Gallbladder']
    
    for idx, (mesh_prim, name) in enumerate(zip(meshes, mesh_names)):
        # Get tetrahedral data
        tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
        vertices = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
        
        # Volume histogram (top row)
        ax = axes[0, idx]
        ax.hist(tet_vols, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Tetrahedral Volume (cm³)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name}\n{len(tet_vols)} Tetrahedra', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add statistics text
        mean_vol = tet_vols.mean()
        median_vol = np.median(tet_vols)
        min_vol = tet_vols.min()
        max_vol = tet_vols.max()
        quality_ratio = max_vol / min_vol if min_vol > 0 else float('inf')
        
        stats_text = f'Mean: {mean_vol:.4f}\nMedian: {median_vol:.4f}\nQuality: {quality_ratio:.0f}:1'
        ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3D point cloud (bottom row)
        ax = axes[1, idx]
        ax.scatter(vertices[:, 0], vertices[:, 2], c=vertices[:, 1], 
                   cmap='viridis', s=2, alpha=0.6)
        ax.set_xlabel('X (cm)', fontsize=10)
        ax.set_ylabel('Z (cm)', fontsize=10)
        ax.set_title(f'{name} Vertex Distribution\n{len(vertices)} Vertices', 
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add dimensions
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        dim_text = f'Size: {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} cm'
        ax.text(0.5, 0.02, dim_text, transform=ax.transAxes, 
                fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/yunxin/FF-SRL/tissue_quality_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /home/yunxin/FF-SRL/tissue_quality_analysis.png")
    
    # Print summary table
    print("\n" + "="*80)
    print("MESH QUALITY SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<30} {'Liver':<15} {'Fat':<15} {'Gallbladder':<15}")
    print("-"*80)
    
    for idx, (mesh_prim, name) in enumerate(zip(meshes, mesh_names)):
        if idx == 0:
            tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
            vertices = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
            edges = mesh_prim.GetAttribute("extMesh:edge").Get()
            tets = mesh_prim.GetAttribute("extMesh:elem").Get()
            
            print(f"{'Vertices':<30} {len(vertices):<15}", end='')
        elif idx == 1:
            tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
            vertices = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
            edges = mesh_prim.GetAttribute("extMesh:edge").Get()
            tets = mesh_prim.GetAttribute("extMesh:elem").Get()
            
            print(f"{len(vertices):<15}", end='')
        else:
            tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
            vertices = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
            edges = mesh_prim.GetAttribute("extMesh:edge").Get()
            tets = mesh_prim.GetAttribute("extMesh:elem").Get()
            
            print(f"{len(vertices):<15}")
    
    # Print each metric
    metrics = []
    for mesh_prim in meshes:
        tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
        vertices = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
        edges = mesh_prim.GetAttribute("extMesh:edge").Get()
        tets = mesh_prim.GetAttribute("extMesh:elem").Get()
        
        metrics.append({
            'vertices': len(vertices),
            'edges': len(edges) // 2,
            'tets': len(tets) // 4,
            'total_vol': tet_vols.sum(),
            'avg_vol': tet_vols.mean(),
            'quality': tet_vols.max() / tet_vols.min() if tet_vols.min() > 0 else float('inf')
        })
    
    print(f"{'Edges':<30} {metrics[0]['edges']:<15} {metrics[1]['edges']:<15} {metrics[2]['edges']:<15}")
    print(f"{'Tetrahedra':<30} {metrics[0]['tets']:<15} {metrics[1]['tets']:<15} {metrics[2]['tets']:<15}")
    print(f"{'Total Volume (cm³)':<30} {metrics[0]['total_vol']:<15.2f} {metrics[1]['total_vol']:<15.2f} {metrics[2]['total_vol']:<15.2f}")
    print(f"{'Avg Tet Volume (cm³)':<30} {metrics[0]['avg_vol']:<15.4f} {metrics[1]['avg_vol']:<15.4f} {metrics[2]['avg_vol']:<15.4f}")
    print(f"{'Quality Ratio':<30} {metrics[0]['quality']:<15.0f} {metrics[1]['quality']:<15.0f} {metrics[2]['quality']:<15.0f}")
    print("-"*80)
    
    # Quality assessment
    print("\nQUALITY ASSESSMENT:")
    for idx, (name, m) in enumerate(zip(mesh_names, metrics)):
        status = "⚠️  POOR" if m['quality'] > 10000 else "✅ GOOD" if m['quality'] < 1000 else "⚠️  FAIR"
        print(f"  {name:<20} Quality Ratio: {m['quality']:>10.0f}  →  {status}")
    
    print("\nRecommended Quality Ratio: < 100:1 (ideal), < 1000:1 (acceptable)")
    print("="*80)

if __name__ == "__main__":
    usd_path = "/home/yunxin/FF-SRL/FF_SRL/FF_SRL/scenes/liverRetractionTexture.usd"
    analyze_and_plot(usd_path)
