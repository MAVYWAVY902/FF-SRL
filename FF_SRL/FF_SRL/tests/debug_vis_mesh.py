"""
Debug: compare vis mesh before/after pointToVertex mapping.
Diagnose why tumor rendering looks broken.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warp as wp
wp.init()
from pxr import Usd
import FF_SRL as dk

DEVICE = "cuda:0"
USD_PATH = os.path.join(os.path.dirname(__file__), '..', 'scenes', 'tumorBone.usd')

stage = Usd.Stage.Open(USD_PATH)
simModel = dk.SimModelDO(
    stage, numEnvs=1, device=DEVICE,
    simSubsteps=8, simFrameRate=30, simConstraintsSteps=1,
    globalKsDistance=1.0, globalKsVolume=1.0, globalKsDrag=1.0,
    globalLaparoscopeDragLookupRadius=0.5, environmentGroundLevel=-100.0,
    globalAdhesionDContact=0.03, globalAdhesionDRest=0.32,
    globalAdhesionDNeutralStart=0.5, globalAdhesionBreakRatio=1.15,
    globalAdhesionStretchAbsMin=0.15, globalAdhesionAlpha=1e-6,
)

# After constructor, allVisPoint has been overwritten by transformSimMeshData
current_vis = simModel.allVisPoint.numpy()
initial_vis = simModel.allVisPointInitial.numpy()
n_mesh = simModel.numVisPoints

print(f"numVisPoints (tumor vis): {n_mesh}")
print(f"numVertices (sim tet): {simModel.numVertices}")
print(f"numEnvAllVertices: {simModel.numEnvAllVertices}")

# Compare initial vis points (from USD) vs current (after pointToVertex remap)
tumor_initial = initial_vis[:n_mesh]
tumor_current = current_vis[:n_mesh]
diffs = np.linalg.norm(tumor_current - tumor_initial, axis=1)
print(f"\n--- Tumor vis points: initial vs after remap ---")
print(f"Max diff: {diffs.max():.6f} cm")
print(f"Mean diff: {diffs.mean():.6f} cm")
print(f"Num with diff > 0.001: {np.sum(diffs > 0.001)}")
print(f"Num with diff > 0.01: {np.sum(diffs > 0.01)}")
print(f"Num with diff > 0.1: {np.sum(diffs > 0.1)}")

# Check pointToVertex mapping
ptv = simModel.pointToVertex.numpy()
print(f"\npointToVertex shape: {ptv.shape}")
print(f"pointToVertex range: [{ptv.min()}, {ptv.max()}]")
print(f"Unique entries: {len(np.unique(ptv[:n_mesh]))}")

# Check: does vertex[pointToVertex[i]] match initial visPoint[i]?
sim_verts = simModel.vertex.numpy()
print(f"\nvertex shape: {sim_verts.shape}")
for i in range(min(10, n_mesh)):
    vi = ptv[i]
    vis_pos = tumor_initial[i]
    sim_pos = sim_verts[vi]
    d = np.linalg.norm(vis_pos - sim_pos)
    print(f"  vis[{i}]={vis_pos} -> vertex[{vi}]={sim_pos}  diff={d:.6f}")

# Check for worst mismatches
mapped_pos = sim_verts[ptv[:n_mesh]]
map_diffs = np.linalg.norm(tumor_initial - mapped_pos, axis=1)
worst = np.argsort(map_diffs)[-10:]
print(f"\nWorst 10 pointToVertex mismatches:")
for idx in worst:
    vi = ptv[idx]
    print(f"  vis[{idx}]={tumor_initial[idx]} -> vertex[{vi}]={sim_verts[vi]}  diff={map_diffs[idx]:.6f}")

# Check vis face connectivity - are there cracks?
vis_faces = simModel.allVisFace.numpy()
num_mesh_faces = simModel.numEnvMeshesVisFaces
print(f"\n--- Face topology check ---")
print(f"Tumor vis faces: {num_mesh_faces}")

# Check for degenerate triangles
degen = 0
for i in range(num_mesh_faces):
    i0, i1, i2 = vis_faces[i*3], vis_faces[i*3+1], vis_faces[i*3+2]
    v0, v1, v2 = tumor_current[i0], tumor_current[i1], tumor_current[i2]
    area = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0))
    if area < 1e-10:
        degen += 1
print(f"Degenerate triangles (area~0): {degen}")

# Check edge sharing: count how many triangles share each edge
from collections import defaultdict
edge_count = defaultdict(int)
for i in range(num_mesh_faces):
    i0, i1, i2 = vis_faces[i*3], vis_faces[i*3+1], vis_faces[i*3+2]
    for e in [(min(i0,i1), max(i0,i1)), (min(i1,i2), max(i1,i2)), (min(i0,i2), max(i0,i2))]:
        edge_count[e] += 1

boundary = sum(1 for c in edge_count.values() if c == 1)
manifold = sum(1 for c in edge_count.values() if c == 2)
non_manifold = sum(1 for c in edge_count.values() if c > 2)
print(f"Edges: {len(edge_count)} total, {boundary} boundary, {manifold} manifold, {non_manifold} non-manifold")

print("\nDone.")
