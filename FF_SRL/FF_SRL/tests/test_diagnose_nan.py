"""Diagnose NaN source in tumor simulation."""
import os, sys
import numpy as np

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
    simSubsteps=1,  # single substep for diagnosis
    simFrameRate=30, simConstraintsSteps=1,
    globalKsDistance=1.0, globalKsVolume=1.0,
    globalKsDrag=1.0,
    globalLaparoscopeDragLookupRadius=0.005,
    environmentGroundLevel=-1.0,
)

simIntegrator = dk.SimIntegratorDO(DEVICE)

# Check initial state
print("=== Initial State ===")
verts = simModel.vertex.numpy()
inv_mass = simModel.inverseMass.numpy()
print(f"Vertices: {verts.shape}, range: [{verts.min():.6f}, {verts.max():.6f}]")
print(f"InvMass: min={inv_mass.min():.6f}, max={inv_mass.max():.6f}, mean={inv_mass.mean():.6f}")
print(f"Any NaN in verts: {np.any(np.isnan(verts))}")
print(f"Any NaN in invMass: {np.any(np.isnan(inv_mass))}")

# Check rest volumes
rest_vol = simModel.tetrahedronRestVolume.numpy()
print(f"\nRest volumes: min={rest_vol.min():.2e}, max={rest_vol.max():.2e}")
print(f"Any zero rest vol: {np.sum(rest_vol == 0.0)}")
print(f"Any negative rest vol: {np.sum(rest_vol < 0.0)}")
print(f"Any NaN rest vol: {np.any(np.isnan(rest_vol))}")

# Check edge rest lengths
rest_len = simModel.edgeRestLength.numpy()
print(f"\nEdge rest lengths: min={rest_len.min():.2e}, max={rest_len.max():.2e}")
print(f"Any zero rest len: {np.sum(rest_len == 0.0)}")
print(f"Any NaN rest len: {np.any(np.isnan(rest_len))}")

# Check inverseMass at mesh level vs model level
simMesh = simModel.simEnvironment.simMeshes[0]
mesh_inv_mass = np.array(list(simMesh.inverseMass))
model_inv_mass = simModel.inverseMass.numpy()

print(f"\n=== InverseMass comparison ===")
print(f"simMesh.inverseMass type: {type(simMesh.inverseMass)}")
print(f"simMesh.inverseMass: len={len(mesh_inv_mass)}, zeros={np.sum(mesh_inv_mass==0)}, min={mesh_inv_mass.min():.4f}, max={mesh_inv_mass.max():.4f}")
print(f"simModel.inverseMass: len={len(model_inv_mass)}, zeros={np.sum(model_inv_mass==0)}, min={model_inv_mass.min():.4f}, max={model_inv_mass.max():.4f}")

# Step 20 frames (like the real test)
print("\n=== Stepping 20 frames ===")
initial_verts = simModel.vertex.numpy().copy()
initial_verts_init = simModel.initialVertex.numpy().copy()
print(f"vertex == initialVertex at start: {np.allclose(initial_verts, initial_verts_init)}")
print(f"max diff vertex vs initialVertex: {np.max(np.abs(initial_verts - initial_verts_init)):.6e}")

for frame in range(20):
    simModel.resetCollisionInfo()
    simIntegrator.stepModel(simModel)

verts_after = simModel.vertex.numpy()
model_inv_mass = simModel.inverseMass.numpy()
locked_mask = model_inv_mass == 0.0
free_mask = model_inv_mass > 0.0

print(f"\nAfter 20 frames:")
print(f"Any NaN: {np.any(np.isnan(verts_after))}")

# Check locked vertex displacement (vs initial_verts, not initialVertex)
if np.any(locked_mask):
    locked_disp_vs_start = np.max(np.linalg.norm(verts_after[locked_mask] - initial_verts[locked_mask], axis=1))
    locked_disp_vs_init = np.max(np.linalg.norm(verts_after[locked_mask] - initial_verts_init[locked_mask], axis=1))
    print(f"Locked vertex max disp (vs frame0 vertex): {locked_disp_vs_start*1000:.4f} mm")
    print(f"Locked vertex max disp (vs initialVertex): {locked_disp_vs_init*1000:.4f} mm")

    # Show a few locked vertices: initial vs final
    locked_indices = np.where(locked_mask)[0][:5]
    for idx in locked_indices:
        print(f"  v[{idx}]: initial={initial_verts[idx]}, final={verts_after[idx]}, initV={initial_verts_init[idx]}")

if np.any(free_mask):
    free_disp = np.max(np.linalg.norm(verts_after[free_mask] - initial_verts[free_mask], axis=1))
    print(f"Free vertex max disp: {free_disp*1000:.2f} mm")
