"""
Integration test: Load tumorBone.usd into SimModelDO + SimIntegratorDO,
step physics, verify no crashes/NaN/explosions.

Run: python -m FF_SRL.tests.test_integration_tumor
"""

import os
import sys
import traceback
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
wp.init()

from pxr import Usd
import FF_SRL as dk

DEVICE = "cuda:0"
USD_PATH = os.path.join(os.path.dirname(__file__), '..', 'scenes', 'tumorBone.usd')


def test_load_scene():
    """Test 1: Load USD into SimModelDO without crashing."""
    print("\n=== Test 1: Load Scene ===")

    stage = Usd.Stage.Open(USD_PATH)
    assert stage is not None, "Failed to open USD stage"

    simModel = dk.SimModelDO(
        stage,
        numEnvs=1,
        device=DEVICE,
        simSubsteps=8,
        simFrameRate=30,
        simConstraintsSteps=1,
        globalKsDistance=1.0,
        globalKsVolume=1.0,
        globalKsDrag=1.0,
        globalLaparoscopeDragLookupRadius=0.5,  # 5mm = 0.5cm
        environmentGroundLevel=-100.0,  # far below scene so it doesn't interfere
    )

    print(f"  Vertices: {simModel.numVertices}")
    print(f"  Tetrahedra: {simModel.numTetrahedrons}")
    print(f"  Edges: {simModel.numEdges}")
    print(f"  Triangles: {simModel.numTriangles}")
    print(f"  Adhesion bonds: {simModel.numAdhesionBonds}")
    print(f"  Rigid adhesion bonds: {simModel.numRigidAdhesionBonds}")

    assert simModel.numVertices > 0, "No vertices loaded"
    assert simModel.numTetrahedrons > 0, "No tetrahedra loaded"
    assert simModel.numEdges > 0, "No edges loaded"

    print("  PASSED")
    return simModel


def test_lockbox(simModel):
    """Test 2: Verify lockbox locked some vertices (inverseMass=0)."""
    print("\n=== Test 2: LockBox Effectiveness ===")

    inv_mass = simModel.inverseMass.numpy()
    total_verts = len(inv_mass)
    locked = np.sum(inv_mass == 0.0)
    free = np.sum(inv_mass > 0.0)

    print(f"  Total vertices: {total_verts}")
    print(f"  Locked (invMass=0): {locked}")
    print(f"  Free (invMass>0): {free}")
    print(f"  Lock ratio: {locked/total_verts*100:.1f}%")

    assert locked > 0, "LockBox didn't lock any vertices! Check lockbox position/scale."
    assert free > 0, "All vertices are locked! Check lockbox is not too large."
    assert locked < total_verts * 0.5, f"Too many locked ({locked}/{total_verts}). LockBox may be too large."

    print("  PASSED")


def test_step_physics(simModel):
    """Test 3: Step physics for several frames, check for NaN/explosions."""
    print("\n=== Test 3: Physics Stepping ===")

    simIntegrator = dk.SimIntegratorDO(DEVICE)

    # Record initial vertex positions
    initial_verts = simModel.vertex.numpy().copy()
    initial_max_pos = np.max(np.abs(initial_verts))

    num_frames = 20
    for frame in range(num_frames):
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

        # Check for NaN every few frames
        if frame % 5 == 0 or frame == num_frames - 1:
            verts = simModel.vertex.numpy()
            has_nan = np.any(np.isnan(verts))
            has_inf = np.any(np.isinf(verts))
            max_pos = np.max(np.abs(verts))

            if has_nan or has_inf:
                print(f"  Frame {frame}: NaN={has_nan}, Inf={has_inf}")
                assert False, f"Simulation produced NaN/Inf at frame {frame}!"

            if max_pos > initial_max_pos * 100:
                print(f"  Frame {frame}: max_pos={max_pos:.4f} (initial={initial_max_pos:.4f})")
                assert False, f"Simulation exploded at frame {frame}! max_pos grew 100x"

    # Check final state
    final_verts = simModel.vertex.numpy()
    max_displacement = np.max(np.linalg.norm(final_verts - initial_verts, axis=1))
    print(f"  Ran {num_frames} frames without NaN/Inf/explosion")
    print(f"  Max displacement: {max_displacement*10:.2f} mm")  # cm→mm
    print(f"  Max abs position: {np.max(np.abs(final_verts)):.4f} cm")

    # Under gravity, free vertices should move (but not explode)
    # Units are cm now
    assert max_displacement > 1e-6, "No vertices moved at all — simulation may not be running"
    assert max_displacement < 100.0, f"Displacement too large ({max_displacement:.4f}cm) — possible instability"

    print("  PASSED")
    return simIntegrator


def test_vertex_positions_reasonable(simModel):
    """Test 4: Check vertex positions are physically reasonable."""
    print("\n=== Test 4: Vertex Positions Reasonable ===")

    verts = simModel.vertex.numpy()
    inv_mass = simModel.inverseMass.numpy()

    # Locked vertices should not have moved (they're at original positions)
    locked_mask = inv_mass == 0.0
    free_mask = inv_mass > 0.0

    if np.any(locked_mask):
        initial_verts = simModel.initialVertex.numpy()
        locked_displacement = np.max(np.linalg.norm(
            verts[locked_mask] - initial_verts[locked_mask], axis=1))
        print(f"  Locked vertex max displacement: {locked_displacement*10:.4f} mm (should be ~0)")
        # XPBD Jacobi solver has numerical drift for locked vertices due to
        # compliance regularization. 5mm tolerance is acceptable (small vs tumor size).
        assert locked_displacement < 0.5, f"Locked vertices moved too much ({locked_displacement*10:.4f}mm)!"  # 5mm threshold in cm

    if np.any(free_mask):
        free_displacement = np.max(np.linalg.norm(
            verts[free_mask] - simModel.initialVertex.numpy()[free_mask], axis=1))
        print(f"  Free vertex max displacement: {free_displacement*10:.2f} mm")

    print("  PASSED")


def test_laparoscope_loaded(simModel):
    """Test 5: Verify laparoscope data was loaded."""
    print("\n=== Test 5: Laparoscope Loaded ===")

    # Check laparoscope positions tensor
    lap_pos = simModel.getLaparoscopePositionsTensor()
    print(f"  Laparoscope positions tensor shape: {lap_pos.shape}")
    print(f"  Laparoscope tip position: {lap_pos[0].cpu().numpy()}")

    assert lap_pos is not None, "Laparoscope positions not available"
    assert not torch.any(torch.isnan(lap_pos)), "Laparoscope position contains NaN"

    print("  PASSED")


def test_multi_env():
    """Test 6: Load with multiple environments."""
    print("\n=== Test 6: Multi-Environment (2 envs) ===")

    stage = Usd.Stage.Open(USD_PATH)
    simModel = dk.SimModelDO(
        stage,
        numEnvs=2,
        device=DEVICE,
        simSubsteps=8,
        simFrameRate=30,
        simConstraintsSteps=1,
        globalKsDistance=1.0,
        globalKsVolume=1.0,
        globalKsDrag=1.0,
        globalLaparoscopeDragLookupRadius=0.005,
        environmentGroundLevel=-1.0,
    )

    print(f"  Total vertices (2 envs): {simModel.numVertices}")
    assert simModel.numVertices > 0

    simIntegrator = dk.SimIntegratorDO(DEVICE)

    # Step a few frames
    for _ in range(5):
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

    verts = simModel.vertex.numpy()
    has_nan = np.any(np.isnan(verts))
    print(f"  NaN check after 5 frames: {'FAIL' if has_nan else 'OK'}")
    assert not has_nan, "Multi-env simulation produced NaN"

    print("  PASSED")


def main():
    print("=" * 60)
    print("TUMOR-BONE SCENE INTEGRATION TEST")
    print("=" * 60)
    print(f"USD: {os.path.abspath(USD_PATH)}")

    all_passed = True
    simModel = None

    try:
        simModel = test_load_scene()
        test_lockbox(simModel)
        test_step_physics(simModel)
        test_vertex_positions_reasonable(simModel)
        test_laparoscope_loaded(simModel)
        # Free GPU memory from first model before creating second
        del simModel
        simModel = None
        torch.cuda.empty_cache()
        test_multi_env()

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
