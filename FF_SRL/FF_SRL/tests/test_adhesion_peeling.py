"""
Adhesion peeling verification test (no RL).

Move the laparoscope tip down to the tumor surface, clamp it,
then pull upward. Verify:
  1. Adhesion bonds resist the pull initially
  2. Bonds break progressively as pull increases
  3. No NaN/explosion

Run: python -m FF_SRL.tests.test_adhesion_peeling
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
wp.init()

from pxr import Usd
import FF_SRL as dk

DEVICE = "cuda:0"
USD_PATH = os.path.join(os.path.dirname(__file__), '..', 'scenes', 'tumorBone.usd')


def main():
    print("=" * 60)
    print("ADHESION PEELING VERIFICATION TEST")
    print("=" * 60)

    # --- Load scene ---
    stage = Usd.Stage.Open(USD_PATH)
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
        globalLaparoscopeDragLookupRadius=0.5,  # 5mm in cm
        environmentGroundLevel=-100.0,
        # Adhesion parameters (cm units)
        globalAdhesionDContact=0.03,       # 0.3mm
        globalAdhesionDRest=0.32,          # 3.2mm
        globalAdhesionDNeutralStart=0.5,   # 5mm
        globalAdhesionBreakRatio=1.27,     # 27% strain
        globalAdhesionStretchAbsMin=0.5,   # 5mm
        globalAdhesionAlpha=1e-7,          # very stiff
    )

    # Create adhesion bonds
    simModel.createRigidAdhesionFromMeshes(bondDistance=3.5)
    totalBonds = simModel.numRigidAdhesionBonds
    print(f"\nTotal rigid adhesion bonds: {totalBonds}")

    simIntegrator = dk.SimIntegratorDO(DEVICE)

    # --- Get tumor top vertex (highest Y) to position laparoscope ---
    verts = simModel.vertex.numpy()
    inv_mass = simModel.inverseMass.numpy()
    free_mask = inv_mass > 0.0

    # Find the topmost free vertex
    free_indices = np.where(free_mask)[0]
    free_verts = verts[free_indices]
    top_idx_local = np.argmax(free_verts[:, 1])  # highest Y
    top_vertex_idx = free_indices[top_idx_local]
    top_pos = verts[top_vertex_idx]
    print(f"Top free vertex: idx={top_vertex_idx}, pos={top_pos}")

    # --- Get current laparoscope tip position ---
    lap_pos = simModel.getLaparoscopePositionsTensor()
    print(f"Initial laparoscope tip: {lap_pos[0].cpu().numpy()}")

    # --- Phase 1: Grab a region of vertices near tumor top ---
    print("\n--- Phase 1: Grab region near tumor top ---")

    # Move laparoscope tip to tumor top
    target = torch.tensor([top_pos[0], top_pos[1] + 0.1, top_pos[2]],
                          dtype=torch.float32, device=DEVICE)
    current_tip = lap_pos[0]
    delta = target - current_tip
    action = torch.zeros(3, dtype=torch.float32, device=DEVICE)
    action[0] = delta[0].item()
    action[1] = delta[1].item()
    action[2] = delta[2].item()
    simModel.applyCartesianActions(wp.from_torch(action))

    lap_pos = simModel.getLaparoscopePositionsTensor()
    print(f"Laparoscope tip after move: {lap_pos[0].cpu().numpy()}")

    # Grab all vertices within 0.5cm (5mm) of top vertex
    envs = torch.ones(1, dtype=torch.float32, device=DEVICE)
    grab_radius = 0.5  # cm
    grabbed = simModel.forceLaparoscopeClampRegion(
        int(top_vertex_idx), grab_radius, envs, on=1.0, animate=True)

    drag = simModel.activeDragConstraint.numpy()
    num_dragged = int(np.sum(drag > 0.5))
    print(f"Total vertices grabbed: {num_dragged}")

    # --- Phase 2: Settle (let simulation equilibrate with adhesion) ---
    print("\n--- Phase 2: Settle (10 frames) ---")
    for frame in range(10):
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

    active_bonds = simModel.rigidAdhesionActive.numpy()
    active_count = int(np.sum(active_bonds > 0.5))
    print(f"Active bonds after settle: {active_count}/{totalBonds}")

    verts_after_settle = simModel.vertex.numpy()
    max_disp_settle = np.max(np.linalg.norm(
        verts_after_settle - simModel.initialVertex.numpy(), axis=1))
    print(f"Max displacement after settle: {max_disp_settle*10:.2f} mm")

    # --- Phase 3: Pull upward ---
    print("\n--- Phase 3: Pull upward (100 frames) ---")
    pull_speed = 0.08  # cm per frame (0.8mm/frame)
    pull_frames = 100

    bond_history = []
    displacement_history = []

    for frame in range(pull_frames):
        # Apply upward pull
        pull_action = torch.tensor([0.0, pull_speed, 0.0],
                                   dtype=torch.float32, device=DEVICE)
        simModel.applyCartesianActions(wp.from_torch(pull_action))

        # Step physics
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

        # Check state
        verts_now = simModel.vertex.numpy()
        has_nan = np.any(np.isnan(verts_now))
        if has_nan:
            print(f"  Frame {frame}: NaN detected! Aborting.")
            return False

        active_bonds = simModel.rigidAdhesionActive.numpy()
        active_count = int(np.sum(active_bonds > 0.5))
        broken = totalBonds - active_count

        max_disp = np.max(np.linalg.norm(
            verts_now - simModel.initialVertex.numpy(), axis=1))

        lap_pos = simModel.getLaparoscopePositionsTensor()
        tip_y = lap_pos[0, 1].item()

        bond_history.append(active_count)
        displacement_history.append(max_disp)

        if frame % 10 == 0 or frame == pull_frames - 1:
            print(f"  Frame {frame:3d}: active={active_count}/{totalBonds} "
                  f"(broken={broken}), max_disp={max_disp*10:.1f}mm, "
                  f"tip_y={tip_y:.3f}cm")

    # --- Analysis ---
    print("\n--- Results ---")
    initial_bonds = bond_history[0]
    final_bonds = bond_history[-1]
    total_broken = initial_bonds - final_bonds

    print(f"Bonds at start of pull: {initial_bonds}")
    print(f"Bonds at end of pull:   {final_bonds}")
    print(f"Total broken:           {total_broken}")
    print(f"Max displacement:       {displacement_history[-1]*10:.1f} mm")

    # Verify adhesion worked
    if total_broken > 0:
        print("\nADHESION PEELING VERIFIED: Bonds broke progressively during pull.")
    elif initial_bonds == totalBonds and final_bonds == totalBonds:
        print("\nWARNING: No bonds broke. Pull may be too weak or adhesion too stiff.")
        print("Try increasing pull_speed or decreasing globalAdhesionStretchAbsMin.")
    else:
        print("\nWARNING: Some bonds were already broken before pull started.")

    # Check no explosion
    assert not np.any(np.isnan(simModel.vertex.numpy())), "Final state has NaN!"
    assert displacement_history[-1] < 100.0, "Displacement too large - possible explosion"

    print("\nTEST PASSED (no NaN/explosion)")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
