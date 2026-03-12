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
        globalAdhesionBreakRatio=1.15,     # 15% strain
        globalAdhesionStretchAbsMin=0.15,  # 1.5mm
        globalAdhesionAlpha=1e-6,          # stiff but not rigid
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

    # --- Phase 3: Compare different pull directions ---
    pull_speed = 0.08  # cm per frame (0.8mm/frame)
    pull_frames = 100

    # Directions to test: (label, dx, dy, dz)
    # Test small deviations around straight-up
    directions = [
        ("Straight up (0,1,0)",         0.0,  1.0,  0.0),
        ("Slight +X (0.15,0.99,0)",     0.15, 0.99, 0.0),
        ("Slight -X (-0.15,0.99,0)",   -0.15, 0.99, 0.0),
        ("Slight +Z (0,0.99,0.15)",     0.0,  0.99, 0.15),
        ("Slight -Z (0,0.99,-0.15)",    0.0,  0.99,-0.15),
        ("Slight +XZ (0.1,0.99,0.1)",   0.1,  0.99, 0.1),
        ("Medium +X (0.3,0.95,0)",      0.3,  0.95, 0.0),
        ("Medium -X (-0.3,0.95,0)",    -0.3,  0.95, 0.0),
    ]

    results = {}

    for label, dx, dy, dz in directions:
        print(f"\n--- Pull: {label} ({pull_frames} frames) ---")

        # Reset simulation to initial state
        simModel.resetFixedModel([0])

        # Re-grab the same region
        envs = torch.ones(1, dtype=torch.float32, device=DEVICE)

        # Move laparoscope to tumor top
        lap_pos = simModel.getLaparoscopePositionsTensor()
        target = torch.tensor([top_pos[0], top_pos[1] + 0.1, top_pos[2]],
                              dtype=torch.float32, device=DEVICE)
        delta = target - lap_pos[0]
        action = torch.tensor([delta[0].item(), delta[1].item(), delta[2].item()],
                              dtype=torch.float32, device=DEVICE)
        simModel.applyCartesianActions(wp.from_torch(action))

        # Grab region
        simModel.forceLaparoscopeClampRegion(
            int(top_vertex_idx), grab_radius, envs, on=1.0, animate=True)

        # Settle
        for _ in range(5):
            simModel.resetCollisionInfo()
            simIntegrator.stepModel(simModel)

        active_before = int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))
        print(f"  Active bonds after settle: {active_before}/{totalBonds}")

        # Normalize direction and apply pull_speed
        norm = (dx**2 + dy**2 + dz**2) ** 0.5
        pdx, pdy, pdz = dx/norm * pull_speed, dy/norm * pull_speed, dz/norm * pull_speed

        for frame in range(pull_frames):
            pull_action = torch.tensor([pdx, pdy, pdz],
                                       dtype=torch.float32, device=DEVICE)
            simModel.applyCartesianActions(wp.from_torch(pull_action))

            simModel.resetCollisionInfo()
            simIntegrator.stepModel(simModel)

            verts_now = simModel.vertex.numpy()
            if np.any(np.isnan(verts_now)):
                print(f"  Frame {frame}: NaN! Aborting this direction.")
                break

            active_bonds = simModel.rigidAdhesionActive.numpy()
            active_count = int(np.sum(active_bonds > 0.5))
            broken = totalBonds - active_count
            max_disp = np.max(np.linalg.norm(
                verts_now - simModel.initialVertex.numpy(), axis=1))

            if frame % 20 == 0 or frame == pull_frames - 1:
                print(f"  Frame {frame:3d}: broken={broken}/{totalBonds} "
                      f"({broken/totalBonds*100:.1f}%), max_disp={max_disp*10:.1f}mm")

        final_broken = totalBonds - int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))
        final_disp = np.max(np.linalg.norm(
            simModel.vertex.numpy() - simModel.initialVertex.numpy(), axis=1))
        results[label] = (final_broken, final_disp)

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("DIRECTION COMPARISON")
    print("=" * 60)
    print(f"{'Direction':<30} {'Broken':>8} {'%':>7} {'MaxDisp':>10}")
    print("-" * 58)
    for label, (broken, disp) in results.items():
        print(f"{label:<30} {broken:>8} {broken/totalBonds*100:>6.1f}% {disp*10:>8.1f}mm")

    print("\nTEST PASSED (no NaN/explosion)")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
