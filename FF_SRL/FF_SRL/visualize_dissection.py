"""
Visualize trained SAC dissection policy with GPU ray-traced rendering.

Usage:
    python -m FF_SRL.visualize_dissection --model ./output/dissection_sac_XXXXX/final_model.zip
    python -m FF_SRL.visualize_dissection --model ./output/dissection_sac_XXXXX/best/best_model.zip
"""

import argparse
import os
import numpy as np
import torch

import warp as wp
wp.init()

from stable_baselines3 import SAC
from pxr import Usd
import FF_SRL as dk


def main():
    parser = argparse.ArgumentParser(description="Visualize dissection policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAC model (.zip)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--output", type=str, default="dissection_demo.mp4")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--action_strength", type=float, default=0.04)
    parser.add_argument("--push_radius", type=float, default=0.6)
    parser.add_argument("--push_strength", type=float, default=0.08)
    args = parser.parse_args()

    device = args.device

    # --- Load scene & simulation ---
    usd_path = os.path.join(os.path.dirname(__file__), 'scenes', 'tumorBone.usd')
    stage = Usd.Stage.Open(usd_path)

    simModel = dk.SimModelDO(
        stage, numEnvs=1, device=device,
        simSubsteps=16, simFrameRate=30, simConstraintsSteps=5,
        globalKsDistance=1.0, globalKsVolume=1.0, globalKsDrag=1.0,
        globalLaparoscopeDragLookupRadius=0.5,
        environmentGroundLevel=-100.0,
        globalAdhesionDContact=0.03, globalAdhesionDRest=0.32,
        globalAdhesionDNeutralStart=0.5, globalAdhesionBreakRatio=1.27,
        globalAdhesionStretchAbsMin=0.5, globalAdhesionAlpha=1e-6,
    )

    simModel.globalMu = 1e7
    simModel.globalLambda = 5e7
    simModel.globalDistanceCompliance = 0.1
    simModel.globalVolumeCompliance = 0.1
    simModel.velocityDamping = 0.9

    simModel.createRigidAdhesionFromMeshes(bondDistance=0.25)
    total_bonds = simModel.numRigidAdhesionBonds
    print(f"Total adhesion bonds: {total_bonds}")

    # Store bond rigid points for local observation
    bond_rigid_points = simModel.rigidAdhesionRigidPoint.numpy().copy()
    n_per_env = simModel.numRigidAdhesionBondsPerEnv

    simIntegrator = dk.SimIntegratorDO(device)
    initial_verts = simModel.initialVertex.numpy().copy()

    # Compute bone top Y
    rigidVerts = []
    for simRigid in simModel.simEnvironment.simRigids:
        rigidVerts.append(np.array(simRigid.vertex))
    bone_top_y = float(np.vstack(rigidVerts)[:, 1].max())

    # --- Find interface edge (same as env) ---
    bond_pts = bond_rigid_points[:n_per_env]
    centroid = bond_pts.mean(axis=0)
    xz_dist = np.sqrt((bond_pts[:, 0] - centroid[0])**2 +
                       (bond_pts[:, 2] - centroid[2])**2)
    edge_idx = np.argmax(xz_dist)
    edge_pt = bond_pts[edge_idx].copy()
    direction_xz = edge_pt - centroid
    direction_xz[1] = 0.0
    norm = np.linalg.norm(direction_xz)
    if norm > 1e-6:
        direction_xz /= norm
    start_pos = edge_pt + direction_xz * 0.3
    start_pos[1] = bone_top_y + 0.2
    print(f"Dissector start: {start_pos}")

    # --- Position dissector at interface edge ---
    target = torch.tensor(start_pos, dtype=torch.float32, device=device)
    lap_pos = simModel.getLaparoscopePositionsTensor()
    delta = target - lap_pos[0]
    action = torch.tensor(
        [delta[0].item(), delta[1].item(), delta[2].item()],
        dtype=torch.float32, device=device)
    simModel.applyCartesianActions(wp.from_torch(action))

    # Setup push arrays
    tip_pos = start_pos.copy()
    simModel.pushTipPos = wp.array(
        [wp.vec3(tip_pos[0], tip_pos[1], tip_pos[2])],
        dtype=wp.vec3, device=device)
    simModel.pushRadius = args.push_radius
    simModel.pushStrength = args.push_strength

    # Settle (push disabled during settle)
    simModel.pushTipPos = None
    original_damping = simModel.velocityDamping
    simModel.velocityDamping = 0.5
    for _ in range(50):
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)
    simModel.velocityDamping = original_damping

    # Re-enable push
    simModel.pushTipPos = wp.array(
        [wp.vec3(tip_pos[0], tip_pos[1], tip_pos[2])],
        dtype=wp.vec3, device=device)

    # --- Fix pointToVertex mapping ---
    ptv = simModel.pointToVertex.numpy()
    n_mesh_vis = simModel.numVisPoints
    if len(ptv) == n_mesh_vis:
        pair_ptv = np.empty(n_mesh_vis * 2, dtype=np.int32)
        pair_ptv[0::2] = np.arange(n_mesh_vis, dtype=np.int32)
        pair_ptv[1::2] = ptv
        simModel.pointToVertex = wp.array(pair_ptv, dtype=wp.int32, device=device)
        simModel.transformSimMeshData()
        print(f"Fixed pointToVertex: {n_mesh_vis} simple -> {n_mesh_vis*2} pair format")

    # --- Fix triangle winding ---
    vis_pts = simModel.allVisPoint.numpy()
    vis_faces = simModel.allVisFace.numpy()
    num_mesh_faces = simModel.numEnvMeshesVisFaces
    tumor_vis_pts = vis_pts[:n_mesh_vis]
    tumor_center = tumor_vis_pts.mean(axis=0)
    flipped = 0
    for i in range(num_mesh_faces):
        i0, i1, i2 = vis_faces[i*3], vis_faces[i*3+1], vis_faces[i*3+2]
        v0, v1, v2 = vis_pts[i0], vis_pts[i1], vis_pts[i2]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_center = (v0 + v1 + v2) / 3.0
        outward = face_center - tumor_center
        if np.dot(face_normal, outward) < 0:
            vis_faces[i*3+1], vis_faces[i*3+2] = vis_faces[i*3+2], vis_faces[i*3+1]
            flipped += 1
    print(f"Fixed triangle winding: flipped {flipped}/{num_mesh_faces} faces")
    simModel.allVisFace = wp.array(vis_faces, dtype=wp.int32, device=device)

    # --- BVH & Renderer ---
    bvh_path = usd_path.replace(".usd", "_vis.npz")
    simBVH = dk.SimBVH(device, simModel, binsNumber=8, load=False, save=True, path=bvh_path)

    all_verts = simModel.vertex.numpy()
    scene_center = all_verts.mean(axis=0)
    scene_extent = np.max(all_verts.max(axis=0) - all_verts.min(axis=0))
    cam_dist = scene_extent * 4.0
    cam_pos = [scene_center[0] + cam_dist * 0.3,
               scene_center[1] + cam_dist * 0.6,
               scene_center[2] + cam_dist]
    look_at = scene_center.tolist()

    renderer = dk.render.WarpRaycastRendererDO(
        device=device, simModel=simModel, resolution=args.resolution,
        cameraPos=cam_pos, cameraRot=[0.0, 0.0, 0.0],
        lightPos=[scene_center[0], scene_center[1] + cam_dist, scene_center[2] + cam_dist * 0.5],
        lightIntensity=0.08, mode="headless",
    )
    renderer.render_mode = dk.render.RenderMode.vertex_color

    # Set colors
    colors = simModel.allVisPointColor.numpy()
    n_mesh = simModel.numVisPoints
    n_rigid = simModel.numEnvRigidsVisPoints
    n_lap = simModel.numEnvLaparoscopeVisPoints
    colors[:n_mesh] = [0.85, 0.25, 0.25]
    colors[n_mesh:n_mesh + n_rigid] = [0.9, 0.9, 0.9]
    colors[n_mesh + n_rigid:n_mesh + n_rigid + n_lap] = [0.4, 0.5, 0.7]
    simModel.allVisPointColor = wp.array(colors, dtype=wp.vec3, device=device)
    renderer.lookAt(np.array(look_at))

    # --- Load trained model ---
    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, device="cpu")
    print("Model loaded!")

    # --- Observation helper ---
    def get_obs():
        lp = simModel.getLaparoscopePositionsTensor()
        tip = lp[0].cpu().numpy()
        active = simModel.rigidAdhesionActive.numpy()
        alive_ratio = float(np.sum(active > 0.5)) / max(total_bonds, 1)
        v = simModel.vertex.numpy()
        max_deform = float(np.max(np.linalg.norm(v - initial_verts, axis=1)))

        # Local broken ratio
        bp = bond_rigid_points[:n_per_env]
        act = active[:n_per_env]
        dists = np.linalg.norm(bp - tip, axis=1)
        nearby = dists < 1.0
        n_nearby = int(np.sum(nearby))
        local_broken = int(np.sum((nearby) & (act < 0.5))) / max(n_nearby, 1)

        tip_above_bone = tip[1] - bone_top_y

        return np.array([tip[0], tip[1], tip[2], alive_ratio, max_deform,
                         local_broken, tip_above_bone], dtype=np.float32)

    # --- Render loop ---
    frames = []
    obs = get_obs()
    prev_broken = total_bonds - int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))

    # Render initial frame
    simBVH.refitBVH()
    img_tensor = renderer.render(simBVH, simModel)
    if img_tensor is not None:
        frame = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)
        try:
            import imageio
            imageio.imwrite("debug_dissection_frame0.png", frame)
            print("Saved debug_dissection_frame0.png")
        except Exception:
            pass

    print(f"\nRunning dissection policy for {args.max_steps} steps...")
    print(f"{'Step':>5} {'Broken':>8} {'%':>7} {'NewBrk':>7} {'TipAbvBone':>11} {'Action':>30}")
    print("-" * 75)

    for step in range(args.max_steps):
        action_raw, _ = model.predict(obs, deterministic=True)
        action_raw = np.clip(action_raw, -1.0, 1.0)

        # Apply action to laparoscope
        scaled = action_raw * args.action_strength
        action_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
        simModel.applyCartesianActions(wp.from_torch(action_tensor))

        # Update push tip position
        lp = simModel.getLaparoscopePositionsTensor()
        tip_torch = lp[0].cpu().numpy()
        tip_pos = tip_torch.copy()
        simModel.pushTipPos = wp.array(
            [wp.vec3(tip_pos[0], tip_pos[1], tip_pos[2])],
            dtype=wp.vec3, device=device)

        # Step physics
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

        # Get new state
        obs = get_obs()
        broken_now = total_bonds - int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))
        new_breaks = broken_now - prev_broken
        break_ratio = broken_now / max(total_bonds, 1)
        prev_broken = broken_now

        if step % 10 == 0 or step == args.max_steps - 1 or break_ratio >= 0.6:
            action_str = f"[{action_raw[0]:+.3f}, {action_raw[1]:+.3f}, {action_raw[2]:+.3f}]"
            print(f"{step:5d} {broken_now:8d} {break_ratio*100:6.1f}% {new_breaks:7d} {obs[6]:+10.3f} {action_str}")

        # Render
        simBVH.refitBVH()
        img_tensor = renderer.render(simBVH, simModel)
        if img_tensor is not None:
            frame = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(frame)

        if break_ratio >= 0.6:
            print(f"\n*** SUCCESS! Reached {break_ratio*100:.1f}% breakage at step {step} ***")
            for _ in range(10):
                simBVH.refitBVH()
                img_tensor = renderer.render(simBVH, simModel)
                if img_tensor is not None:
                    frame = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    frames.append(frame)
            break

    print(f"\nFinal breakage: {broken_now}/{total_bonds} ({break_ratio*100:.1f}%)")
    print(f"Frames rendered: {len(frames)}")

    # --- Save video ---
    if frames:
        try:
            import imageio
            print(f"\nSaving video to {args.output}...")
            writer = imageio.get_writer(args.output, fps=args.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved: {args.output} ({len(frames)} frames)")
        except ImportError:
            os.makedirs("dissection_frames", exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(f"dissection_frames/frame_{i:04d}.png")
            print(f"Saved {len(frames)} frames to dissection_frames/")


if __name__ == "__main__":
    main()
