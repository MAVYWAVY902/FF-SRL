"""
Visualize trained SAC peeling policy with GPU ray-traced rendering.

Renders the tumor peeling episode using FF-SRL's WarpRaycastRendererDO
and saves frames as an MP4 video.

Usage:
    python -m FF_SRL.visualize_peeling --model ./output/peeling_sac_20260311_230600/final_model.zip
    python -m FF_SRL.visualize_peeling --model ./output/peeling_sac_20260311_230600/best/best_model.zip
"""

import argparse
import os
import sys
import numpy as np
import torch

import warp as wp
wp.init()

from stable_baselines3 import SAC
from pxr import Usd
import FF_SRL as dk


def main():
    parser = argparse.ArgumentParser(description="Visualize peeling policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained SAC model (.zip)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--output", type=str, default="peeling_demo.mp4")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--action_strength", type=float, default=0.08)
    parser.add_argument("--no_video", action="store_true",
                        help="Show live window instead of saving video")
    args = parser.parse_args()

    device = args.device

    # --- Load scene & simulation (same as env) ---
    usd_path = os.path.join(os.path.dirname(__file__), 'scenes', 'tumorBone.usd')
    stage = Usd.Stage.Open(usd_path)

    simModel = dk.SimModelDO(
        stage,
        numEnvs=1,
        device=device,
        simSubsteps=16,
        simFrameRate=30,
        simConstraintsSteps=5,
        globalKsDistance=1.0,
        globalKsVolume=1.0,
        globalKsDrag=1.0,
        globalLaparoscopeDragLookupRadius=0.5,
        environmentGroundLevel=-100.0,
        globalAdhesionDContact=0.03,
        globalAdhesionDRest=0.32,
        globalAdhesionDNeutralStart=0.5,
        globalAdhesionBreakRatio=1.27,
        globalAdhesionStretchAbsMin=0.5,
        globalAdhesionAlpha=1e-6,
    )

    # Fix physics stiffness to match C++ codebase (E=8e4 Pa, ν=0.45)
    # Stable NH uses per-element α = 1/(μ*V₀*dt²). Scale μ/λ for cm units.
    simModel.globalMu = 1e7
    simModel.globalLambda = 5e7
    # Re-enable distance constraints as edge-level stretch protection
    simModel.globalDistanceCompliance = 0.1
    simModel.globalVolumeCompliance = 0.1
    # Velocity damping per substep (matches C++ damping-multiplier behavior)
    simModel.velocityDamping = 0.9

    # Create adhesion bonds
    simModel.createRigidAdhesionFromMeshes(bondDistance=0.5)
    total_bonds = simModel.numRigidAdhesionBonds
    print(f"Total adhesion bonds: {total_bonds}")

    simIntegrator = dk.SimIntegratorDO(device)

    # --- Fix pointToVertex mapping bug ---
    # The remapAToBKernel reads mapArray[tid*2] and mapArray[tid*2+1] (pair format),
    # but tumorBone.usd stores pointToVertex as simple [0,1,2,...,751] (identity).
    # This causes the kernel to read out of bounds, corrupting half the vis points.
    # Fix: rebuild pointToVertex in the correct interleaved pair format.
    ptv = simModel.pointToVertex.numpy()
    n_mesh_vis = simModel.numVisPoints
    if len(ptv) == n_mesh_vis:
        # It's simple format [vertIdx0, vertIdx1, ...], need pair format [visIdx0, vertIdx0, ...]
        pair_ptv = np.empty(n_mesh_vis * 2, dtype=np.int32)
        pair_ptv[0::2] = np.arange(n_mesh_vis, dtype=np.int32)  # vis point indices
        pair_ptv[1::2] = ptv  # vertex indices
        simModel.pointToVertex = wp.array(pair_ptv, dtype=wp.int32, device=device)
        # Re-sync vis points from sim vertices
        simModel.transformSimMeshData()
        print(f"Fixed pointToVertex: {n_mesh_vis} simple -> {n_mesh_vis*2} pair format")

        # Verify fix
        vis_after = simModel.allVisPoint.numpy()[:n_mesh_vis]
        vis_initial = simModel.allVisPointInitial.numpy()[:n_mesh_vis]
        max_diff = np.max(np.linalg.norm(vis_after - vis_initial, axis=1))
        print(f"  Vis point max diff after fix: {max_diff:.6f} cm (should be ~0)")

    # --- Fix tumor triangle winding order for correct rendering ---
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
    print(f"Fixed tumor triangle winding: flipped {flipped}/{num_mesh_faces} faces")
    simModel.allVisFace = wp.array(vis_faces, dtype=wp.int32, device=device)

    # --- BVH for rendering (rebuild after face fix) ---
    bvh_path = usd_path.replace(".usd", "_vis.npz")
    simBVH = dk.SimBVH(device, simModel, binsNumber=8, load=False, save=True, path=bvh_path)

    # --- Renderer ---
    # Figure out scene center from vertex positions
    all_verts = simModel.vertex.numpy()
    scene_center = all_verts.mean(axis=0)
    scene_max = all_verts.max(axis=0)
    scene_min = all_verts.min(axis=0)
    scene_extent = np.max(scene_max - scene_min)
    print(f"Scene center: {scene_center}")
    print(f"Scene extent: {scene_extent:.2f} cm")
    print(f"Scene bounds: {scene_min} -> {scene_max}")

    # Camera: positioned further back for a clear view
    cam_dist = scene_extent * 4.0
    cam_pos = [scene_center[0] + cam_dist * 0.3,
               scene_center[1] + cam_dist * 0.6,
               scene_center[2] + cam_dist]
    look_at = scene_center.tolist()
    print(f"Camera pos: {cam_pos}")
    print(f"Look at: {look_at}")

    renderer = dk.render.WarpRaycastRendererDO(
        device=device,
        simModel=simModel,
        resolution=args.resolution,
        cameraPos=cam_pos,
        cameraRot=[0.0, 0.0, 0.0],
        # Soft lighting from above-front
        lightPos=[scene_center[0], scene_center[1] + cam_dist, scene_center[2] + cam_dist * 0.5],
        lightIntensity=0.08,
        mode="headless",
    )
    # Use vertex_color mode: tumor=red, bone=white, laparoscope=blue
    renderer.render_mode = dk.render.RenderMode.vertex_color
    # Set colors programmatically
    colors = simModel.allVisPointColor.numpy()
    n_mesh = simModel.numVisPoints          # tumor vis points
    n_rigid = simModel.numEnvRigidsVisPoints
    n_lap = simModel.numEnvLaparoscopeVisPoints
    # Tumor: red-ish
    colors[:n_mesh] = [0.85, 0.25, 0.25]
    # Bone: light gray
    colors[n_mesh:n_mesh + n_rigid] = [0.9, 0.9, 0.9]
    # Laparoscope: steel blue
    colors[n_mesh + n_rigid:n_mesh + n_rigid + n_lap] = [0.4, 0.5, 0.7]
    simModel.allVisPointColor = wp.array(colors, dtype=wp.vec3, device=device)
    print(f"Set vertex colors: tumor({n_mesh})=red, bone({n_rigid})=gray, laparoscope({n_lap})=blue")
    renderer.lookAt(np.array(look_at))

    # Debug rendering setup
    print(f"numAllVisPoints: {simModel.numAllVisPoints}")
    print(f"numAllVisFaces: {simModel.numAllVisFaces}")
    print(f"numEnvMeshesVisFaces: {simModel.numEnvMeshesVisFaces}")
    print(f"numEnvRigidsVisFaces: {simModel.numEnvRigidsVisFaces}")
    print(f"numEnvLaparoscopeVisFaces: {simModel.numEnvLaparoscopeVisFaces}")
    print(f"numVisPoints (deformable mesh): {simModel.numVisPoints}")
    print(f"numVertices (sim): {simModel.numVertices}")

    # Check vis point positions
    vis_pts = simModel.allVisPoint.numpy()
    print(f"allVisPoint range: {vis_pts.min(axis=0)} -> {vis_pts.max(axis=0)}")
    print(f"allVisPoint has NaN: {np.any(np.isnan(vis_pts))}")

    # Check BVH bounds
    bvh_min = simBVH.aabbMin.numpy()[:3]
    bvh_max = simBVH.aabbMax.numpy()[:3]
    print(f"BVH root AABB: {bvh_min} -> {bvh_max}")

    # --- Grab tumor top (same as env) ---
    verts = simModel.vertex.numpy()
    inv_mass = simModel.inverseMass.numpy()
    free_indices = np.where(inv_mass > 0.0)[0]
    free_verts = verts[free_indices]
    top_local = np.argmax(free_verts[:, 1])
    grab_vertex_idx = int(free_indices[top_local])
    grab_pos = verts[grab_vertex_idx].copy()
    initial_verts = simModel.initialVertex.numpy().copy()

    # Position laparoscope and grab
    target = torch.tensor(
        [grab_pos[0], grab_pos[1] + 0.1, grab_pos[2]],
        dtype=torch.float32, device=device)
    lap_pos = simModel.getLaparoscopePositionsTensor()
    delta = target - lap_pos[0]
    action = torch.tensor(
        [delta[0].item(), delta[1].item(), delta[2].item()],
        dtype=torch.float32, device=device)
    simModel.applyCartesianActions(wp.from_torch(action))

    envs = torch.ones(1, dtype=torch.float32, device=device)
    simModel.forceLaparoscopeClampRegion(
        grab_vertex_idx, 0.5, envs, on=1.0, animate=True)

    # Settle (let tissue reach equilibrium with damping before rendering)
    for _ in range(30):
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

    # --- Load trained model ---
    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, device="cpu")
    print("Model loaded!")

    # --- Get initial obs ---
    def get_obs():
        lap_pos = simModel.getLaparoscopePositionsTensor()
        tip = lap_pos[0].cpu().numpy()
        active = simModel.rigidAdhesionActive.numpy()
        alive_ratio = float(np.sum(active > 0.5)) / max(total_bonds, 1)
        v = simModel.vertex.numpy()
        max_deform = float(np.max(np.linalg.norm(v - initial_verts, axis=1)))
        return np.array([tip[0], tip[1], tip[2], alive_ratio, max_deform],
                        dtype=np.float32)

    # --- Render loop ---
    frames = []
    obs = get_obs()
    prev_broken = total_bonds - int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))

    # Render initial frame
    simBVH.refitBVH()
    img_tensor = renderer.render(simBVH, simModel)
    if img_tensor is not None:
        raw = img_tensor[0].cpu().numpy()
        print(f"Render output: shape={raw.shape}, min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}")
        frame = (raw * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)
        # Save debug frame
        try:
            import imageio
            imageio.imwrite("debug_frame0.png", frame)
            print("Saved debug_frame0.png")
        except Exception as e:
            print(f"Could not save debug frame: {e}")
    else:
        print("WARNING: render returned None!")

    print(f"\nRunning policy for {args.max_steps} steps...")
    print(f"{'Step':>5} {'Broken':>8} {'%':>7} {'Reward':>8} {'Action':>30}")
    print("-" * 65)

    total_reward = 0
    for step in range(args.max_steps):
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        # Apply action
        scaled = action * args.action_strength
        action_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)
        simModel.applyCartesianActions(wp.from_torch(action_tensor))

        # Step physics
        simModel.resetCollisionInfo()
        simIntegrator.stepModel(simModel)

        # Get new state
        obs = get_obs()
        broken_now = total_bonds - int(np.sum(simModel.rigidAdhesionActive.numpy() > 0.5))
        new_breaks = broken_now - prev_broken
        break_ratio = broken_now / max(total_bonds, 1)

        reward = new_breaks - 0.01 * max(0, obs[4])
        if break_ratio >= 0.6:
            reward += 50.0
        total_reward += reward
        prev_broken = broken_now

        # Print progress
        if step % 10 == 0 or step == args.max_steps - 1 or break_ratio >= 0.6:
            action_str = f"[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]"
            print(f"{step:5d} {broken_now:8d} {break_ratio*100:6.1f}% {reward:8.2f} {action_str}")

        # Render frame
        simBVH.refitBVH()
        img_tensor = renderer.render(simBVH, simModel)
        if img_tensor is not None:
            frame = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(frame)

        if break_ratio >= 0.6:
            print(f"\n*** SUCCESS! Reached {break_ratio*100:.1f}% breakage at step {step} ***")
            # Render a few more frames to show the final state
            for _ in range(10):
                simBVH.refitBVH()
                img_tensor = renderer.render(simBVH, simModel)
                if img_tensor is not None:
                    frame = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    frames.append(frame)
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final breakage: {broken_now}/{total_bonds} ({break_ratio*100:.1f}%)")
    print(f"Frames rendered: {len(frames)}")

    # --- Save video ---
    if frames and not args.no_video:
        try:
            import imageio
            output_path = args.output
            print(f"\nSaving video to {output_path}...")
            writer = imageio.get_writer(output_path, fps=args.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved: {output_path} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
        except ImportError:
            print("imageio not installed. Saving frames as images instead...")
            os.makedirs("peeling_frames", exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(f"peeling_frames/frame_{i:04d}.png")
            print(f"Saved {len(frames)} frames to peeling_frames/")


if __name__ == "__main__":
    main()
