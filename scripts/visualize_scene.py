#!/usr/bin/env python3
"""
Interactive 3D visualization of the tumorBone.usd scene.
Drag mouse to rotate, scroll to zoom.

Usage:
    python scripts/visualize_scene.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Usd, UsdGeom
import os

USD_PATH = os.path.join(os.path.dirname(__file__), '..', 'FF_SRL', 'FF_SRL', 'scenes', 'tumorBone.usd')


def get_lockbox_corners(prim):
    translate = np.array(prim.GetAttribute("xformOp:translate").Get())
    scale = np.array(prim.GetAttribute("xformOp:scale").Get())
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=float)
    return corners * scale + translate


def draw_box(ax, corners, color='green', alpha=0.1):
    faces = [
        [corners[i] for i in idx]
        for idx in [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=1))


def make_cylinder_mesh(base, top, radius, n=12):
    """Create triangulated cylinder surface between base and top points."""
    direction = top - base
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return np.array([]), np.array([])
    d = direction / length

    # Find perpendicular vectors
    if abs(d[0]) < 0.9:
        perp1 = np.cross(d, [1, 0, 0])
    else:
        perp1 = np.cross(d, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)

    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    verts = []
    tris = []
    for a in angles:
        offset = radius * (np.cos(a) * perp1 + np.sin(a) * perp2)
        verts.append(base + offset)
        verts.append(top + offset)
    verts = np.array(verts)
    for i in range(n):
        b0, t0 = 2 * i, 2 * i + 1
        b1, t1 = 2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1
        tris.append([b0, b1, t0])
        tris.append([t0, b1, t1])
    return verts, np.array(tris)


def main():
    print(f"Loading: {os.path.abspath(USD_PATH)}")
    stage = Usd.Stage.Open(USD_PATH)

    # --- Tumor ---
    tumor_prim = stage.GetPrimAtPath("/World/Tumor")
    tumor_verts = np.array(tumor_prim.GetAttribute("extMesh:vertex").Get())
    tumor_vis_verts = np.array(tumor_prim.GetAttribute("points").Get())
    tumor_faces = np.array(tumor_prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
    inv_mass = np.array(tumor_prim.GetAttribute("extMesh:inverseMass").Get())
    locked_mask = inv_mass == 0.0
    free_mask = inv_mass > 0.0

    print(f"Tumor: {len(tumor_verts)} sim verts ({locked_mask.sum()} locked, {free_mask.sum()} free)")
    print(f"  bbox: [{tumor_verts.min(0)}] -> [{tumor_verts.max(0)}]")

    # --- Bone ---
    tbone_prim = stage.GetPrimAtPath("/World/TBone")
    tbone_verts = np.array(tbone_prim.GetAttribute("points").Get())
    tbone_faces_raw = np.array(tbone_prim.GetAttribute("faceVertexIndices").Get())
    # Handle mixed face counts
    face_counts = np.array(tbone_prim.GetAttribute("faceVertexCounts").Get())
    tbone_tris = []
    idx = 0
    for fc in face_counts:
        if fc == 3:
            tbone_tris.append(tbone_faces_raw[idx:idx+3])
        elif fc == 4:
            # Split quad into 2 triangles
            tbone_tris.append(tbone_faces_raw[idx:idx+3])
            tbone_tris.append([tbone_faces_raw[idx], tbone_faces_raw[idx+2], tbone_faces_raw[idx+3]])
        idx += fc
    tbone_tris = np.array(tbone_tris)
    print(f"Bone: {len(tbone_verts)} verts, {len(tbone_tris)} triangles")

    # --- LockBox ---
    lockbox_prim = stage.GetPrimAtPath("/World/LockBox")
    lb_corners = get_lockbox_corners(lockbox_prim)
    lb_t = np.array(lockbox_prim.GetAttribute("xformOp:translate").Get())
    lb_s = np.array(lockbox_prim.GetAttribute("xformOp:scale").Get())
    print(f"LockBox: center={lb_t}, scale={lb_s}")

    # --- Laparoscope ---
    inst_prim = stage.GetPrimAtPath("/World/Instrument")
    inst_t = np.array(inst_prim.GetAttribute("xformOp:translate").Get())
    rod_prim = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod")
    rod_t = np.array(rod_prim.GetAttribute("xformOp:translate").Get())
    rod_cap = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod/rodCapsule")
    rod_h = float(rod_cap.GetAttribute("height").Get())
    rod_r = float(rod_cap.GetAttribute("radius").Get())
    rod_world = inst_t + rod_t

    lc_rc = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod/leftClampRotationCenter")
    lc_t = np.array(lc_rc.GetAttribute("xformOp:translate").Get())
    lc_p = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod/leftClampRotationCenter/leftClamp")
    lc_local = np.array(lc_p.GetAttribute("xformOp:translate").Get())
    lc_h = float(lc_p.GetAttribute("height").Get())
    lc_r = float(lc_p.GetAttribute("radius").Get())

    rc_rc = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod/rightClampRotationCenter")
    rc_t = np.array(rc_rc.GetAttribute("xformOp:translate").Get())
    rc_p = stage.GetPrimAtPath("/World/Instrument/laparoscope/rotationCenter/rod/rightClampRotationCenter/rightClamp")
    rc_local = np.array(rc_p.GetAttribute("xformOp:translate").Get())
    rc_h = float(rc_p.GetAttribute("height").Get())
    rc_r = float(rc_p.GetAttribute("radius").Get())

    print(f"Laparoscope: inst={inst_t}, rod_center={rod_world}, rod_h={rod_h}cm, rod_r={rod_r}cm")

    # ========================
    # Interactive 3D plot
    # ========================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Tumor-Bone Scene (drag to rotate, scroll to zoom)\nUnits: cm", fontsize=13)

    # --- Draw Bone as point cloud ---
    ax.scatter(tbone_verts[:, 0], tbone_verts[:, 1], tbone_verts[:, 2],
               c='gray', s=3, alpha=0.5, label=f'Bone ({len(tbone_verts)})', depthshade=True)

    # --- Draw Tumor surface (vis mesh) ---
    tumor_tri_polys = [tumor_vis_verts[tumor_faces[i]] for i in range(len(tumor_faces))]
    ax.add_collection3d(Poly3DCollection(
        tumor_tri_polys, alpha=0.35, facecolor='salmon', edgecolor='darkred', linewidth=0.15))

    # --- Tumor sim verts: locked vs free ---
    if np.any(free_mask):
        ax.scatter(tumor_verts[free_mask, 0], tumor_verts[free_mask, 1], tumor_verts[free_mask, 2],
                   c='red', s=4, alpha=0.6, label=f'Tumor free ({free_mask.sum()})', depthshade=True)
    if np.any(locked_mask):
        ax.scatter(tumor_verts[locked_mask, 0], tumor_verts[locked_mask, 1], tumor_verts[locked_mask, 2],
                   c='blue', s=8, alpha=0.9, label=f'Tumor locked ({locked_mask.sum()})', depthshade=True)

    # --- LockBox ---
    draw_box(ax, lb_corners, color='limegreen', alpha=0.06)
    # Also draw edges more visibly
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot(*zip(lb_corners[i], lb_corners[j]), color='limegreen', linewidth=1.5, alpha=0.7)
    lb_center = lb_corners.mean(axis=0)
    ax.text(lb_center[0], lb_center[1], lb_center[2], 'LockBox', color='green', fontsize=9, fontweight='bold')

    # --- Laparoscope Rod (cylinder) ---
    rod_bottom = rod_world.copy(); rod_bottom[2] -= rod_h / 2
    rod_top = rod_world.copy();    rod_top[2] += rod_h / 2
    cyl_v, cyl_t = make_cylinder_mesh(rod_bottom, rod_top, rod_r, n=12)
    if len(cyl_t) > 0:
        rod_polys = [cyl_v[cyl_t[i]] for i in range(len(cyl_t))]
        ax.add_collection3d(Poly3DCollection(
            rod_polys, alpha=0.6, facecolor='orange', edgecolor='darkorange', linewidth=0.3))

    # --- Clamps (small cylinders) ---
    lc_world = rod_world + lc_t + lc_local
    lc_bot = lc_world.copy(); lc_bot[2] -= lc_h / 2
    lc_top = lc_world.copy(); lc_top[2] += lc_h / 2
    cv, ct = make_cylinder_mesh(lc_bot, lc_top, lc_r, n=8)
    if len(ct) > 0:
        ax.add_collection3d(Poly3DCollection(
            [cv[ct[i]] for i in range(len(ct))], alpha=0.7, facecolor='red', edgecolor='darkred', linewidth=0.2))

    rc_world = rod_world + rc_t + rc_local
    rc_bot = rc_world.copy(); rc_bot[2] -= rc_h / 2
    rc_top = rc_world.copy(); rc_top[2] += rc_h / 2
    cv, ct = make_cylinder_mesh(rc_bot, rc_top, rc_r, n=8)
    if len(ct) > 0:
        ax.add_collection3d(Poly3DCollection(
            [cv[ct[i]] for i in range(len(ct))], alpha=0.7, facecolor='crimson', edgecolor='darkred', linewidth=0.2))

    # --- Rotation center ---
    ax.scatter(*inst_t, color='yellow', s=80, marker='o', edgecolors='orange', linewidth=2,
               label='Rotation center', zorder=10)

    # --- Labels on objects ---
    ax.text(rod_top[0], rod_top[1], rod_top[2] + 0.3, 'Rod', color='darkorange', fontsize=9, fontweight='bold')
    ax.text(lc_world[0], lc_world[1] + 0.3, lc_world[2], 'L-Clamp', color='red', fontsize=8)
    ax.text(rc_world[0], rc_world[1] - 0.5, rc_world[2], 'R-Clamp', color='crimson', fontsize=8)

    # --- Equal aspect ratio ---
    all_pts = np.vstack([tumor_verts, tbone_verts, lb_corners,
                         [rod_bottom, rod_top, lc_world, rc_world, inst_t]])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.3
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.legend(fontsize=8, loc='upper left', markerscale=2)

    # Start with a nice view angle
    ax.view_init(elev=20, azim=-60)

    print("\nInteractive window opened. Drag to rotate, scroll to zoom. Close window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
