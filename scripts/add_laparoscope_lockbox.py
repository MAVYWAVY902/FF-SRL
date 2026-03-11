#!/usr/bin/env python3
"""
Add a properly-scaled laparoscope and lockbox to the tumorBone.usd scene.

The tumor/tbone meshes are in SI meters. The original laparoscope from
liverRetractionTexture.usd is in cm-like units, so we scale all dimensions
by ~0.001 and position the tool above the tumor.

Usage:
    python scripts/add_laparoscope_lockbox.py
"""

import numpy as np
from pxr import Usd, UsdGeom, Sdf, Vt, Gf

USD_PATH = "/home/yunxin/FF-SRL/FF_SRL/FF_SRL/scenes/tumorBone.usd"


def get_mesh_bounds(stage, prim_path):
    """Get bounding box of a mesh prim."""
    prim = stage.GetPrimAtPath(prim_path)
    pts = np.array(prim.GetAttribute("points").Get())
    return pts.min(axis=0), pts.max(axis=0)


def add_laparoscope(stage, tumor_min, tumor_max):
    """Add a capsule-based laparoscope (no mesh geometry) scaled for meter-unit scene.

    Hierarchy:
      /World/Instrument (Xform) - positioned above tumor
        /laparoscope (Xform, simLaparoscope=True)
          /rotationCenter (Xform)
            /rotationCenterSphere (Sphere, simLaparoscopeRotationCenter=True)
            /rod (Xform, simLaparoscopeRod=True)
              /rodCapsule (Capsule, simLaparoscopeRodCapsule=True)
              /leftClampRotationCenter (Xform)
                /leftClamp (Capsule, simLaparoscopeLeftClamp=True)
              /rightClampRotationCenter (Xform)
                /rightClamp (Capsule, simLaparoscopeRightClamp=True)
    """
    tumor_center = (tumor_min + tumor_max) / 2.0
    tumor_size = tumor_max - tumor_min

    # Physically realistic dimensions for a dVRK-like instrument (in cm, matching scene units).
    # Real dVRK: 8mm shaft diameter, ~300mm shaft, 8-12mm grasper jaws.
    # We only model the visible portion near the tissue.
    rod_height = 4.0           # 40mm = 4.0 cm
    rod_radius = 0.25          # 2.5mm = 0.25 cm
    clamp_height = 0.8         # 8mm = 0.8 cm
    clamp_radius = 0.15        # 1.5mm = 0.15 cm
    sphere_radius = 0.3        # 3mm = 0.3 cm

    # Position instrument so the rod tip is just above the tumor top.
    # The rod extends along Z in local space (center at z=0, tip at z=+height/2).
    instrument_pos = Gf.Vec3d(
        float(tumor_center[0]),
        float(tumor_max[1]) + rod_height * 0.3,  # above tumor
        float(tumor_center[2])
    )

    rod_z_offset = -1.2  # rod offset along Z (cm)

    # Clamp positions (at tip of rod)
    clamp_z = rod_height / 2.0 + 0.1

    print(f"  Instrument position: ({instrument_pos[0]:.4f}, {instrument_pos[1]:.4f}, {instrument_pos[2]:.4f})")
    print(f"  Rod: height={rod_height*10:.1f}mm, radius={rod_radius*10:.2f}mm")
    print(f"  Clamp: height={clamp_height*10:.1f}mm, radius={clamp_radius*10:.2f}mm")

    def make_xform(stage, path):
        xf = UsdGeom.Xform.Define(stage, path)
        prim = xf.GetPrim()
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
        prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
        prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
        xf.GetOrderedXformOps()  # Force creation of xformOpOrder
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
            ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
        return prim

    # /World/Instrument
    inst_prim = make_xform(stage, "/World/Instrument")
    inst_prim.GetAttribute("xformOp:translate").Set(instrument_pos)

    # /World/Instrument/laparoscope
    lap_prim = make_xform(stage, "/World/Instrument/laparoscope")
    lap_prim.CreateAttribute("simLaparoscope", Sdf.ValueTypeNames.Bool).Set(True)
    lap_prim.CreateAttribute("hasLaparoscopeMesh", Sdf.ValueTypeNames.Bool).Set(False)
    lap_prim.CreateAttribute("rodColor", Sdf.ValueTypeNames.Color4f).Set(Gf.Vec4f(1, 0.45, 0.3, 1))
    lap_prim.CreateAttribute("leftClampColor", Sdf.ValueTypeNames.Color4f).Set(Gf.Vec4f(1, 0, 0, 1))
    lap_prim.CreateAttribute("rightClampColor", Sdf.ValueTypeNames.Color4f).Set(Gf.Vec4f(1, 0, 0, 1))

    # rotationCenter
    rc_prim = make_xform(stage, "/World/Instrument/laparoscope/rotationCenter")

    # rotationCenterSphere
    sphere = UsdGeom.Sphere.Define(stage, "/World/Instrument/laparoscope/rotationCenter/rotationCenterSphere")
    sp = sphere.GetPrim()
    sp.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    sp.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    sp.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
    sp.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
    sphere.GetRadiusAttr().Set(sphere_radius)
    sp.CreateAttribute("simLaparoscopeRotationCenter", Sdf.ValueTypeNames.Bool).Set(True)

    # rod
    rod_prim = make_xform(stage, "/World/Instrument/laparoscope/rotationCenter/rod")
    rod_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, rod_z_offset))
    rod_prim.CreateAttribute("simLaparoscopeRod", Sdf.ValueTypeNames.Bool).Set(True)

    # rodCapsule
    rod_capsule = UsdGeom.Capsule.Define(stage, "/World/Instrument/laparoscope/rotationCenter/rod/rodCapsule")
    rcp = rod_capsule.GetPrim()
    rcp.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    rcp.CreateAttribute("xformOp:rotateZYX", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    rcp.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
    rcp.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"])
    rcp.CreateAttribute("height", Sdf.ValueTypeNames.Float).Set(rod_height)
    rcp.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(rod_radius)
    rcp.CreateAttribute("simLaparoscopeRodCapsule", Sdf.ValueTypeNames.Bool).Set(True)

    # leftClampRotationCenter
    lcrc_prim = make_xform(stage, "/World/Instrument/laparoscope/rotationCenter/rod/leftClampRotationCenter")
    lcrc_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, clamp_z))
    # Use rotateZYX for clamp rotation centers (matching original)
    lcrc_prim.GetAttribute("xformOp:rotateXYZ").Clear()
    lcrc_prim.CreateAttribute("xformOp:rotateZYX", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    lcrc_prim.GetAttribute("xformOpOrder").Set(
        ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"])
    lcrc_prim.CreateAttribute("simLaparoscopeLeftClampRotationCenter", Sdf.ValueTypeNames.Bool).Set(True)

    # leftClamp
    lc = UsdGeom.Capsule.Define(stage, "/World/Instrument/laparoscope/rotationCenter/rod/leftClampRotationCenter/leftClamp")
    lcp = lc.GetPrim()
    lcp.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0.05, 0.05))
    lcp.CreateAttribute("xformOp:rotateZYX", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    lcp.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
    lcp.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"])
    lcp.CreateAttribute("height", Sdf.ValueTypeNames.Float).Set(clamp_height)
    lcp.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(clamp_radius)
    lcp.CreateAttribute("simLaparoscopeLeftClamp", Sdf.ValueTypeNames.Bool).Set(True)

    # rightClampRotationCenter
    rcrc_prim = make_xform(stage, "/World/Instrument/laparoscope/rotationCenter/rod/rightClampRotationCenter")
    rcrc_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, clamp_z))
    rcrc_prim.GetAttribute("xformOp:rotateXYZ").Clear()
    rcrc_prim.CreateAttribute("xformOp:rotateZYX", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    rcrc_prim.GetAttribute("xformOpOrder").Set(
        ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"])
    rcrc_prim.CreateAttribute("simLaparoscopeRightClampRotationCenter", Sdf.ValueTypeNames.Bool).Set(True)

    # rightClamp
    rc = UsdGeom.Capsule.Define(stage, "/World/Instrument/laparoscope/rotationCenter/rod/rightClampRotationCenter/rightClamp")
    rclp = rc.GetPrim()
    rclp.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, -0.05, 0.05))
    rclp.CreateAttribute("xformOp:rotateZYX", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    rclp.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
    rclp.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateZYX", "xformOp:scale"])
    rclp.CreateAttribute("height", Sdf.ValueTypeNames.Float).Set(clamp_height)
    rclp.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(clamp_radius)
    rclp.CreateAttribute("simLaparoscopeRightClamp", Sdf.ValueTypeNames.Bool).Set(True)

    print("  Laparoscope added (capsule-based, no mesh geometry)")


def add_lockbox(stage, tumor_min, tumor_max):
    """Add a lockbox that locks the bottom portion of the tumor.

    The lockbox is a unit cube [-1,1]^3 in local space. We scale and position it
    to cover the bottom ~20% of the tumor, simulating where the tumor
    is attached to surrounding tissue.
    """
    tumor_size = tumor_max - tumor_min
    tumor_center = (tumor_min + tumor_max) / 2.0

    # Lock the bottom 20% of the tumor (Y is up)
    lock_fraction = 0.20
    lock_height = tumor_size[1] * lock_fraction
    lock_center_y = tumor_min[1] + lock_height / 2.0

    # Scale: the cube goes from -1 to +1, so half-extent = 1.
    # We want it to cover the full X/Z extent of the tumor (with margin)
    # and the bottom lock_height in Y.
    margin = 1.2  # 20% extra to ensure coverage
    scale_x = tumor_size[0] * margin / 2.0
    scale_y = lock_height / 2.0
    scale_z = tumor_size[2] * margin / 2.0

    cube = UsdGeom.Cube.Define(stage, "/World/LockBox")
    prim = cube.GetPrim()
    prim.CreateAttribute("simLockBox", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(
        Gf.Vec3d(float(tumor_center[0]), float(lock_center_y), float(tumor_center[2])))
    prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(
        Gf.Vec3d(float(scale_x), float(scale_y), float(scale_z)))
    prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])

    print(f"  LockBox: center=({tumor_center[0]:.4f}, {lock_center_y:.4f}, {tumor_center[2]:.4f})")
    print(f"  LockBox: scale=({scale_x:.5f}, {scale_y:.5f}, {scale_z:.5f})")
    print(f"  Locks bottom {lock_fraction*100:.0f}% of tumor (Y < {tumor_min[1] + lock_height:.4f})")


def main():
    print("=" * 60)
    print("Adding Laparoscope + LockBox to tumorBone.usd")
    print("=" * 60)

    stage = Usd.Stage.Open(USD_PATH)

    # Get tumor bounds (use sim verts for accuracy)
    tumor_prim = stage.GetPrimAtPath("/World/Tumor")
    sim_verts = np.array(tumor_prim.GetAttribute("extMesh:vertex").Get())
    tumor_min = sim_verts.min(axis=0)
    tumor_max = sim_verts.max(axis=0)
    tumor_size = tumor_max - tumor_min
    tumor_center = (tumor_min + tumor_max) / 2.0

    print(f"\nTumor bounds:")
    print(f"  min: [{tumor_min[0]:.4f}, {tumor_min[1]:.4f}, {tumor_min[2]:.4f}]")
    print(f"  max: [{tumor_max[0]:.4f}, {tumor_max[1]:.4f}, {tumor_max[2]:.4f}]")
    print(f"  size: [{tumor_size[0]*1000:.1f}, {tumor_size[1]*1000:.1f}, {tumor_size[2]*1000:.1f}] mm")

    print("\nAdding laparoscope...")
    add_laparoscope(stage, tumor_min, tumor_max)

    print("\nAdding lockbox...")
    add_lockbox(stage, tumor_min, tumor_max)

    stage.GetRootLayer().Save()
    print(f"\nSaved: {USD_PATH}")

    # Validate
    print("\n--- Validation ---")
    stage2 = Usd.Stage.Open(USD_PATH)
    for prim in stage2.Traverse():
        for attr_name in ["simMesh", "simRigid", "simLaparoscope", "simLockBox"]:
            if prim.GetAttribute(attr_name).Get() == True:
                print(f"  {prim.GetPath()} → {attr_name}=True")
                break


if __name__ == "__main__":
    main()
