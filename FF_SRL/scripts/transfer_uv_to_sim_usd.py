"""
Transfer UV coordinates and texture material bindings from a Blender-exported USD
back into the original simulation USD (preserving all sim attributes).

Usage:
    python scripts/transfer_uv_to_sim_usd.py \
        --source scenes/tumorBone_textured.usd \
        --target scenes/tumorBone.usd \
        --texture_dir assets/textures \
        --output scenes/tumorBone_final.usd

This copies:
  - primvars:st (UV coordinates) from source meshes to target meshes
  - Material + Shader prims with texture file references
  - MaterialBindingAPI relationships
"""

import argparse
import shutil
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf


def find_meshes(stage):
    """Return dict of mesh_name -> UsdGeom.Mesh."""
    meshes = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes[prim.GetName()] = UsdGeom.Mesh(prim)
    return meshes


def transfer_uvs(source_stage, target_stage):
    """Copy UV primvars from source meshes to matching target meshes."""
    src_meshes = find_meshes(source_stage)
    tgt_meshes = find_meshes(target_stage)

    transferred = []
    for name, src_mesh in src_meshes.items():
        if name not in tgt_meshes:
            print(f"  SKIP: '{name}' not found in target USD")
            continue

        tgt_mesh = tgt_meshes[name]

        # Get UV primvar from source
        src_pv_api = UsdGeom.PrimvarsAPI(src_mesh.GetPrim())
        st_pv = src_pv_api.GetPrimvar("st")
        if not st_pv or not st_pv.HasValue():
            # Try "UVMap" (Blender default name)
            st_pv = src_pv_api.GetPrimvar("UVMap")
        if not st_pv or not st_pv.HasValue():
            print(f"  SKIP: '{name}' has no UV primvar in source")
            continue

        uv_values = st_pv.Get()
        uv_indices = st_pv.GetIndices()
        interpolation = st_pv.GetInterpolation()

        # Create or overwrite UV primvar on target
        tgt_pv_api = UsdGeom.PrimvarsAPI(tgt_mesh.GetPrim())
        tgt_st = tgt_pv_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
        tgt_st.Set(uv_values)
        if uv_indices is not None and len(uv_indices) > 0:
            tgt_st.SetIndices(uv_indices)

        transferred.append(name)
        print(f"  OK: '{name}' — {len(uv_values)} UV coords transferred")

    return transferred


def transfer_materials(source_stage, target_stage, texture_dir=None):
    """Copy material prims and bindings from source to target."""
    src_meshes = find_meshes(source_stage)
    tgt_meshes = find_meshes(target_stage)

    # Ensure /World/Looks scope exists in target
    looks_path = Sdf.Path("/World/Looks")
    if not target_stage.GetPrimAtPath(looks_path):
        UsdGeom.Scope.Define(target_stage, looks_path)

    for name, src_mesh in src_meshes.items():
        if name not in tgt_meshes:
            continue

        # Check if source mesh has material binding
        src_binding = UsdShade.MaterialBindingAPI(src_mesh.GetPrim())
        src_mat_path = src_binding.GetDirectBinding().GetMaterialPath()
        if not src_mat_path or src_mat_path.isEmpty:
            continue

        src_mat_prim = source_stage.GetPrimAtPath(src_mat_path)
        if not src_mat_prim:
            continue

        # Copy material prim tree to target under /World/Looks
        mat_name = src_mat_prim.GetName()
        tgt_mat_path = looks_path.AppendChild(mat_name)

        # If material already exists in target, skip creation
        if not target_stage.GetPrimAtPath(tgt_mat_path):
            # Use Sdf.CopySpec to deep-copy the material subtree
            src_layer = source_stage.GetRootLayer()
            tgt_layer = target_stage.GetRootLayer()
            Sdf.CopySpec(src_layer, src_mat_path, tgt_layer, tgt_mat_path)
            print(f"  Material '{mat_name}' copied to {tgt_mat_path}")

            # Fix texture file paths if needed
            if texture_dir:
                _fix_texture_paths(target_stage, tgt_mat_path, texture_dir)

        # Bind material to target mesh
        tgt_binding = UsdShade.MaterialBindingAPI.Apply(tgt_meshes[name].GetPrim())
        tgt_mat = UsdShade.Material(target_stage.GetPrimAtPath(tgt_mat_path))
        tgt_binding.Bind(tgt_mat)
        print(f"  Bound '{mat_name}' → '{name}'")


def _fix_texture_paths(stage, mat_path, texture_dir):
    """Update texture file paths in shader nodes to point to texture_dir."""
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(mat_path):
            continue
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            file_input = shader.GetInput("file")
            if file_input and file_input.Get():
                asset_path = file_input.Get()
                # Extract filename and repoint to texture_dir
                import os
                filename = os.path.basename(str(asset_path).replace("@", ""))
                new_path = Sdf.AssetPath(os.path.join(texture_dir, filename))
                file_input.Set(new_path)
                print(f"    Texture path updated: {filename} → {new_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer UV + materials from Blender USD to sim USD")
    parser.add_argument("--source", required=True,
                        help="Blender-exported USD with UVs and materials")
    parser.add_argument("--target", required=True,
                        help="Original simulation USD (will NOT be modified)")
    parser.add_argument("--output", required=True,
                        help="Output USD path (copy of target + UVs/materials)")
    parser.add_argument("--texture_dir", default=None,
                        help="Directory containing texture PNGs (to fix paths)")
    args = parser.parse_args()

    print(f"Source (Blender): {args.source}")
    print(f"Target (sim):     {args.target}")
    print(f"Output:           {args.output}")

    # Copy target to output (preserve everything)
    shutil.copy2(args.target, args.output)

    source_stage = Usd.Stage.Open(args.source)
    target_stage = Usd.Stage.Open(args.output)

    print("\n--- Transferring UVs ---")
    transferred = transfer_uvs(source_stage, target_stage)

    print("\n--- Transferring Materials ---")
    transfer_materials(source_stage, target_stage, args.texture_dir)

    target_stage.GetRootLayer().Save()
    print(f"\nDone! Output saved to: {args.output}")
    print(f"Meshes with UVs: {transferred}")


if __name__ == "__main__":
    main()
