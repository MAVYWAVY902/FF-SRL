#!/usr/bin/env python3
"""
OBJ + GMSH .msh → FF-SRL USD Scene Converter

Converts tumor (deformable) and tbone (rigid) meshes from xpbd-tissue-sim
into a USD scene compatible with FF-SRL's SimModelDO pipeline.

Usage:
    python scripts/obj_to_usd.py \
        --tumor-obj  ../xpbd-tissue-sim/resource/tumor_fixed/tumor_squeezed_uv03.obj \
        --tumor-msh  ../xpbd-tissue-sim/resource/tumor_fixed/tumor_squeezed_uv03.msh \
        --tbone-obj  ../xpbd-tissue-sim/resource/tbone_fixed/tbone_MOD4.obj \
        --output     FF_SRL/FF_SRL/scenes/tumorBone.usd

What this creates:
    /World
      /Tumor    (UsdGeom.Mesh, simMesh=True)   - deformable tissue
      /TBone    (UsdGeom.Mesh, simRigid=True)  - rigid bone
"""

import argparse
import numpy as np
import trimesh
import meshio
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom, Sdf, Vt, Gf


# =============================================================================
# Mesh processing utilities
# =============================================================================

def load_surface_obj(path):
    """Load OBJ surface mesh via trimesh."""
    mesh = trimesh.load(path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    normals = np.array(mesh.vertex_normals, dtype=np.float64)
    uv = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv = np.array(mesh.visual.uv, dtype=np.float64)
    return verts, faces, normals, uv


def load_tet_msh(path):
    """Load GMSH .msh file with tetrahedra via meshio."""
    m = meshio.read(path)
    verts = np.array(m.points, dtype=np.float64)

    tets = None
    tris = None
    for cell_block in m.cells:
        if cell_block.type == 'tetra':
            tets = np.array(cell_block.data, dtype=np.int32)
        elif cell_block.type == 'triangle':
            tris = np.array(cell_block.data, dtype=np.int32)

    if tets is None:
        raise ValueError(f"No tetrahedra found in {path}")
    return verts, tets, tris


def extract_edges_from_tets(tets):
    """Extract unique edges from tetrahedra. Returns (N, 2) array."""
    edge_set = set()
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                e = (min(tet[i], tet[j]), max(tet[i], tet[j]))
                edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int32)
    return edges


def extract_surface_tris_from_tets(tets):
    """Extract surface triangles from tetrahedra (faces appearing exactly once)."""
    face_count = {}
    for tet in tets:
        # 4 faces per tet, each face is 3 vertices
        faces_of_tet = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for f in faces_of_tet:
            face_count[f] = face_count.get(f, 0) + 1

    surface_tris = np.array([list(f) for f, c in face_count.items() if c == 1], dtype=np.int32)
    return surface_tris


def compute_edge_rest_lengths(verts, edges):
    """Compute rest length for each edge."""
    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    return np.linalg.norm(v1 - v0, axis=1).astype(np.float64)


def compute_tet_rest_volumes(verts, tets):
    """Compute signed rest volume for each tetrahedron (1/6 * det)."""
    volumes = np.zeros(len(tets), dtype=np.float64)
    for i, tet in enumerate(tets):
        v0, v1, v2, v3 = verts[tet[0]], verts[tet[1]], verts[tet[2]], verts[tet[3]]
        mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        volumes[i] = abs(np.linalg.det(mat)) / 6.0
    return volumes


def compute_inverse_rest_positions(verts, tets):
    """Compute Q = X^(-1) where X = [v0-v3, v1-v3, v2-v3] for each tet."""
    n_tets = len(tets)
    inv_rest = np.zeros((n_tets, 3, 3), dtype=np.float64)
    for i, tet in enumerate(tets):
        v0, v1, v2, v3 = verts[tet[0]], verts[tet[1]], verts[tet[2]], verts[tet[3]]
        X = np.column_stack([v0 - v3, v1 - v3, v2 - v3])
        try:
            inv_rest[i] = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            inv_rest[i] = np.eye(3)  # degenerate tet fallback
    return inv_rest


def compute_inverse_masses(verts, tets, density=1000.0):
    """Compute inverse mass per vertex.
    FF-SRL's PBD implementation uses uniform inverseMass=1.0 for all vertices
    (matching the liver scene). The lockbox sets locked vertices to 0.0."""
    n_verts = len(verts)
    # FF-SRL uses uniform inverse mass = 1.0 (not physically-based).
    # The gravity kernel multiplies velocity by invMass, so invMass=1.0
    # means gravity acts as a direct acceleration scaling factor.
    inv_mass = np.ones(n_verts, dtype=np.float64)
    return inv_mass


def compute_point_to_vertex_mapping(vis_points, sim_verts):
    """Map visualization surface points to nearest simulation vertices."""
    tree = cKDTree(sim_verts)
    _, indices = tree.query(vis_points)
    return indices.astype(np.int32)


def compute_vertex_adjacency(n_verts, elements, elements_per):
    """Compute variable-length vertex → element adjacency lists.
    Returns (flat_list, lengths) matching FF-SRL's mapping format."""
    adjacency = [[] for _ in range(n_verts)]
    n_elements = len(elements) // elements_per

    for i in range(n_elements):
        for j in range(elements_per):
            v = elements[i * elements_per + j]
            adjacency[v].append(i)

    flat_list = []
    lengths = []
    for adj in adjacency:
        flat_list.extend(adj)
        lengths.append(len(adj))

    return np.array(flat_list, dtype=np.int32), np.array(lengths, dtype=np.int32)


def compute_vertex_to_edge(n_verts, edges_flat):
    """Compute vertex → edge adjacency. Edge i connects edges_flat[2*i] and edges_flat[2*i+1]."""
    n_edges = len(edges_flat) // 2
    adjacency = [[] for _ in range(n_verts)]

    for i in range(n_edges):
        v0 = edges_flat[2 * i]
        v1 = edges_flat[2 * i + 1]
        adjacency[v0].append(i)
        adjacency[v1].append(i)

    flat_list = []
    lengths = []
    for adj in adjacency:
        flat_list.extend(adj)
        lengths.append(len(adj))

    return np.array(flat_list, dtype=np.int32), np.array(lengths, dtype=np.int32)


def compute_vertex_to_triangle(n_verts, triangles_flat):
    """Compute vertex → triangle adjacency."""
    return compute_vertex_adjacency(n_verts, triangles_flat, 3)


def compute_vertex_to_tetrahedron(n_verts, tets_flat):
    """Compute vertex → tetrahedron adjacency."""
    return compute_vertex_adjacency(n_verts, tets_flat, 4)


# =============================================================================
# USD authoring
# =============================================================================

def author_deformable_mesh(stage, prim_path, vis_verts, vis_faces, vis_normals,
                           sim_verts, sim_tets, sim_tris, sim_edges,
                           edge_rest_lengths, tet_rest_volumes,
                           inv_rest_positions, inv_mass, inv_mass_nh,
                           point_to_vertex, vt_edge, vt_edge_len,
                           vt_tet, vt_tet_len, vt_tri, vt_tri_len,
                           uv_coords=None,
                           density=1000.0, ks_distance=1.0, ks_volume=1.0,
                           ks_contact=1.0, ks_drag=1.0,
                           volume_compliance=1e-5, deviatoric_compliance=1e-5,
                           break_factor=5.0, break_threshold=0.5):
    """Author a deformable tissue mesh prim with all SimMeshDO attributes."""

    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    prim = mesh_prim.GetPrim()

    # --- Identification ---
    prim.CreateAttribute("simMesh", Sdf.ValueTypeNames.Bool).Set(True)

    # --- Visualization mesh ---
    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vis_verts]))
    mesh_prim.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in vis_normals]))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(vis_faces.ravel().tolist()))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(vis_faces) // 3 if vis_faces.ndim == 1 else len(vis_faces))))

    # Display colors (default gray)
    n_vis = len(vis_verts)
    colors = Vt.Vec3fArray([Gf.Vec3f(0.8, 0.6, 0.6)] * n_vis)
    prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(colors)

    # UV coordinates
    if uv_coords is not None:
        prim.CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray).Set(
            Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uv_coords]))

    # --- Simulation mesh data (extMesh: namespace) ---
    prim.CreateAttribute("extMesh:vertex", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray([Gf.Vec3f(*v) for v in sim_verts]))

    prim.CreateAttribute("extMesh:triangle", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(sim_tris.ravel().tolist()))

    prim.CreateAttribute("extMesh:edge", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(sim_edges.ravel().tolist()))

    prim.CreateAttribute("extMesh:elem", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(sim_tets.ravel().tolist()))

    prim.CreateAttribute("extMesh:inverseMass", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(inv_mass.tolist()))

    prim.CreateAttribute("extMesh:edgeRestLength", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(edge_rest_lengths.tolist()))

    prim.CreateAttribute("extMesh:tetrahedronRestVolume", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(tet_rest_volumes.tolist()))

    # Inverse rest positions (flattened 3x3 matrices)
    inv_rest_flat = inv_rest_positions.reshape(-1, 9)
    mat_list = []
    for row in inv_rest_flat:
        mat_list.append(Gf.Matrix3f(
            row[0], row[1], row[2],
            row[3], row[4], row[5],
            row[6], row[7], row[8]))
    prim.CreateAttribute("extMesh:inverseRestPosition", Sdf.ValueTypeNames.Matrix3dArray).Set(
        [Gf.Matrix3d(*[float(x) for x in row]) for row in inv_rest_flat])

    prim.CreateAttribute("extMesh:inverseMassNeoHookean", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(inv_mass_nh.tolist()))

    # --- Mapping attributes ---
    prim.CreateAttribute("mapping:pointToVertex", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(point_to_vertex.tolist()))

    prim.CreateAttribute("mapping:vertexToEdge", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_edge.tolist()))
    prim.CreateAttribute("mapping:vertexToEdgeLen", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_edge_len.tolist()))

    prim.CreateAttribute("mapping:vertexToTetrahedron", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_tet.tolist()))
    prim.CreateAttribute("mapping:vertexToTetrahedronLen", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_tet_len.tolist()))

    prim.CreateAttribute("mapping:vertexToTriangle", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_tri.tolist()))
    prim.CreateAttribute("mapping:vertexToTriangleLen", Sdf.ValueTypeNames.IntArray).Set(
        Vt.IntArray(vt_tri_len.tolist()))

    # --- Material parameters ---
    prim.CreateAttribute("param:density", Sdf.ValueTypeNames.Float).Set(float(density))
    prim.CreateAttribute("param:ksDistance", Sdf.ValueTypeNames.Float).Set(float(ks_distance))
    prim.CreateAttribute("param:ksVolume", Sdf.ValueTypeNames.Float).Set(float(ks_volume))
    prim.CreateAttribute("param:ksContact", Sdf.ValueTypeNames.Float).Set(float(ks_contact))
    prim.CreateAttribute("param:ksDrag", Sdf.ValueTypeNames.Float).Set(float(ks_drag))
    prim.CreateAttribute("param:volumeCompliance", Sdf.ValueTypeNames.Float).Set(float(volume_compliance))
    prim.CreateAttribute("param:deviatoricCompliance", Sdf.ValueTypeNames.Float).Set(float(deviatoric_compliance))
    prim.CreateAttribute("param:breakFactor", Sdf.ValueTypeNames.Float).Set(float(break_factor))
    prim.CreateAttribute("param:breakThreshold", Sdf.ValueTypeNames.Float).Set(float(break_threshold))
    prim.CreateAttribute("param:heatLoss", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("param:heatConduction", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("param:heatTransfer", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("param:heatLimit", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("param:heatSphereRadius", Sdf.ValueTypeNames.Float).Set(0.001)
    prim.CreateAttribute("param:heatQuantum", Sdf.ValueTypeNames.Float).Set(0.0)

    print(f"  Authored deformable mesh at {prim_path}")
    print(f"    Vis: {n_vis} points, {len(vis_faces.ravel())//3} faces")
    print(f"    Sim: {len(sim_verts)} verts, {len(sim_tets)} tets, {len(sim_edges)} edges, {len(sim_tris)} tris")


def author_rigid_mesh(stage, prim_path, verts, faces, normals, uv_coords=None):
    """Author a rigid (non-deformable) mesh prim."""

    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    prim = mesh_prim.GetPrim()

    prim.CreateAttribute("simRigid", Sdf.ValueTypeNames.Bool).Set(True)

    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh_prim.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.ravel().tolist()))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))

    n_vis = len(verts)
    colors = Vt.Vec3fArray([Gf.Vec3f(0.9, 0.9, 0.85)] * n_vis)
    prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(colors)

    if uv_coords is not None:
        prim.CreateAttribute("primvars:st", Sdf.ValueTypeNames.TexCoord2fArray).Set(
            Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uv_coords]))

    print(f"  Authored rigid mesh at {prim_path}")
    print(f"    {n_vis} points, {len(faces)} faces")


# =============================================================================
# Main conversion pipeline
# =============================================================================

def convert(tumor_obj_path, tumor_msh_path, tbone_obj_path, output_path,
            tumor_density=1000.0, tumor_scale=None, tbone_scale=None):
    """Full conversion pipeline: OBJ + MSH → USD."""

    print("=" * 60)
    print("OBJ → USD Conversion for FF-SRL")
    print("=" * 60)

    # --- Load tumor surface (OBJ) ---
    print("\n[1/6] Loading tumor surface OBJ...")
    tumor_vis_verts, tumor_vis_faces, tumor_vis_normals, tumor_uv = load_surface_obj(tumor_obj_path)
    print(f"  Surface: {len(tumor_vis_verts)} verts, {len(tumor_vis_faces)} faces")

    # --- Load tumor volume (MSH) ---
    print("\n[2/6] Loading tumor tet mesh (GMSH .msh)...")
    tumor_sim_verts, tumor_tets, tumor_msh_tris = load_tet_msh(tumor_msh_path)
    print(f"  Volume: {len(tumor_sim_verts)} verts, {len(tumor_tets)} tets")

    # --- Load tbone surface (OBJ) ---
    print("\n[3/6] Loading tbone surface OBJ...")
    tbone_verts, tbone_faces, tbone_normals, tbone_uv = load_surface_obj(tbone_obj_path)
    print(f"  Surface: {len(tbone_verts)} verts, {len(tbone_faces)} faces")

    # --- Unit conversion: meters → centimeters ---
    # FF-SRL's PBD implementation (gravity, constraint stiffness, etc.) is tuned
    # for cm-scale coordinates (matching the liver scene: ~35 units across).
    # The tumor OBJ/MSH files are in SI meters (~0.03 units). Without scaling,
    # gravity displacement per substep (0.17mm) exceeds 34% of the smallest edge
    # (0.48mm), causing immediate instability.
    UNIT_SCALE = 100.0  # m → cm
    tumor_sim_verts *= UNIT_SCALE
    tumor_vis_verts *= UNIT_SCALE
    tbone_verts *= UNIT_SCALE
    tumor_size_cm = tumor_sim_verts.max(axis=0) - tumor_sim_verts.min(axis=0)
    print(f"\n  Unit conversion: m → cm (×{UNIT_SCALE})")
    print(f"  Tumor size after scaling: [{tumor_size_cm[0]:.2f}, {tumor_size_cm[1]:.2f}, {tumor_size_cm[2]:.2f}] cm")

    # --- Optional additional scaling ---
    if tumor_scale is not None:
        cur_size = tumor_sim_verts.max(axis=0) - tumor_sim_verts.min(axis=0)
        scale_factor = tumor_scale / cur_size.max()
        tumor_sim_verts *= scale_factor
        tumor_vis_verts *= scale_factor
        print(f"  Tumor additionally scaled by {scale_factor:.4f} to max-size {tumor_scale}")

    if tbone_scale is not None:
        cur_size = tbone_verts.max(axis=0) - tbone_verts.min(axis=0)
        scale_factor = tbone_scale / cur_size.max()
        tbone_verts *= scale_factor
        print(f"  TBone additionally scaled by {scale_factor:.4f} to max-size {tbone_scale}")

    # --- Compute derived data for tumor ---
    print("\n[4/6] Computing simulation data...")

    # Extract edges from tetrahedra
    sim_edges = extract_edges_from_tets(tumor_tets)
    print(f"  Edges: {len(sim_edges)}")

    # Surface triangles from tets (or use MSH triangles)
    if tumor_msh_tris is not None and len(tumor_msh_tris) > 0:
        sim_tris = tumor_msh_tris
    else:
        sim_tris = extract_surface_tris_from_tets(tumor_tets)
    print(f"  Surface tris: {len(sim_tris)}")

    # Rest lengths, volumes, inverse rest positions
    edge_rest_lengths = compute_edge_rest_lengths(tumor_sim_verts, sim_edges)
    tet_rest_volumes = compute_tet_rest_volumes(tumor_sim_verts, tumor_tets)
    inv_rest_positions = compute_inverse_rest_positions(tumor_sim_verts, tumor_tets)

    # Inverse masses
    inv_mass = compute_inverse_masses(tumor_sim_verts, tumor_tets, density=tumor_density)
    inv_mass_nh = inv_mass.copy()  # Same for Neo-Hookean

    print(f"  Tet volumes: min={tet_rest_volumes.min():.2e}, max={tet_rest_volumes.max():.2e}")
    print(f"  Edge lengths: min={edge_rest_lengths.min()*1000:.2f}mm, max={edge_rest_lengths.max()*1000:.2f}mm")
    print(f"  Inv mass: min={inv_mass[inv_mass>0].min():.2f}, max={inv_mass.max():.2f}")

    # --- Compute mappings ---
    print("\n[5/6] Computing mappings...")

    # Point to vertex (vis surface → sim volume)
    point_to_vertex = compute_point_to_vertex_mapping(tumor_vis_verts, tumor_sim_verts)
    print(f"  Point-to-vertex mapping: {len(point_to_vertex)} entries")

    # Vertex adjacency lists
    n_sim_verts = len(tumor_sim_verts)
    edges_flat = sim_edges.ravel()
    tris_flat = sim_tris.ravel()
    tets_flat = tumor_tets.ravel()

    vt_edge, vt_edge_len = compute_vertex_to_edge(n_sim_verts, edges_flat)
    vt_tri, vt_tri_len = compute_vertex_to_triangle(n_sim_verts, tris_flat)
    vt_tet, vt_tet_len = compute_vertex_to_tetrahedron(n_sim_verts, tets_flat)

    print(f"  Vertex-to-edge: {len(vt_edge)} entries")
    print(f"  Vertex-to-tri:  {len(vt_tri)} entries")
    print(f"  Vertex-to-tet:  {len(vt_tet)} entries")

    # --- Author USD ---
    print(f"\n[6/6] Writing USD to {output_path}...")

    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # World root
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    # Tumor (deformable)
    author_deformable_mesh(
        stage, "/World/Tumor",
        vis_verts=tumor_vis_verts,
        vis_faces=tumor_vis_faces if tumor_vis_faces.ndim == 1 else tumor_vis_faces.ravel(),
        vis_normals=tumor_vis_normals,
        sim_verts=tumor_sim_verts,
        sim_tets=tumor_tets,
        sim_tris=sim_tris,
        sim_edges=sim_edges,
        edge_rest_lengths=edge_rest_lengths,
        tet_rest_volumes=tet_rest_volumes,
        inv_rest_positions=inv_rest_positions,
        inv_mass=inv_mass,
        inv_mass_nh=inv_mass_nh,
        point_to_vertex=point_to_vertex,
        vt_edge=vt_edge, vt_edge_len=vt_edge_len,
        vt_tet=vt_tet, vt_tet_len=vt_tet_len,
        vt_tri=vt_tri, vt_tri_len=vt_tri_len,
        uv_coords=tumor_uv,
        density=tumor_density,
    )

    # TBone (rigid)
    author_rigid_mesh(
        stage, "/World/TBone",
        verts=tbone_verts,
        faces=tbone_faces,
        normals=tbone_normals,
        uv_coords=tbone_uv,
    )

    stage.GetRootLayer().Save()

    print(f"\nDone! USD saved to: {output_path}")
    print(f"  Tumor: {len(tumor_sim_verts)} sim verts, {len(tumor_tets)} tets")
    print(f"  TBone: {len(tbone_verts)} verts (rigid)")

    # Quick validation
    print("\n--- Validation ---")
    stage2 = Usd.Stage.Open(output_path)
    meshes = [x for x in stage2.Traverse() if x.IsA(UsdGeom.Mesh)]
    for m in meshes:
        path = m.GetPath()
        is_sim = m.GetAttribute("simMesh").Get()
        is_rigid = m.GetAttribute("simRigid").Get()
        pts = m.GetAttribute("points").Get()
        print(f"  {path}: {len(pts)} points, simMesh={is_sim}, simRigid={is_rigid}")
        if is_sim:
            sv = m.GetAttribute("extMesh:vertex").Get()
            se = m.GetAttribute("extMesh:elem").Get()
            print(f"    extMesh: {len(sv)} verts, {len(se)//4} tets")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OBJ+MSH meshes to FF-SRL USD scene")
    parser.add_argument("--tumor-obj", required=True, help="Path to tumor surface OBJ")
    parser.add_argument("--tumor-msh", required=True, help="Path to tumor tet mesh (GMSH .msh)")
    parser.add_argument("--tbone-obj", required=True, help="Path to tbone surface OBJ")
    parser.add_argument("--output", required=True, help="Output USD path")
    parser.add_argument("--tumor-density", type=float, default=1000.0, help="Tumor density (kg/m^3)")
    parser.add_argument("--tumor-scale", type=float, default=None, help="Scale tumor to max-size (meters)")
    parser.add_argument("--tbone-scale", type=float, default=None, help="Scale tbone to max-size (meters)")

    args = parser.parse_args()

    convert(
        tumor_obj_path=args.tumor_obj,
        tumor_msh_path=args.tumor_msh,
        tbone_obj_path=args.tbone_obj,
        output_path=args.output,
        tumor_density=args.tumor_density,
        tumor_scale=args.tumor_scale,
        tbone_scale=args.tbone_scale,
    )
