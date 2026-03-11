#!/usr/bin/env python3
"""
Script to examine the structure, dimensions, and complexity of tissue in USD file
"""

from pxr import Usd, UsdGeom
import numpy as np

def examine_tissue_structure(usd_path):
    """Examine tissue mesh structure in USD file"""
    
    print(f"\n{'='*80}")
    print(f"EXAMINING USD FILE: {usd_path}")
    print(f"{'='*80}\n")
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    
    # Find all mesh objects marked as simMesh (tissue)
    meshes = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simMesh").Get() == True]
    
    print(f"Found {len(meshes)} tissue mesh(es)\n")
    
    for idx, mesh_prim in enumerate(meshes):
        print(f"\n{'='*80}")
        print(f"TISSUE MESH {idx + 1}: {mesh_prim.GetPath()}")
        print(f"{'='*80}\n")
        
        # Get visualization mesh data (surface)
        vis_points = mesh_prim.GetAttribute("points").Get()
        vis_faces = mesh_prim.GetAttribute("faceVertexIndices").Get()
        normals = mesh_prim.GetAttribute("normals").Get()
        
        # Get simulation mesh data (tetrahedral)
        sim_vertices = mesh_prim.GetAttribute("extMesh:vertex").Get()
        sim_edges = mesh_prim.GetAttribute("extMesh:edge").Get()
        sim_triangles = mesh_prim.GetAttribute("extMesh:triangle").Get()
        sim_tetrahedrons = mesh_prim.GetAttribute("extMesh:elem").Get()
        
        # Get material properties
        inverse_masses = mesh_prim.GetAttribute("extMesh:inverseMass").Get()
        edge_rest_lengths = mesh_prim.GetAttribute("extMesh:edgeRestLength").Get()
        tet_rest_volumes = mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get()
        inverse_rest_positions = mesh_prim.GetAttribute("extMesh:inverseRestPosition").Get()
        inverse_mass_nh = mesh_prim.GetAttribute("extMesh:inverseMassNeoHookean").Get()
        
        # Get mesh properties
        density = mesh_prim.GetAttribute("density").Get()
        ks_distance = mesh_prim.GetAttribute("ksDistance").Get()
        ks_volume = mesh_prim.GetAttribute("ksVolume").Get()
        volume_compliance = mesh_prim.GetAttribute("volumeCompliance").Get()
        deviatoric_compliance = mesh_prim.GetAttribute("deviatoricCompliance").Get()
        
        # VISUALIZATION MESH STATISTICS
        print("VISUALIZATION MESH (Surface Rendering):")
        print("-" * 80)
        num_vis_points = len(vis_points) if vis_points else 0
        num_vis_faces = len(vis_faces) // 3 if vis_faces else 0
        print(f"  • Surface Points (vertices): {num_vis_points:,}")
        print(f"  • Surface Triangles (faces): {num_vis_faces:,}")
        
        if vis_points:
            vis_points_np = np.array(vis_points)
            vis_min = vis_points_np.min(axis=0)
            vis_max = vis_points_np.max(axis=0)
            vis_extent = vis_max - vis_min
            vis_center = (vis_min + vis_max) / 2
            print(f"  • Bounding Box Min: ({vis_min[0]:.2f}, {vis_min[1]:.2f}, {vis_min[2]:.2f})")
            print(f"  • Bounding Box Max: ({vis_max[0]:.2f}, {vis_max[1]:.2f}, {vis_max[2]:.2f})")
            print(f"  • Dimensions (W×H×D): {vis_extent[0]:.2f} × {vis_extent[1]:.2f} × {vis_extent[2]:.2f}")
            print(f"  • Center: ({vis_center[0]:.2f}, {vis_center[1]:.2f}, {vis_center[2]:.2f})")
        
        # SIMULATION MESH STATISTICS
        print("\n\nSIMULATION MESH (Tetrahedral Physics):")
        print("-" * 80)
        num_vertices = len(sim_vertices) if sim_vertices else 0
        num_edges = len(sim_edges) // 2 if sim_edges else 0
        num_tris = len(sim_triangles) // 3 if sim_triangles else 0
        num_tets = len(sim_tetrahedrons) // 4 if sim_tetrahedrons else 0
        
        print(f"  • Simulation Vertices: {num_vertices:,}")
        print(f"  • Edges: {num_edges:,}")
        print(f"  • Surface Triangles: {num_tris:,}")
        print(f"  • Tetrahedral Elements: {num_tets:,}")
        
        if sim_vertices:
            sim_verts_np = np.array(sim_vertices)
            sim_min = sim_verts_np.min(axis=0)
            sim_max = sim_verts_np.max(axis=0)
            sim_extent = sim_max - sim_min
            sim_center = (sim_min + sim_max) / 2
            print(f"  • Simulation Bounding Box Min: ({sim_min[0]:.2f}, {sim_min[1]:.2f}, {sim_min[2]:.2f})")
            print(f"  • Simulation Bounding Box Max: ({sim_max[0]:.2f}, {sim_max[1]:.2f}, {sim_max[2]:.2f})")
            print(f"  • Simulation Dimensions: {sim_extent[0]:.2f} × {sim_extent[1]:.2f} × {sim_extent[2]:.2f}")
        
        # COMPLEXITY METRICS
        print("\n\nCOMPLEXITY METRICS:")
        print("-" * 80)
        if num_vertices > 0:
            avg_edges_per_vertex = (num_edges * 2) / num_vertices
            avg_tets_per_vertex = (num_tets * 4) / num_vertices
            print(f"  • Average Edges per Vertex: {avg_edges_per_vertex:.1f}")
            print(f"  • Average Tets per Vertex: {avg_tets_per_vertex:.1f}")
        
        if num_tets > 0 and tet_rest_volumes:
            tet_vols = np.array(tet_rest_volumes)
            total_volume = tet_vols.sum()
            avg_tet_vol = tet_vols.mean()
            min_tet_vol = tet_vols.min()
            max_tet_vol = tet_vols.max()
            print(f"  • Total Mesh Volume: {total_volume:.2f} cubic units")
            print(f"  • Average Tet Volume: {avg_tet_vol:.6f}")
            print(f"  • Min Tet Volume: {min_tet_vol:.6f}")
            print(f"  • Max Tet Volume: {max_tet_vol:.6f}")
            print(f"  • Tet Volume Ratio (max/min): {max_tet_vol/min_tet_vol:.2f}")
        
        # MATERIAL PROPERTIES
        print("\n\nMATERIAL PROPERTIES:")
        print("-" * 80)
        print(f"  • Density: {density if density else 'N/A'}")
        print(f"  • Distance Constraint Stiffness (ksDistance): {ks_distance if ks_distance else 'N/A'}")
        print(f"  • Volume Constraint Stiffness (ksVolume): {ks_volume if ks_volume else 'N/A'}")
        print(f"  • Volume Compliance: {volume_compliance if volume_compliance else 'N/A'}")
        print(f"  • Deviatoric Compliance: {deviatoric_compliance if deviatoric_compliance else 'N/A'}")
        
        if inverse_masses:
            masses = 1.0 / np.array(inverse_masses)
            masses = masses[masses < 1e10]  # Filter out fixed vertices
            if len(masses) > 0:
                total_mass = masses.sum()
                avg_mass = masses.mean()
                print(f"  • Total Mass: {total_mass:.4f}")
                print(f"  • Average Vertex Mass: {avg_mass:.6f}")
        
        # MEMORY FOOTPRINT
        print("\n\nMEMORY FOOTPRINT ESTIMATE:")
        print("-" * 80)
        # Vertex data: position (3 floats), velocity (3 floats), etc.
        vertex_mem = num_vertices * 3 * 4 * 4  # ~4 vec3 per vertex
        edge_mem = num_edges * 2 * 4  # 2 ints per edge
        tet_mem = num_tets * 4 * 4  # 4 ints per tet
        tet_data_mem = num_tets * (1 + 1 + 9) * 4  # volume, mass, inverse rest (3x3 matrix)
        total_mem = vertex_mem + edge_mem + tet_mem + tet_data_mem
        
        print(f"  • Vertex Data: ~{vertex_mem / 1024:.1f} KB")
        print(f"  • Edge Data: ~{edge_mem / 1024:.1f} KB")
        print(f"  • Tetrahedral Data: ~{(tet_mem + tet_data_mem) / 1024:.1f} KB")
        print(f"  • TOTAL ESTIMATE: ~{total_mem / 1024:.1f} KB (~{total_mem / (1024*1024):.2f} MB)")
        
        # MAPPING INFORMATION
        print("\n\nMESH MAPPING:")
        print("-" * 80)
        point_to_vertex = mesh_prim.GetAttribute("mapping:pointToVertex").Get()
        vertex_to_edge = mesh_prim.GetAttribute("mapping:vertexToEdge").Get()
        vertex_to_tet = mesh_prim.GetAttribute("mapping:vertexToTetrahedron").Get()
        
        if point_to_vertex:
            print(f"  • Visualization-to-Simulation mapping: {len(point_to_vertex):,} entries")
            print(f"  • Mapping type: Surface points → Simulation vertices")
        
        if vertex_to_edge:
            print(f"  • Vertex-to-Edge mapping: {len(vertex_to_edge):,} entries")
        
        if vertex_to_tet:
            print(f"  • Vertex-to-Tetrahedron mapping: {len(vertex_to_tet):,} entries")
        
        # COMPUTATIONAL COMPLEXITY
        print("\n\nCOMPUTATIONAL COMPLEXITY (Per Time Step):")
        print("-" * 80)
        print(f"  • Distance Constraints: O({num_edges:,}) edge evaluations")
        print(f"  • Volume Constraints: O({num_tets:,}) tetrahedral evaluations")
        print(f"  • Surface Collision: O({num_tris:,}) triangle checks")
        print(f"  • Rendering: O({num_vis_faces:,}) surface triangles")
        
        # Check for texture mapping
        tex_coords = mesh_prim.GetAttribute("primvars:st").Get()
        if tex_coords:
            print(f"\n\nTEXTURE MAPPING:")
            print("-" * 80)
            print(f"  • Texture Coordinates: {len(tex_coords):,} UV pairs")
            print(f"  • Has texture mapping: Yes")


if __name__ == "__main__":
    # Examine the liver retraction scene
    usd_path = "/home/yunxin/FF-SRL/FF_SRL/FF_SRL/scenes/liverRetractionTexture.usd"
    examine_tissue_structure(usd_path)
    
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The tissue mesh uses a dual representation:
1. VISUALIZATION MESH: High-resolution surface for rendering
2. SIMULATION MESH: Tetrahedral volumetric mesh for physics

This allows efficient rendering while maintaining accurate physics simulation.
The tetrahedral mesh captures the 3D volumetric behavior of soft tissue,
enabling realistic deformation under Neo-Hookean material model.
""")
