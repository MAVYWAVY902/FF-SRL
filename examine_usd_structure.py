#!/usr/bin/env python3
"""
Deep dive into USD tissue data structure
Shows exactly what data is stored and how it's organized
"""

from pxr import Usd, UsdGeom, UsdShade
import numpy as np

def examine_data_structure(usd_path):
    """Examine the detailed data structure of tissue in USD"""
    
    print("\n" + "="*100)
    print("DETAILED USD DATA STRUCTURE EXAMINATION")
    print("="*100 + "\n")
    
    stage = Usd.Stage.Open(usd_path)
    meshes = [x for x in stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simMesh").Get() == True]
    
    # Examine just the first mesh (Liver) in detail
    mesh_prim = meshes[0]
    
    print(f"Examining: {mesh_prim.GetPath()}")
    print(f"Prim Type: {mesh_prim.GetTypeName()}\n")
    
    print("="*100)
    print("ALL ATTRIBUTES IN USD FILE")
    print("="*100)
    
    attributes = mesh_prim.GetAttributes()
    
    # Group attributes by category
    viz_attrs = []
    sim_attrs = []
    mapping_attrs = []
    material_attrs = []
    other_attrs = []
    
    for attr in attributes:
        name = attr.GetName()
        if name.startswith('extMesh:'):
            sim_attrs.append(attr)
        elif name.startswith('mapping:'):
            mapping_attrs.append(attr)
        elif name in ['points', 'faceVertexIndices', 'normals', 'primvars:st', 'primvars:displayColor']:
            viz_attrs.append(attr)
        elif name in ['density', 'ksDistance', 'ksVolume', 'volumeCompliance', 'deviatoricCompliance',
                     'breakFactor', 'breakThreshold', 'heatLoss', 'heatConduction']:
            material_attrs.append(attr)
        else:
            other_attrs.append(attr)
    
    # Print categorized
    print("\n1. VISUALIZATION MESH ATTRIBUTES (for rendering)")
    print("-" * 100)
    for attr in viz_attrs:
        val = attr.Get()
        if val is not None:
            if hasattr(val, '__len__'):
                print(f"  {attr.GetName():<30} = Array[{len(val)}] of {type(val[0]).__name__ if len(val) > 0 else 'unknown'}")
                if len(val) <= 5:
                    print(f"      Values: {val}")
            else:
                print(f"  {attr.GetName():<30} = {val}")
    
    print("\n2. SIMULATION MESH ATTRIBUTES (for physics)")
    print("-" * 100)
    for attr in sim_attrs:
        val = attr.Get()
        if val is not None:
            dtype = type(val[0]).__name__ if hasattr(val, '__len__') and len(val) > 0 else 'unknown'
            if hasattr(val, '__len__'):
                print(f"  {attr.GetName():<40} = Array[{len(val):>6}] of {dtype}")
                
                # Show structure for key arrays
                if 'elem' in attr.GetName() or 'edge' in attr.GetName() or 'triangle' in attr.GetName():
                    arr = np.array(val)
                    if 'elem' in attr.GetName():
                        print(f"      → Tetrahedral indices: {len(arr)//4} tets × 4 vertices")
                        print(f"      → Example tet: {arr[:4]}")
                    elif 'edge' in attr.GetName() and 'Length' not in attr.GetName():
                        print(f"      → Edge indices: {len(arr)//2} edges × 2 vertices")
                        print(f"      → Example edge: {arr[:2]}")
                    elif 'triangle' in attr.GetName():
                        print(f"      → Triangle indices: {len(arr)//3} triangles × 3 vertices")
                        print(f"      → Example triangle: {arr[:3]}")
                elif 'inverseRestPosition' in attr.GetName():
                    print(f"      → 3×3 inverse deformation gradient matrices")
                    arr = np.array(val).reshape(-1, 3, 3)
                    print(f"      → Example matrix:\\n{arr[0]}")
            else:
                print(f"  {attr.GetName():<40} = {val}")
    
    print("\n3. MAPPING ATTRIBUTES (visualization ↔ simulation)")
    print("-" * 100)
    for attr in mapping_attrs:
        val = attr.Get()
        if val is not None:
            if hasattr(val, '__len__'):
                print(f"  {attr.GetName():<40} = Array[{len(val)}]")
                arr = np.array(val)
                
                if 'pointToVertex' in attr.GetName():
                    print(f"      → Maps {len(arr)} surface points to simulation vertices")
                    print(f"      → Example mapping: point[0] → vertex[{arr[0]}]")
                elif 'vertexToEdge' in attr.GetName() and 'Len' not in attr.GetName():
                    print(f"      → Variable-length edge lists per vertex")
                elif 'Len' in attr.GetName():
                    print(f"      → Length array for variable-length mapping")
                    print(f"      → Example: vertex 0 has {arr[0]} connected elements")
    
    print("\n4. MATERIAL PROPERTIES")
    print("-" * 100)
    for attr in material_attrs:
        val = attr.Get()
        print(f"  {attr.GetName():<40} = {val if val is not None else 'None (set at runtime)'}")
    
    print("\n5. OTHER ATTRIBUTES")
    print("-" * 100)
    for attr in other_attrs[:10]:  # Limit output
        val = attr.Get()
        if val is not None:
            if hasattr(val, '__len__') and len(val) > 10:
                print(f"  {attr.GetName():<40} = Array[{len(val)}]")
            else:
                val_str = str(val)[:80]
                print(f"  {attr.GetName():<40} = {val_str}")
    
    # Detailed analysis of key structures
    print("\n\n" + "="*100)
    print("DETAILED STRUCTURE ANALYSIS")
    print("="*100)
    
    # Tetrahedra structure
    print("\n▶ TETRAHEDRAL MESH STRUCTURE")
    print("-" * 100)
    tets = np.array(mesh_prim.GetAttribute("extMesh:elem").Get())
    verts = np.array(mesh_prim.GetAttribute("extMesh:vertex").Get())
    tet_vols = np.array(mesh_prim.GetAttribute("extMesh:tetrahedronRestVolume").Get())
    inv_rest = np.array(mesh_prim.GetAttribute("extMesh:inverseRestPosition").Get()).reshape(-1, 3, 3)
    inv_mass_nh = np.array(mesh_prim.GetAttribute("extMesh:inverseMassNeoHookean").Get())
    
    print(f"  Total vertices: {len(verts)}")
    print(f"  Total tetrahedra: {len(tets)//4}")
    print(f"\\n  Tetrahedral Element [0]:")
    print(f"    Vertex indices: {tets[:4]}")
    print(f"    Vertex positions:")
    for i, idx in enumerate(tets[:4]):
        print(f"      v{i} [{idx}]: {verts[idx]}")
    print(f"    Rest volume: {tet_vols[0]:.6f}")
    print(f"    Inverse mass (Neo-Hookean): {inv_mass_nh[0]:.6f}")
    print(f"    Inverse rest position matrix Dm^(-1):")
    print(f"{inv_rest[0]}")
    
    # Edge structure
    print("\n▶ EDGE CONSTRAINTS STRUCTURE")
    print("-" * 100)
    edges = np.array(mesh_prim.GetAttribute("extMesh:edge").Get())
    edge_lens = np.array(mesh_prim.GetAttribute("extMesh:edgeRestLength").Get())
    
    print(f"  Total edges: {len(edges)//2}")
    print(f"\\n  Edge [0]:")
    print(f"    Vertex indices: [{edges[0]}, {edges[1]}]")
    v0, v1 = verts[edges[0]], verts[edges[1]]
    print(f"    Vertex 0 position: {v0}")
    print(f"    Vertex 1 position: {v1}")
    print(f"    Rest length: {edge_lens[0]:.6f}")
    computed_len = np.linalg.norm(v1 - v0)
    print(f"    Computed length: {computed_len:.6f}")
    print(f"    Match: {'✅' if abs(edge_lens[0] - computed_len) < 0.001 else '❌'}")
    
    # Visualization mapping
    print("\n▶ VISUALIZATION TO SIMULATION MAPPING")
    print("-" * 100)
    vis_points = np.array(mesh_prim.GetAttribute("points").Get())
    point_to_vertex = np.array(mesh_prim.GetAttribute("mapping:pointToVertex").Get())
    
    print(f"  Visualization points: {len(vis_points)}")
    print(f"  Simulation vertices: {len(verts)}")
    print(f"  Mapping array size: {len(point_to_vertex)}")
    print(f"\\n  Example mapping:")
    print(f"    Vis point [0]: {vis_points[0]}")
    print(f"    Maps to sim vertex [{point_to_vertex[0]}]: {verts[point_to_vertex[0]]}")
    print(f"    Match: {'✅' if np.allclose(vis_points[0], verts[point_to_vertex[0]]) else '❌ (interpolated)'}")
    
    # Memory layout
    print("\n\n" + "="*100)
    print("MEMORY LAYOUT FOR GPU")
    print("="*100)
    
    print("\\nData arrays that get transferred to GPU:")
    print("-" * 100)
    
    arrays_info = [
        ("vertex", verts, "vec3", "Position of each simulation vertex"),
        ("edge", edges, "int2", "Vertex indices for distance constraints"),
        ("edgeRestLength", edge_lens, "float", "Rest length for each edge"),
        ("tetrahedron", tets, "int4", "Vertex indices for volume elements"),
        ("tetrahedronRestVolume", tet_vols, "float", "Rest volume for each tet"),
        ("tetrahedronInverseMassNeoHookean", inv_mass_nh, "float", "Constraint mass scaling"),
        ("tetrahedronInverseRestPosition", inv_rest, "mat3", "Dm^(-1) deformation gradient reference"),
    ]
    
    total_bytes = 0
    for name, arr, dtype, desc in arrays_info:
        if dtype == "vec3":
            bytes_per = 12
        elif dtype == "int2":
            bytes_per = 8
        elif dtype == "int4":
            bytes_per = 16
        elif dtype == "mat3":
            bytes_per = 36
        else:
            bytes_per = 4
        
        if 'mat3' in dtype:
            count = len(arr)
        elif '2' in dtype:
            count = len(arr) // 2
        elif '3' in dtype:
            count = len(arr)
        elif '4' in dtype:
            count = len(arr) // 4
        else:
            count = len(arr)
        
        total = count * bytes_per
        total_bytes += total
        
        print(f"  {name:<40} {count:>8} × {dtype:<8} = {total:>10} bytes")
        print(f"      → {desc}")
    
    print("-" * 100)
    print(f"  TOTAL GPU MEMORY (per tissue object): ~{total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    
    print("\\n" + "="*100)
    print("CONCLUSION")
    print("="*100)
    print("""
The USD file stores TWO mesh representations:

1. VISUALIZATION MESH (points, faceVertexIndices, normals):
   - Triangle soup for rendering
   - Higher resolution surface
   - Texture coordinates (primvars:st)
   
2. SIMULATION MESH (extMesh:*):
   - Tetrahedral volumetric mesh
   - Stores topology (vertex, edge, triangle, tetrahedron indices)
   - Stores geometry (rest lengths, volumes, deformation gradients)
   - Stores material (inverse masses, compliance parameters)
   
3. MAPPING (mapping:*):
   - Links visualization surface to simulation volume
   - Allows surface to deform with interior
   - Variable-length adjacency lists for efficient updates

This dual structure enables:
✅ High-quality rendering (fine surface mesh)
✅ Fast physics (coarse volumetric mesh)
✅ Stable Neo-Hookean material (precomputed Dm^(-1))
✅ GPU-friendly layout (flat arrays, no pointers)
""")

if __name__ == "__main__":
    usd_path = "/home/yunxin/FF-SRL/FF_SRL/FF_SRL/scenes/liverRetractionTexture.usd"
    examine_data_structure(usd_path)
