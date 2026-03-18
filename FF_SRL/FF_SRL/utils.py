import numpy as np
import warp as wp
import math
import trimesh

from pxr import Usd, UsdGeom, Gf, Sdf

@wp.kernel
def transformVecArrayKernel(src: wp.array(dtype=wp.vec3),
                            dest: wp.array(dtype=wp.vec3),
                            xform: wp.mat44):

    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(xform, p)

    dest[tid] = m

@wp.kernel
def transformVecArrayKernel2(src: wp.array(dtype=wp.vec3),
                             dest: wp.array(dtype=wp.vec3),
                             transform: wp.transform):

    tid = wp.tid()

    p = src[tid]
    m = wp.transform_point(transform, p)

    dest[tid] = m

@wp.kernel
def remapAToBKernel(aArray: wp.array(dtype=wp.vec3),
                    bArray: wp.array(dtype=wp.vec3),
                    mapArray: wp.array(dtype=wp.int32)):

    tid = wp.tid()

    pointId = mapArray[tid * 2]
    vertexId = mapArray[tid * 2 + 1]

    aArray[pointId] = bArray[vertexId]


def launchTransformVecArrayWarp(srcVecArray, outVecArray, xForm, device="cuda:0") -> None:
    wp.launch(kernel=transformVecArrayKernel, 
              dim=len(srcVecArray), 
              inputs=[srcVecArray,
                      outVecArray,
                      np.array(xForm).T],
              device=device)
    
def launchTransformVecArrayWarp2(srcVecArray, outVecArray, transform, device="cuda:0") -> None:
    wp.launch(kernel=transformVecArrayKernel2, 
              dim=len(srcVecArray), 
              inputs=[srcVecArray,
                      outVecArray,
                      transform],
              device=device)

def launchRemapAToB(srcVecArray, outVecArray, mapArray, device="cuda:0") -> None:
    wp.launch(kernel=remapAToBKernel, 
              dim=len(srcVecArray), 
              inputs=[srcVecArray,
                      outVecArray,
                      mapArray],
              device=device)
    
def launchRemapAToBLimited(srcVecArray, outVecArray, mapArray, limit, device="cuda:0") -> None:
    wp.launch(kernel=remapAToBKernel, 
              dim=limit, 
              inputs=[srcVecArray,
                      outVecArray,
                      mapArray],
              device=device)

def transformVecArray(vecArray, xForm):
    globalVec = []
    for i in range(len(vecArray)):
        globalVec.append(xForm.Transform(Gf.Vec3f(tuple(vecArray[i]))))

    return globalVec

def transformVec(vec, xForm):
    transformed = xForm.Transform(Gf.Vec3f(tuple(vec)))

    return transformed

def build_adjacency(faces, num_verts):
    """Build vertex adjacency list from triangle faces (flat int array)."""
    from collections import defaultdict
    adj = defaultdict(set)
    num_tris = len(faces) // 3
    for i in range(num_tris):
        i0, i1, i2 = faces[i*3], faces[i*3+1], faces[i*3+2]
        adj[i0].add(i1); adj[i0].add(i2)
        adj[i1].add(i0); adj[i1].add(i2)
        adj[i2].add(i0); adj[i2].add(i1)
    return adj


def laplacian_smooth_vis_points(allVisPoint_wp, allVisFace_wp, num_mesh_vis, num_mesh_faces,
                                 iterations=2, lam=0.5, device="cuda:0"):
    """Apply Laplacian smoothing to deformable mesh vis points only.

    Smooths only the first `num_mesh_vis` points (tumor), leaving rigid/laparoscope untouched.
    Uses Taubin smoothing (smooth + inflate) to avoid shrinkage.

    Args:
        allVisPoint_wp: warp array of all vis points
        allVisFace_wp: warp array of all vis face indices (flat)
        num_mesh_vis: number of deformable mesh vis points
        num_mesh_faces: number of deformable mesh vis faces
        iterations: number of smooth-inflate cycles
        lam: smoothing factor (0-1)
        device: warp device
    Returns:
        Updated warp array
    """
    pts = allVisPoint_wp.numpy()
    faces = allVisFace_wp.numpy()

    # Build adjacency from deformable mesh faces only
    adj = build_adjacency(faces[:num_mesh_faces * 3], num_mesh_vis)

    mesh_pts = pts[:num_mesh_vis].copy()
    mu = -lam / 0.9  # Taubin inflate factor (slightly less than -lam to avoid shrinkage)

    for _ in range(iterations):
        # Smooth step
        new_pts = mesh_pts.copy()
        for v in range(num_mesh_vis):
            neighbors = adj.get(v)
            if not neighbors:
                continue
            neighbor_mean = mesh_pts[list(neighbors)].mean(axis=0)
            new_pts[v] = mesh_pts[v] + lam * (neighbor_mean - mesh_pts[v])
        mesh_pts = new_pts

        # Inflate step (Taubin)
        new_pts = mesh_pts.copy()
        for v in range(num_mesh_vis):
            neighbors = adj.get(v)
            if not neighbors:
                continue
            neighbor_mean = mesh_pts[list(neighbors)].mean(axis=0)
            new_pts[v] = mesh_pts[v] + mu * (neighbor_mean - mesh_pts[v])
        mesh_pts = new_pts

    pts[:num_mesh_vis] = mesh_pts
    return wp.array(pts, dtype=wp.vec3, device=device)


def getBoxSpanVectors(xForm):
    p1 = xForm.Transform(Gf.Vec3f(-0.5, -0.5, -0.5))
    p2 = xForm.Transform(Gf.Vec3f(0.5, -0.5, -0.5))
    p3 = xForm.Transform(Gf.Vec3f(-0.5, 0.5, -0.5))
    p4 = xForm.Transform(Gf.Vec3f(-0.5, -0.5, 0.5))

    return p1, p2, p3, p4

def checkIfWithinBounds(vec, pi, pj, pk) -> bool:
    if(0.0 < Gf.Dot(vec, pi) and  Gf.Dot(vec, pi) < Gf.Dot(pi, pi)):
        if(0.0 < Gf.Dot(vec, pj) and  Gf.Dot(vec, pj) < Gf.Dot(pj, pj)):
            if(0.0 < Gf.Dot(vec, pk) and  Gf.Dot(vec, pk) < Gf.Dot(pk, pk)):
                return True

    return False

def generateTriMeshCapsule(radius:float=1.0, height:float=1.0, sectionsX:int=4, sectionsY:int=4):
    triMesh = trimesh.creation.capsule(height, radius, [sectionsX, sectionsY])
    return triMesh.vertices, triMesh.faces

def generateTriMeshSphere(radius:float=1.0, sectionsX:int=4, sectionsY:int=4):
    triMesh = trimesh.creation.uv_sphere(radius, [sectionsX, sectionsY])
    return triMesh.vertices, triMesh.faces

def calculateListVectorNormalized(inList):
    norm = math.sqrt(inList[0] * inList[0] + inList[1] * inList[1] + inList[2] * inList[2])
    return [inList[0]/norm, inList[1]/norm, inList[2]/norm]

def addListVectors(inList1, inList2):
    return [inList1[0] + inList2[0], inList1[1] + inList2[1], inList1[2] + inList2[2]]

def subtractListVectors(inList1, inList2):
    return [inList1[0] - inList2[0], inList1[1] - inList2[1], inList1[2] - inList2[2]]

def multiplyListVector(inList1, factor):
    return [inList1[0] * factor, inList1[1] * factor, inList1[2] * factor]

def crossListVector(inList1, inList2):

    return [inList1[1]*inList2[2] - inList1[2]*inList2[1],
            inList1[0]*inList2[2] - inList1[2]*inList2[0],
            inList1[0]*inList2[1] - inList1[1]*inList2[0]]

def getTransformationMatrix(shiftX: float, shiftY: float, shiftZ: float, angleX: float, angleY: float, angleZ: float) -> wp.mat44f:

    trans = wp.vec3(shiftX, shiftY, shiftZ)
    transZero = wp.vec3()
    rotZero = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 0.0), 0.0)
    rotX = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angleX)
    rotY = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angleY)
    rotZ = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angleZ)
    scale = wp.vec3(1.0, 1.0, 1.0)
    
    matrixShift = wp.transform(trans, rotZero)
    matrixX = wp.transform(transZero, rotX)
    matrixY = wp.transform(transZero, rotY)
    matrixZ = wp.transform(transZero, rotZ)

    # y=RX*RY*RZ*T*x
    # return wp.mul(matrixX, wp.mul(matrixY, wp.mul(matrixZ, matrixShift)))
    # y=T*RX*RY*RZ*x
    return wp.mul(matrixShift, wp.mul(matrixX, wp.mul(matrixY, matrixZ)))