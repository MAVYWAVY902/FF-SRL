"""
Unified Distance Constraint for GPU-based tissue adhesion/dissection.

Port of InterDeformUnifiedDistanceConstraint from xpbd-tissue-sim (C++ Gauss-Seidel)
to NVIDIA Warp (GPU Jacobi solver).

Constraint: C(d) = d - d*(d), where:
  - d: current unsigned separation distance (vertex to triangle anchor point)
  - d*(d): smooth C¹-continuous target distance curve
  - Frozen contact frame: barycentric coords fixed at init, normal updated per iteration

Key properties:
  - C¹ continuous everywhere (no gradient jumps)
  - Monotonic: dC/dd > 0 globally (XPBD-stable under Jacobi averaging)
  - Dynamic compliance: hard when d < d_contact, soft otherwise
  - Elongation-based breaking: stretch = d - d_initial > threshold

References:
  - InterDeformUnifiedDistanceConstraint.cpp (xpbd-tissue-sim)
  - Macklin et al., "Small Steps in Physics Simulation" (2019)
"""

import warp as wp
import sys

FLOAT_EPSILON = wp.constant(sys.float_info.epsilon)


@wp.func
def compute_target_distance(
    d: float,
    d0: float,        # initial_distance (equilibrium)
    d_contact: float,  # half-width of C1 transition zone (delta)
    d_rest: float,     # mid-range target distance
    d_neutral_start: float  # transition zone parameter
) -> float:
    """
    Compute target distance d*(d) using C¹-smooth slope-interpolation.

    Three regions:
      - d < d0 - delta: compression (slope = 0.01, hard push-back)
      - d > d0 + delta: extension (slope = beta, soft pull-back)
      - |d - d0| <= delta: C¹ smooth transition (cubic blend)

    Guarantees:
      - dd*/dd in [0.01, beta] globally
      - dC/dd in [1-beta, 0.99] globally (monotonic, always positive)
    """
    # Beta: extension stiffness slope
    beta = 0.3  # fallback
    if d_neutral_start > 1e-6:
        beta = d_rest / d_neutral_start
        beta = wp.max(0.01, wp.min(0.99, beta))

    # Delta: half-width of C1 transition zone
    delta = wp.max(1e-5, d_contact)

    slope_left = 0.01   # hard push-back for compression
    slope_right = beta   # soft pull-back for extension

    # Region 1: Compression zone (d < d0 - delta)
    if d < d0 - delta:
        return d0 + slope_left * (d - d0)

    # Region 2: Extension zone (d > d0 + delta)
    if d > d0 + delta:
        return d0 + slope_right * (d - d0)

    # Region 3: C1 smooth transition zone [d0-delta, d0+delta]
    x = d - (d0 - delta)           # distance from left edge, in [0, 2*delta]
    t = x / (2.0 * delta)          # normalized parameter in [0, 1]

    # Target at left edge
    d_target_left = d0 + slope_left * (-delta)  # = d0 - slope_left * delta

    # Integral of cubic blend: integral(3t^2 - 2t^3) = t^3 - t^4/2
    term_blend = t * t * t - 0.5 * t * t * t * t

    # d* = d_left + slope_left * x + (slope_right - slope_left) * 2*delta * integral_blend
    d_target = d_target_left + slope_left * x + (slope_right - slope_left) * (2.0 * delta) * term_blend

    return d_target


@wp.func
def compute_dC_dd(
    d: float,
    d0: float,
    d_contact: float,
    d_rest: float,
    d_neutral_start: float
) -> float:
    """
    Compute dC/dd = 1 - dd*/dd via finite differences.
    This is needed for the chain-rule gradient scaling.
    """
    eps = 1e-8
    d_target_plus = compute_target_distance(d + eps, d0, d_contact, d_rest, d_neutral_start)
    d_target_minus = compute_target_distance(d - eps, d0, d_contact, d_rest, d_neutral_start)
    dd_target_dd = (d_target_plus - d_target_minus) / (2.0 * eps)
    return 1.0 - dd_target_dd


# =============================================================================
# Main adhesion constraint kernel (Jacobi parallel, atomic dP accumulation)
# =============================================================================

@wp.kernel
def unifiedAdhesionConstraintsDO(
    predictedVertex: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    constraintsNumber: wp.array(dtype=int),
    # Per-bond topology (flat arrays, all envs concatenated)
    adhesionVertexId: wp.array(dtype=int),      # vertex index from object A (1 per bond)
    adhesionTriId: wp.array(dtype=int),          # triangle vertex indices from object B (3 per bond, flat)
    adhesionTriBar: wp.array(dtype=wp.vec3),     # frozen barycentric coords (1 per bond)
    adhesionRestGap: wp.array(dtype=float),      # initial_distance d0 (1 per bond)
    adhesionActive: wp.array(dtype=float),       # 1.0=active, 0.0=broken (1 per bond)
    # Per-bond cached state
    adhesionNormalCache: wp.array(dtype=wp.vec3),  # cached normal (updated per substep)
    adhesionCacheValid: wp.array(dtype=int),       # 0=invalid, 1=valid
    # Curve parameters (global, same for all bonds)
    d_contact: float,
    d_rest: float,
    d_neutral_start: float,
    # Breaking parameters
    breakRatio: float,
    stretchAbsMin: float,
    # Compliance
    defaultAlpha: float,
    dT: float):
    """
    Unified distance constraint for deform-deform adhesion.

    Port of InterDeformUnifiedDistanceConstraint::evaluate() + gradient()
    adapted for Jacobi parallel solver with atomic dP accumulation.

    Each thread processes one adhesion bond.
    """

    tid = wp.tid()

    # Early exit for broken bonds
    active = adhesionActive[tid]
    if active < 0.5:
        return

    # --- Read bond topology ---
    vertexId = adhesionVertexId[tid]
    triId0 = adhesionTriId[tid * 3 + 0]
    triId1 = adhesionTriId[tid * 3 + 1]
    triId2 = adhesionTriId[tid * 3 + 2]

    vertexPos = predictedVertex[vertexId]
    triP1 = predictedVertex[triId0]
    triP2 = predictedVertex[triId1]
    triP3 = predictedVertex[triId2]

    baryCoords = adhesionTriBar[tid]
    d0 = adhesionRestGap[tid]  # initial_distance

    # --- Frozen Contact Frame ---
    # Reconstruct anchor point using FROZEN barycentric coords
    anchorPoint = triP1 * baryCoords[0] + triP2 * baryCoords[1] + triP3 * baryCoords[2]

    # Compute separation vector and distance
    diff = vertexPos - anchorPoint
    d = wp.length(diff)

    # Compute or update normal
    cache_valid = adhesionCacheValid[tid]

    normal = wp.vec3(0.0, 0.0, 1.0)  # fallback

    if cache_valid == 0:
        # First evaluation this substep - compute and freeze contact geometry
        use_face_normal = (d < d_contact)

        if use_face_normal:
            edge1 = triP2 - triP1
            edge2 = triP3 - triP1
            tri_normal = wp.cross(edge1, edge2)
            area2 = wp.length(tri_normal)
            if area2 > 1e-12:
                normal = tri_normal / area2
                # Ensure normal points toward vertex
                if wp.dot(diff, normal) < 0.0:
                    normal = -normal
            # else: keep fallback
        elif d > 1e-12:
            normal = diff / d
        # else: keep fallback

        adhesionNormalCache[tid] = normal
        adhesionCacheValid[tid] = 1
    else:
        # Cache valid - update normal direction (prevents sliding)
        if d > 1e-12:
            normal = diff / d
            adhesionNormalCache[tid] = normal
        else:
            normal = adhesionNormalCache[tid]

    # --- Dynamic Compliance ---
    alpha = defaultAlpha
    if d <= d_contact:
        alpha = 1e-9  # Hard constraint (collision-like)

    alpha_tilde = alpha / (dT * dT)

    # --- Safety check ---
    if d > 1.0:
        d = d0  # Use safe fallback

    # --- Breaking check: elongation-based ---
    current_stretch = d - d0
    stretch_tolerance = d0 * (breakRatio - 1.0)
    max_allowed_stretch = wp.max(stretch_tolerance, stretchAbsMin)

    if current_stretch > max_allowed_stretch:
        adhesionActive[tid] = 0.0  # Permanently break bond
        return

    # --- Constraint evaluation: C(d) = d - d*(d) ---
    d_target = compute_target_distance(d, d0, d_contact, d_rest, d_neutral_start)
    C = d - d_target

    # --- Gradient with chain rule: dC/dp = (dC/dd) * (dd/dp) ---
    dC_dd = compute_dC_dd(d, d0, d_contact, d_rest, d_neutral_start)

    b1 = baryCoords[0]
    b2 = baryCoords[1]
    b3 = baryCoords[2]

    # Gradient w.r.t. vertex: dC/dp_vertex = dC_dd * n
    g_vertex = normal * dC_dd
    # Gradient w.r.t. triangle vertices: dC/dp_i = -dC_dd * b_i * n
    g_tri1 = normal * (-dC_dd * b1)
    g_tri2 = normal * (-dC_dd * b2)
    g_tri3 = normal * (-dC_dd * b3)

    # --- XPBD projection (Jacobi: accumulate into dP) ---
    # Effective inverse mass (using unit masses for adhesion vertices)
    invMassVertex = 1.0
    invMassTri1 = 1.0
    invMassTri2 = 1.0
    invMassTri3 = 1.0

    w = (invMassVertex * wp.dot(g_vertex, g_vertex) +
         invMassTri1 * wp.dot(g_tri1, g_tri1) +
         invMassTri2 * wp.dot(g_tri2, g_tri2) +
         invMassTri3 * wp.dot(g_tri3, g_tri3))

    if w < FLOAT_EPSILON:
        return

    # XPBD delta lambda (no persistent lambda for Jacobi - use single-iteration form)
    dLambda = -C / (w + alpha_tilde)

    # Position corrections
    wp.atomic_add(dP, vertexId, g_vertex * dLambda * invMassVertex)
    wp.atomic_add(dP, triId0, g_tri1 * dLambda * invMassTri1)
    wp.atomic_add(dP, triId1, g_tri2 * dLambda * invMassTri2)
    wp.atomic_add(dP, triId2, g_tri3 * dLambda * invMassTri3)

    wp.atomic_add(constraintsNumber, vertexId, 1)
    wp.atomic_add(constraintsNumber, triId0, 1)
    wp.atomic_add(constraintsNumber, triId1, 1)
    wp.atomic_add(constraintsNumber, triId2, 1)


# =============================================================================
# Reset cached contact frame at start of each substep
# =============================================================================

@wp.kernel
def resetAdhesionCacheKernel(
    adhesionCacheValid: wp.array(dtype=int)):
    """Reset frozen contact frame validity at start of each substep."""
    tid = wp.tid()
    adhesionCacheValid[tid] = 0


# =============================================================================
# Count remaining active bonds (for observation/reward in RL)
# =============================================================================

@wp.kernel
def countActiveBondsPerEnvKernel(
    adhesionActive: wp.array(dtype=float),
    bondsPerEnv: int,
    activeCounts: wp.array(dtype=int)):
    """
    Count active adhesion bonds per environment.
    Thread id = bond id. Uses atomic_add to accumulate per-env counts.
    """
    tid = wp.tid()

    if adhesionActive[tid] < 0.5:
        return

    envId = tid / bondsPerEnv
    wp.atomic_add(activeCounts, envId, 1)


# =============================================================================
# Utility: Compute barycentric coordinates for bond initialization
# =============================================================================

@wp.func
def compute_barycentric(
    vertex_pos: wp.vec3,
    tri_p1: wp.vec3,
    tri_p2: wp.vec3,
    tri_p3: wp.vec3
) -> wp.vec3:
    """
    Compute barycentric coordinates of the closest point projection
    of vertex_pos onto triangle (tri_p1, tri_p2, tri_p3).
    Returns (w, v, u) where closest = w*p1 + v*p2 + u*p3.
    """
    edge1 = tri_p2 - tri_p1
    edge2 = tri_p3 - tri_p1
    tri_normal = wp.cross(edge1, edge2)
    area = wp.length(tri_normal)

    if area < 1e-12:
        return wp.vec3(1.0, 0.0, 0.0)

    normal = tri_normal / area

    # Signed distance to plane
    signed_dist = wp.dot(vertex_pos - tri_p1, normal)

    # Project onto plane
    projected = vertex_pos - signed_dist * normal

    # Barycentric coordinates via dot products
    v0 = edge2
    v1 = edge1
    v2 = projected - tri_p1

    dot00 = wp.dot(v0, v0)
    dot01 = wp.dot(v0, v1)
    dot02 = wp.dot(v0, v2)
    dot11 = wp.dot(v1, v1)
    dot12 = wp.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if wp.abs(denom) < 1e-12:
        return wp.vec3(1.0, 0.0, 0.0)

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    # Clamp if outside triangle
    u_c = wp.max(0.0, wp.min(1.0, u))
    v_c = wp.max(0.0, wp.min(1.0, v))
    s = u_c + v_c
    if s > 1.0:
        inv_s = 1.0 / s
        u_c = u_c * inv_s
        v_c = v_c * inv_s
    w_c = 1.0 - u_c - v_c

    return wp.vec3(w_c, v_c, u_c)


# =============================================================================
# Rigid-Deform Adhesion Constraint (bone=fixed, tumor=deformable)
# Port of UnifiedDistanceConstraint from xpbd-tissue-sim
# =============================================================================

@wp.kernel
def rigidDeformAdhesionConstraintsDO(
    predictedVertex: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    constraintsNumber: wp.array(dtype=int),
    # Per-bond topology (flat arrays, all envs concatenated)
    rigidAdhesionRigidPoint: wp.array(dtype=wp.vec3),  # fixed rigid body point (world space, 1 per bond)
    rigidAdhesionTriId: wp.array(dtype=int),            # triangle vertex indices (3 per bond, flat)
    rigidAdhesionTriBar: wp.array(dtype=wp.vec3),       # frozen barycentric coords (1 per bond)
    rigidAdhesionRestGap: wp.array(dtype=float),        # initial_distance d0 (1 per bond)
    rigidAdhesionActive: wp.array(dtype=float),         # 1.0=active, 0.0=broken (1 per bond)
    # Per-bond cached state
    rigidAdhesionNormalCache: wp.array(dtype=wp.vec3),
    rigidAdhesionCacheValid: wp.array(dtype=int),
    # Curve parameters
    d_contact: float,
    d_rest: float,
    d_neutral_start: float,
    # Breaking parameters
    breakRatio: float,
    stretchAbsMin: float,
    # Compliance
    defaultAlpha: float,
    dT: float):
    """
    Unified distance constraint for rigid-deform adhesion.

    Port of UnifiedDistanceConstraint::evaluate() + gradient()
    for Jacobi parallel solver. The rigid point (e.g., bone surface)
    is fixed in world space; only the deformable triangle vertices
    (e.g., tumor surface) receive position corrections.

    Each thread processes one adhesion bond.
    """

    tid = wp.tid()

    # Early exit for broken bonds
    active = rigidAdhesionActive[tid]
    if active < 0.5:
        return

    # --- Read bond topology ---
    rigidPoint = rigidAdhesionRigidPoint[tid]  # fixed world-space point on bone
    triId0 = rigidAdhesionTriId[tid * 3 + 0]
    triId1 = rigidAdhesionTriId[tid * 3 + 1]
    triId2 = rigidAdhesionTriId[tid * 3 + 2]

    triP1 = predictedVertex[triId0]
    triP2 = predictedVertex[triId1]
    triP3 = predictedVertex[triId2]

    baryCoords = rigidAdhesionTriBar[tid]
    d0 = rigidAdhesionRestGap[tid]

    # --- Frozen Contact Frame ---
    # Anchor point on deformable triangle using FROZEN barycentric coords
    anchorPoint = triP1 * baryCoords[0] + triP2 * baryCoords[1] + triP3 * baryCoords[2]

    # Separation: rigid point to deformable anchor
    diff = rigidPoint - anchorPoint
    d = wp.length(diff)

    # Compute or update normal
    cache_valid = rigidAdhesionCacheValid[tid]

    normal = wp.vec3(0.0, 0.0, 1.0)  # fallback

    if cache_valid == 0:
        use_face_normal = (d < d_contact)

        if use_face_normal:
            edge1 = triP2 - triP1
            edge2 = triP3 - triP1
            tri_normal = wp.cross(edge1, edge2)
            area2 = wp.length(tri_normal)
            if area2 > 1e-12:
                normal = tri_normal / area2
                if wp.dot(diff, normal) < 0.0:
                    normal = -normal
        elif d > 1e-12:
            normal = diff / d

        rigidAdhesionNormalCache[tid] = normal
        rigidAdhesionCacheValid[tid] = 1
    else:
        if d > 1e-12:
            normal = diff / d
            rigidAdhesionNormalCache[tid] = normal
        else:
            normal = rigidAdhesionNormalCache[tid]

    # --- Dynamic Compliance ---
    alpha = defaultAlpha
    if d <= d_contact:
        alpha = 1e-9

    alpha_tilde = alpha / (dT * dT)

    # --- Safety check ---
    if d > 1.0:
        d = d0

    # --- Breaking check: elongation-based ---
    current_stretch = d - d0
    stretch_tolerance = d0 * (breakRatio - 1.0)
    max_allowed_stretch = wp.max(stretch_tolerance, stretchAbsMin)

    if current_stretch > max_allowed_stretch:
        rigidAdhesionActive[tid] = 0.0
        return

    # --- Constraint evaluation: C(d) = d - d*(d) ---
    d_target = compute_target_distance(d, d0, d_contact, d_rest, d_neutral_start)
    C = d - d_target

    # --- Gradient with chain rule ---
    dC_dd = compute_dC_dd(d, d0, d_contact, d_rest, d_neutral_start)

    b1 = baryCoords[0]
    b2 = baryCoords[1]
    b3 = baryCoords[2]

    # Rigid point gradient (NOT applied - rigid body is fixed, invMass=0)
    # g_rigid = normal * dC_dd  # would be applied if rigid body could move

    # Triangle vertex gradients: dC/dp_i = -dC_dd * b_i * n
    g_tri1 = normal * (-dC_dd * b1)
    g_tri2 = normal * (-dC_dd * b2)
    g_tri3 = normal * (-dC_dd * b3)

    # --- XPBD projection (only deformable vertices get corrections) ---
    invMassTri1 = 1.0
    invMassTri2 = 1.0
    invMassTri3 = 1.0

    # w only includes deformable terms (rigid invMass = 0)
    w = (invMassTri1 * wp.dot(g_tri1, g_tri1) +
         invMassTri2 * wp.dot(g_tri2, g_tri2) +
         invMassTri3 * wp.dot(g_tri3, g_tri3))

    if w < FLOAT_EPSILON:
        return

    dLambda = -C / (w + alpha_tilde)

    # Only apply corrections to deformable triangle vertices
    wp.atomic_add(dP, triId0, g_tri1 * dLambda * invMassTri1)
    wp.atomic_add(dP, triId1, g_tri2 * dLambda * invMassTri2)
    wp.atomic_add(dP, triId2, g_tri3 * dLambda * invMassTri3)

    wp.atomic_add(constraintsNumber, triId0, 1)
    wp.atomic_add(constraintsNumber, triId1, 1)
    wp.atomic_add(constraintsNumber, triId2, 1)


# =============================================================================
# Reset rigid adhesion cache
# =============================================================================

@wp.kernel
def resetRigidAdhesionCacheKernel(
    rigidAdhesionCacheValid: wp.array(dtype=int)):
    """Reset frozen contact frame for rigid-deform adhesion bonds."""
    tid = wp.tid()
    rigidAdhesionCacheValid[tid] = 0


# =============================================================================
# Count active rigid adhesion bonds per env
# =============================================================================

@wp.kernel
def countActiveRigidBondsPerEnvKernel(
    rigidAdhesionActive: wp.array(dtype=float),
    bondsPerEnv: int,
    activeCounts: wp.array(dtype=int)):
    """Count active rigid-deform adhesion bonds per environment."""
    tid = wp.tid()

    if rigidAdhesionActive[tid] < 0.5:
        return

    envId = tid / bondsPerEnv
    wp.atomic_add(activeCounts, envId, 1)


# =============================================================================
# Initialize rigid-deform adhesion bonds
# =============================================================================

@wp.kernel
def initRigidAdhesionBondsKernel(
    vertex: wp.array(dtype=wp.vec3),
    rigidAdhesionRigidPoint: wp.array(dtype=wp.vec3),
    rigidAdhesionTriId: wp.array(dtype=int),
    rigidAdhesionTriBar: wp.array(dtype=wp.vec3),
    rigidAdhesionRestGap: wp.array(dtype=float)):
    """
    Initialize barycentric coordinates and rest distances for rigid-deform bonds.
    The rigid point is already set (fixed world-space position on bone).
    Computes barycentric projection of rigid point onto the triangle, and rest gap.
    """
    tid = wp.tid()

    rigidPos = rigidAdhesionRigidPoint[tid]
    t0 = rigidAdhesionTriId[tid * 3 + 0]
    t1 = rigidAdhesionTriId[tid * 3 + 1]
    t2 = rigidAdhesionTriId[tid * 3 + 2]

    p1 = vertex[t0]
    p2 = vertex[t1]
    p3 = vertex[t2]

    bary = compute_barycentric(rigidPos, p1, p2, p3)
    rigidAdhesionTriBar[tid] = bary

    anchor = p1 * bary[0] + p2 * bary[1] + p3 * bary[2]
    dist = wp.length(rigidPos - anchor)
    rigidAdhesionRestGap[tid] = dist


# =============================================================================
# Deform-Deform adhesion bond initialization
# =============================================================================

@wp.kernel
def initAdhesionBondsKernel(
    vertex: wp.array(dtype=wp.vec3),
    adhesionVertexId: wp.array(dtype=int),
    adhesionTriId: wp.array(dtype=int),
    adhesionTriBar: wp.array(dtype=wp.vec3),
    adhesionRestGap: wp.array(dtype=float)):
    """
    Initialize barycentric coordinates and rest distances for adhesion bonds.
    Called once at scene setup.
    """
    tid = wp.tid()

    vId = adhesionVertexId[tid]
    t0 = adhesionTriId[tid * 3 + 0]
    t1 = adhesionTriId[tid * 3 + 1]
    t2 = adhesionTriId[tid * 3 + 2]

    vPos = vertex[vId]
    p1 = vertex[t0]
    p2 = vertex[t1]
    p3 = vertex[t2]

    # Compute barycentric coordinates
    bary = compute_barycentric(vPos, p1, p2, p3)
    adhesionTriBar[tid] = bary

    # Compute rest distance
    anchor = p1 * bary[0] + p2 * bary[1] + p3 * bary[2]
    dist = wp.length(vPos - anchor)
    adhesionRestGap[tid] = dist
