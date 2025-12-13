"""
Stable Neo-Hookean Constraints for XPBD
Based on Macklin et al. "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (2016)
and "Nonsmooth Newton Methods for Deformable Multi-Body Dynamics" (2017/2019)

This matches the C++ xpbd-tissue-sim implementation exactly.

Author: Adapted from C++ xpbd-tissue-sim implementation
Date: December 2025
"""

import warp as wp
import math

# Constants
EPSILON = wp.constant(1e-8)

# ============================================================================
# Helper Functions: Deformation Gradient Computation
# ============================================================================

@wp.func
def computeDeformationGradient(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    Dm_inv: wp.mat33) -> wp.mat33:
    """
    Compute deformation gradient F = Ds * Dm^-1
    
    Args:
        p0, p1, p2, p3: Current vertex positions of tetrahedron
        Dm_inv: Inverse of rest shape matrix (precomputed)
    
    Returns:
        F: 3x3 deformation gradient matrix
    """
    # Ds = [p1-p0 | p2-p0 | p3-p0]
    # Each column is an edge from vertex 0
    Ds = wp.mat33(
        p1 - p0,  # column 0
        p2 - p0,  # column 1
        p3 - p0   # column 2
    )
    
    # F = Ds * Dm^-1
    F = Ds * Dm_inv
    
    return F


@wp.func
def computeGradients(
    F: wp.mat33,
    Dm_inv: wp.mat33) -> wp.mat33:
    """
    Compute ∂C/∂x for a constraint C(F)
    Uses the chain rule: ∂C/∂x = ∂C/∂F * ∂F/∂x
    
    For tetrahedral elements:
    ∂F/∂x₀ = -G₀ - G₁ - G₂
    ∂F/∂x₁ = G₀
    ∂F/∂x₂ = G₁
    ∂F/∂x₃ = G₂
    
    where G = ∂F/∂Ds * (Dm^-1)^T
    """
    # For deformation gradient: ∂F/∂Ds = I ⊗ (Dm^-1)^T
    # This gives us the gradient matrix G
    Dm_inv_T = wp.transpose(Dm_inv)
    
    # Compute cross product for gradient calculation
    # This is equivalent to computing dF/dDs in tensor notation
    dF1 = wp.cross(F[1], F[2])
    dF2 = wp.cross(F[2], F[0])
    dF3 = wp.cross(F[0], F[1])
    
    dF = wp.mat33(dF1, dF2, dF3)
    
    # Transform to vertex gradients
    G = dF * Dm_inv_T
    
    return G


# ============================================================================
# Stable Neo-Hookean Energy Functions
# ============================================================================

@wp.func
def computeI1(F: wp.mat33) -> float:
    """Compute first invariant I₁ = tr(F^T F) = ||F||²_F (Frobenius norm squared)"""
    # ||F||²_F = sum of all elements squared
    I1 = (F[0,0]*F[0,0] + F[0,1]*F[0,1] + F[0,2]*F[0,2] +
          F[1,0]*F[1,0] + F[1,1]*F[1,1] + F[1,2]*F[1,2] +
          F[2,0]*F[2,0] + F[2,1]*F[2,1] + F[2,2]*F[2,2])
    return I1


@wp.func
def computeJ(F: wp.mat33) -> float:
    """Compute Jacobian J = det(F)"""
    return wp.determinant(F)


# ============================================================================
# Deviatoric Constraint (Macklin 2017 formulation)
# ============================================================================

@wp.kernel
def stableNeohookeanDeviatoricConstraint(
    predictedVertex: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    constraintsNumber: wp.array(dtype=wp.int32),
    tetrahedron: wp.array(dtype=wp.int32),
    invRestMatrix: wp.array(dtype=wp.mat33),
    lambdas: wp.array(dtype=wp.float32),
    inverseMass: wp.array(dtype=wp.float32),
    restVolume: wp.array(dtype=wp.float32),
    activeTetrahedron: wp.array(dtype=wp.float32),
    mu: float,
    lambda_: float,
    dt: float):
    """
    Stable Neo-Hookean Deviatoric Constraint (Macklin 2017 formulation)
    
    Constraint: C = ||F||_F (Frobenius norm of deformation gradient)
    
    At rest (F=I): C = √3
    The XPBD solver will naturally handle this offset.
    """
    
    tid = wp.tid()
    
    # Check if tetrahedron is active
    if activeTetrahedron[tid] < 0.5:
        return
    
    # Get vertex indices
    v0_idx = tetrahedron[tid * 4 + 0]
    v1_idx = tetrahedron[tid * 4 + 1]
    v2_idx = tetrahedron[tid * 4 + 2]
    v3_idx = tetrahedron[tid * 4 + 3]
    
    # Get current positions
    p0 = predictedVertex[v0_idx]
    p1 = predictedVertex[v1_idx]
    p2 = predictedVertex[v2_idx]
    p3 = predictedVertex[v3_idx]
    
    # Get inverse masses
    w0 = inverseMass[v0_idx]
    w1 = inverseMass[v1_idx]
    w2 = inverseMass[v2_idx]
    w3 = inverseMass[v3_idx]
    
    # Get precomputed Dm^-1 and volume
    Dm_inv = invRestMatrix[tid]
    V0 = restVolume[tid]
    
    # Compute deformation gradient F = Ds * Dm^-1
    F = computeDeformationGradient(p0, p1, p2, p3, Dm_inv)
    
    # Compute constraint value: C = ||F||_F
    I1 = computeI1(F)
    C = wp.sqrt(I1)
    
    # Prevent numerical issues
    if C < EPSILON:
        return
    
    # Compute gradient: ∂C/∂x = (1/C) * F * Q^T
    # Following C++ implementation exactly
    inv_C = 1.0 / C
    Q_T = wp.transpose(Dm_inv)  # Q = Dm_inv in our notation
    
    # Compute A = (1/C) * F * Q^T (column-major)
    # Column 0 of A is gradient wrt vertex 1
    # Column 1 of A is gradient wrt vertex 2  
    # Column 2 of A is gradient wrt vertex 3
    # Gradient wrt vertex 0 = -(g1 + g2 + g3)
    
    # Manual matrix multiplication: A = inv_C * F * Q_T
    # g1 = A[:,0], g2 = A[:,1], g3 = A[:,2]
    g1_x = inv_C * (F[0,0]*Q_T[0,0] + F[0,1]*Q_T[1,0] + F[0,2]*Q_T[2,0])
    g1_y = inv_C * (F[1,0]*Q_T[0,0] + F[1,1]*Q_T[1,0] + F[1,2]*Q_T[2,0])
    g1_z = inv_C * (F[2,0]*Q_T[0,0] + F[2,1]*Q_T[1,0] + F[2,2]*Q_T[2,0])
    
    g2_x = inv_C * (F[0,0]*Q_T[0,1] + F[0,1]*Q_T[1,1] + F[0,2]*Q_T[2,1])
    g2_y = inv_C * (F[1,0]*Q_T[0,1] + F[1,1]*Q_T[1,1] + F[1,2]*Q_T[2,1])
    g2_z = inv_C * (F[2,0]*Q_T[0,1] + F[2,1]*Q_T[1,1] + F[2,2]*Q_T[2,1])
    
    g3_x = inv_C * (F[0,0]*Q_T[0,2] + F[0,1]*Q_T[1,2] + F[0,2]*Q_T[2,2])
    g3_y = inv_C * (F[1,0]*Q_T[0,2] + F[1,1]*Q_T[1,2] + F[1,2]*Q_T[2,2])
    g3_z = inv_C * (F[2,0]*Q_T[0,2] + F[2,1]*Q_T[1,2] + F[2,2]*Q_T[2,2])
    
    g1 = wp.vec3(g1_x, g1_y, g1_z)
    g2 = wp.vec3(g2_x, g2_y, g2_z)
    g3 = wp.vec3(g3_x, g3_y, g3_z)
    g0 = -(g1 + g2 + g3)
    
    # Constraint mass (sum of weighted gradient magnitudes)
    w_sum = w0 * wp.dot(g0, g0) + \
            w1 * wp.dot(g1, g1) + \
            w2 * wp.dot(g2, g2) + \
            w3 * wp.dot(g3, g3)
    
    if w_sum < EPSILON:
        return
    
    # Compliance: α = 1/(μ*V₀)
    alpha = 1.0 / (mu * V0)
    
    # XPBD lambda update
    lam = lambdas[tid]
    delta_lambda = -(C + alpha * lam / (dt * dt)) / (w_sum + alpha / (dt * dt))
    lambdas[tid] = lam + delta_lambda
    
    # Apply position corrections
    wp.atomic_add(dP, v0_idx, g0 * delta_lambda)
    wp.atomic_add(dP, v1_idx, g1 * delta_lambda)
    wp.atomic_add(dP, v2_idx, g2 * delta_lambda)
    wp.atomic_add(dP, v3_idx, g3 * delta_lambda)
    
    # Update constraint counters
    wp.atomic_add(constraintsNumber, v0_idx, 1)
    wp.atomic_add(constraintsNumber, v1_idx, 1)
    wp.atomic_add(constraintsNumber, v2_idx, 1)
    wp.atomic_add(constraintsNumber, v3_idx, 1)


# ============================================================================
# Hydrostatic Constraint (Macklin 2017 formulation with Taylor series)
# ============================================================================

@wp.kernel
def stableNeohookeanHydrostaticConstraint(
    predictedVertex: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    constraintsNumber: wp.array(dtype=wp.int32),
    tetrahedron: wp.array(dtype=wp.int32),
    invRestMatrix: wp.array(dtype=wp.mat33),
    lambdas: wp.array(dtype=wp.float32),
    inverseMass: wp.array(dtype=wp.float32),
    restVolume: wp.array(dtype=wp.float32),
    activeTetrahedron: wp.array(dtype=wp.float32),
    mu: float,
    lambda_: float,
    dt: float):
    """
    Stable Neo-Hookean Hydrostatic Constraint (Macklin 2017 formulation)
    
    Constraint: C = -γ + log(J) where γ = μ/λ
    
    For J >= 1: log(J) is computed directly
    For J < 1: log(J) approximated by Taylor series to avoid log(0)
               log(J) ≈ (J-1) - (J-1)²/2 + (J-1)³/3
    
    This ensures stability under compression while preserving volume.
    """
    
    tid = wp.tid()
    
    if activeTetrahedron[tid] < 0.5:
        return
    
    # Get vertex indices
    v0_idx = tetrahedron[tid * 4 + 0]
    v1_idx = tetrahedron[tid * 4 + 1]
    v2_idx = tetrahedron[tid * 4 + 2]
    v3_idx = tetrahedron[tid * 4 + 3]
    
    # Get current positions
    p0 = predictedVertex[v0_idx]
    p1 = predictedVertex[v1_idx]
    p2 = predictedVertex[v2_idx]
    p3 = predictedVertex[v3_idx]
    
    # Get inverse masses
    w0 = inverseMass[v0_idx]
    w1 = inverseMass[v1_idx]
    w2 = inverseMass[v2_idx]
    w3 = inverseMass[v3_idx]
    
    # Get precomputed values
    Dm_inv = invRestMatrix[tid]
    V0 = restVolume[tid]
    
    # Compute deformation gradient
    F = computeDeformationGradient(p0, p1, p2, p3, Dm_inv)
    
    # Compute Jacobian
    J = computeJ(F)
    
    # γ = μ/λ (material-dependent offset)
    gamma = mu / (lambda_ + EPSILON)
    
    # Compute constraint value with Taylor series stability
    C = 0.0
    grad_factor = 0.0  # Derivative of log(J) term
    
    if J >= 1.0:
        # Normal case: C = -γ + log(J)
        C = -gamma + wp.log(J + EPSILON)
        grad_factor = 1.0 / (J + EPSILON)
    else:
        # Compressed case: Taylor series for log(J)
        # log(J) ≈ (J-1) - (J-1)²/2 + (J-1)³/3
        J_minus_1 = J - 1.0
        log_J_approx = J_minus_1 - 0.5*J_minus_1*J_minus_1 + (1.0/3.0)*J_minus_1*J_minus_1*J_minus_1
        C = -gamma + log_J_approx
        
        # Derivative: d/dJ[log(J)] ≈ 1 - (J-1) + (J-1)²
        grad_factor = 1.0 - J_minus_1 + J_minus_1*J_minus_1
    
    # Compute gradient ∂C/∂x using cofactor matrix (F_cross)
    # Following C++ implementation exactly
    # F_cross = [f2 × f3, f3 × f1, f1 × f2] where f_i are columns of F
    
    # Cofactor matrix columns (cross products of F columns)
    f0 = wp.vec3(F[0,0], F[1,0], F[2,0])  # column 0 of F
    f1 = wp.vec3(F[0,1], F[1,1], F[2,1])  # column 1 of F
    f2 = wp.vec3(F[0,2], F[1,2], F[2,2])  # column 2 of F
    
    F_cross_col0 = wp.cross(f1, f2)  # f2 × f3 (column 1 × column 2)
    F_cross_col1 = wp.cross(f2, f0)  # f3 × f1 (column 2 × column 0)
    F_cross_col2 = wp.cross(f0, f1)  # f1 × f2 (column 0 × column 1)
    
    # Assemble F_cross matrix
    F_cross = wp.mat33(F_cross_col0, F_cross_col1, F_cross_col2)
    
    # Compute A = fac * F_cross * Q^T
    Q_T = wp.transpose(Dm_inv)
    
    # Manual matrix multiplication for gradients
    # g1 = first column of A (gradient wrt vertex 1)
    g1_x = grad_factor * (F_cross[0,0]*Q_T[0,0] + F_cross[0,1]*Q_T[1,0] + F_cross[0,2]*Q_T[2,0])
    g1_y = grad_factor * (F_cross[1,0]*Q_T[0,0] + F_cross[1,1]*Q_T[1,0] + F_cross[1,2]*Q_T[2,0])
    g1_z = grad_factor * (F_cross[2,0]*Q_T[0,0] + F_cross[2,1]*Q_T[1,0] + F_cross[2,2]*Q_T[2,0])
    
    # g2 = second column of A (gradient wrt vertex 2)
    g2_x = grad_factor * (F_cross[0,0]*Q_T[0,1] + F_cross[0,1]*Q_T[1,1] + F_cross[0,2]*Q_T[2,1])
    g2_y = grad_factor * (F_cross[1,0]*Q_T[0,1] + F_cross[1,1]*Q_T[1,1] + F_cross[1,2]*Q_T[2,1])
    g2_z = grad_factor * (F_cross[2,0]*Q_T[0,1] + F_cross[2,1]*Q_T[1,1] + F_cross[2,2]*Q_T[2,1])
    
    # g3 = third column of A (gradient wrt vertex 3)
    g3_x = grad_factor * (F_cross[0,0]*Q_T[0,2] + F_cross[0,1]*Q_T[1,2] + F_cross[0,2]*Q_T[2,2])
    g3_y = grad_factor * (F_cross[1,0]*Q_T[0,2] + F_cross[1,1]*Q_T[1,2] + F_cross[1,2]*Q_T[2,2])
    g3_z = grad_factor * (F_cross[2,0]*Q_T[0,2] + F_cross[2,1]*Q_T[1,2] + F_cross[2,2]*Q_T[2,2])
    
    g1 = wp.vec3(g1_x, g1_y, g1_z)
    g2 = wp.vec3(g2_x, g2_y, g2_z)
    g3 = wp.vec3(g3_x, g3_y, g3_z)
    g0 = -(g1 + g2 + g3)  # gradient wrt vertex 0
    
    # Constraint mass (weighted squared gradient norm)
    w_sum = w0 * wp.dot(g0, g0) + \
            w1 * wp.dot(g1, g1) + \
            w2 * wp.dot(g2, g2) + \
            w3 * wp.dot(g3, g3)
    
    if w_sum < EPSILON:
        return
    
    # Compliance: α = 1/(λ*V₀)
    alpha = 1.0 / (lambda_ * V0 + EPSILON)
    
    # XPBD lambda update
    lam = lambdas[tid]
    delta_lambda = -(C + alpha * lam / (dt * dt)) / (w_sum + alpha / (dt * dt))
    new_lambda = lam + delta_lambda
    lambdas[tid] = new_lambda
    
    # Apply position corrections
    correction = delta_lambda
    wp.atomic_add(dP, v0_idx, g0 * correction)
    wp.atomic_add(dP, v1_idx, g1 * correction)
    wp.atomic_add(dP, v2_idx, g2 * correction)
    wp.atomic_add(dP, v3_idx, g3 * correction)
    
    # Update constraint counters
    wp.atomic_add(constraintsNumber, v0_idx, 1)
    wp.atomic_add(constraintsNumber, v1_idx, 1)
    wp.atomic_add(constraintsNumber, v2_idx, 1)
    wp.atomic_add(constraintsNumber, v3_idx, 1)


# ============================================================================
# Utility: Precompute Inverse Rest Matrix
# ============================================================================

@wp.kernel
def computeInverseRestMatrices(
    vertex: wp.array(dtype=wp.vec3),
    tetrahedron: wp.array(dtype=wp.int32),
    invRestMatrix: wp.array(dtype=wp.mat33),
    restVolume: wp.array(dtype=wp.float32)):
    """
    Precompute Dm^-1 for all tetrahedra
    
    Dm = [x1-x0 | x2-x0 | x3-x0] (rest configuration)
    """
    tid = wp.tid()
    
    v0_idx = tetrahedron[tid * 4 + 0]
    v1_idx = tetrahedron[tid * 4 + 1]
    v2_idx = tetrahedron[tid * 4 + 2]
    v3_idx = tetrahedron[tid * 4 + 3]
    
    x0 = vertex[v0_idx]
    x1 = vertex[v1_idx]
    x2 = vertex[v2_idx]
    x3 = vertex[v3_idx]
    
    # Dm matrix (rest shape)
    Dm = wp.mat33(
        x1 - x0,
        x2 - x0,
        x3 - x0
    )
    
    # Compute inverse
    Dm_inv = wp.inverse(Dm)
    invRestMatrix[tid] = Dm_inv
    
    # Compute rest volume
    det_Dm = wp.determinant(Dm)
    V0 = wp.abs(det_Dm) / 6.0
    restVolume[tid] = V0
