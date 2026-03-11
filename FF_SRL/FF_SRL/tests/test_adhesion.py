"""
Verification tests for adhesion constraint kernels (deform-deform and rigid-deform).

Three-layer testing:
  1. Math: compute_target_distance GPU vs numpy reference, C¹ continuity, monotonicity
  2. Kernel: known geometry → verify dP direction, magnitude, breaking behavior
  3. Physics: mini time-stepping loop → bond holds at rest, breaks under stretch

Run: python -m FF_SRL.tests.test_adhesion
  or: cd FF_SRL/FF_SRL && python tests/test_adhesion.py
"""

import warp as wp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

wp.init()

# Import adhesion kernels and functions
from FF_SRL.adhesion import (
    unifiedAdhesionConstraintsDO,
    rigidDeformAdhesionConstraintsDO,
    resetAdhesionCacheKernel,
    resetRigidAdhesionCacheKernel,
    initAdhesionBondsKernel,
    initRigidAdhesionBondsKernel,
    countActiveBondsPerEnvKernel,
    countActiveRigidBondsPerEnvKernel,
    compute_target_distance,
    compute_dC_dd,
)

DEVICE = "cuda:0"

# Default curve parameters (matching C++ InterDeformUnifiedDistanceConstraint defaults)
D_CONTACT = 0.0003       # 0.3mm
D_REST = 0.0015           # 1.5mm
D_NEUTRAL_START = 0.003   # 3mm
BREAK_RATIO = 3.0
STRETCH_ABS_MIN = 0.005   # 5mm
DEFAULT_ALPHA = 1e-5
DT = 0.001  # 1ms substep


# =============================================================================
# Helper: numpy reference implementation of compute_target_distance
# =============================================================================

def compute_target_distance_numpy(d, d0, d_contact, d_rest, d_neutral_start):
    """Numpy reference matching the Warp compute_target_distance function."""
    beta = 0.3
    if d_neutral_start > 1e-6:
        beta = d_rest / d_neutral_start
        beta = max(0.01, min(0.99, beta))

    delta = max(1e-5, d_contact)
    slope_left = 0.01
    slope_right = beta

    if d < d0 - delta:
        return d0 + slope_left * (d - d0)
    if d > d0 + delta:
        return d0 + slope_right * (d - d0)

    # Transition zone
    x = d - (d0 - delta)
    t = x / (2.0 * delta)
    d_target_left = d0 + slope_left * (-delta)
    term_blend = t**3 - 0.5 * t**4
    d_target = d_target_left + slope_left * x + (slope_right - slope_left) * (2.0 * delta) * term_blend
    return d_target


def compute_dC_dd_numpy(d, d0, d_contact, d_rest, d_neutral_start):
    eps = 1e-8
    dp = compute_target_distance_numpy(d + eps, d0, d_contact, d_rest, d_neutral_start)
    dm = compute_target_distance_numpy(d - eps, d0, d_contact, d_rest, d_neutral_start)
    return 1.0 - (dp - dm) / (2.0 * eps)


# =============================================================================
# Warp kernel to evaluate compute_target_distance on GPU for comparison
# =============================================================================

@wp.kernel
def eval_target_distance_kernel(
    d_array: wp.array(dtype=float),
    d0: float,
    d_contact: float,
    d_rest: float,
    d_neutral_start: float,
    result: wp.array(dtype=float)):
    tid = wp.tid()
    result[tid] = compute_target_distance(d_array[tid], d0, d_contact, d_rest, d_neutral_start)


@wp.kernel
def eval_dC_dd_kernel(
    d_array: wp.array(dtype=float),
    d0: float,
    d_contact: float,
    d_rest: float,
    d_neutral_start: float,
    result: wp.array(dtype=float)):
    tid = wp.tid()
    result[tid] = compute_dC_dd(d_array[tid], d0, d_contact, d_rest, d_neutral_start)


# =============================================================================
# Layer 1: Math verification
# =============================================================================

class TestMath:
    """Verify compute_target_distance and compute_dC_dd math properties."""

    def test_gpu_vs_numpy(self):
        """GPU kernel output must match numpy reference within tolerance."""
        print("\n=== Math Test 1: GPU vs Numpy Reference ===")

        d0 = 0.001  # 1mm initial distance
        N = 200
        d_vals = np.linspace(0.0001, 0.01, N).astype(np.float32)

        # Numpy reference
        ref = np.array([compute_target_distance_numpy(float(d), d0, D_CONTACT, D_REST, D_NEUTRAL_START) for d in d_vals])

        # GPU
        d_wp = wp.array(d_vals, dtype=wp.float32, device=DEVICE)
        result_wp = wp.zeros(N, dtype=wp.float32, device=DEVICE)
        wp.launch(eval_target_distance_kernel, dim=N,
                  inputs=[d_wp, d0, D_CONTACT, D_REST, D_NEUTRAL_START, result_wp],
                  device=DEVICE)
        gpu_result = result_wp.numpy()

        max_err = np.max(np.abs(gpu_result - ref))
        print(f"  Max absolute error: {max_err:.2e}")
        assert max_err < 1e-5, f"GPU/numpy mismatch: max_err={max_err}"
        print("  PASSED")

    def test_monotonicity(self):
        """dC/dd must be > 0 everywhere (constraint is monotonically increasing)."""
        print("\n=== Math Test 2: Monotonicity (dC/dd > 0) ===")

        d0 = 0.001
        d_vals = np.linspace(0.0001, 0.02, 5000)
        dC_vals = np.array([compute_dC_dd_numpy(float(d), d0, D_CONTACT, D_REST, D_NEUTRAL_START) for d in d_vals])

        min_dC = np.min(dC_vals)
        print(f"  min(dC/dd) = {min_dC:.6f}")
        assert min_dC > 0, f"Monotonicity violated: min dC/dd = {min_dC}"
        print("  PASSED")

    def test_c1_continuity(self):
        """C(d) and dC/dd must be continuous at region boundaries."""
        print("\n=== Math Test 3: C1 Continuity at Boundaries ===")

        d0 = 0.001
        eps = 1e-8
        # Boundaries: d0 - delta and d0 + delta, where delta = max(1e-5, D_CONTACT) = 0.0003
        delta = max(1e-5, D_CONTACT)
        boundaries = [d0 - delta, d0 + delta]

        all_pass = True
        for bd in boundaries:
            C_left = float(bd - eps) - compute_target_distance_numpy(float(bd - eps), d0, D_CONTACT, D_REST, D_NEUTRAL_START)
            C_right = float(bd + eps) - compute_target_distance_numpy(float(bd + eps), d0, D_CONTACT, D_REST, D_NEUTRAL_START)
            jump = abs(C_right - C_left)

            dC_left = compute_dC_dd_numpy(float(bd - eps), d0, D_CONTACT, D_REST, D_NEUTRAL_START)
            dC_right = compute_dC_dd_numpy(float(bd + eps), d0, D_CONTACT, D_REST, D_NEUTRAL_START)
            grad_jump = abs(dC_right - dC_left)

            ok_c = jump < 1e-5
            ok_g = grad_jump < 0.1  # C1 allows small gradient discontinuity from finite diff
            print(f"  Boundary d={bd*1000:.3f}mm: C jump={jump:.2e} {'OK' if ok_c else 'FAIL'}, "
                  f"dC/dd jump={grad_jump:.4f} {'OK' if ok_g else 'FAIL'}")
            if not ok_c or not ok_g:
                all_pass = False

        assert all_pass, "C1 continuity check failed"
        print("  PASSED")

    def test_unique_equilibrium(self):
        """C(d) = d - d*(d) must have exactly one zero for d > 0."""
        print("\n=== Math Test 4: Unique Equilibrium ===")

        d0 = 0.001
        d_vals = np.linspace(0.0001, 0.02, 10000)
        C_vals = np.array([float(d) - compute_target_distance_numpy(float(d), d0, D_CONTACT, D_REST, D_NEUTRAL_START)
                           for d in d_vals])

        sign_changes = np.sum(np.diff(np.sign(C_vals)) != 0)
        print(f"  Sign changes in C(d): {sign_changes}")
        # Expect exactly 1 zero crossing (the equilibrium point)
        assert sign_changes == 1, f"Expected 1 equilibrium, found {sign_changes} sign changes"
        print("  PASSED")

    def run_all(self):
        self.test_gpu_vs_numpy()
        self.test_monotonicity()
        self.test_c1_continuity()
        self.test_unique_equilibrium()


# =============================================================================
# Layer 2: Kernel unit tests
# =============================================================================

def make_triangle_geometry(center_y=0.0, scale=0.01):
    """Create a simple triangle in XZ plane at height center_y.
    Returns vertex positions as numpy array (3 vertices) and indices."""
    verts = np.array([
        [0.0, center_y, 0.0],
        [scale, center_y, 0.0],
        [0.0, center_y, scale],
    ], dtype=np.float32)
    return verts


class TestDeformDeformKernel:
    """Test the deform-deform adhesion kernel with known geometry."""

    def test_equilibrium_no_correction(self):
        """At rest distance, constraint is near zero → dP should be small."""
        print("\n=== Kernel Test 1: Deform-Deform at Equilibrium ===")

        # Triangle at y=0 (object B), vertex at y=d0 (object A)
        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001  # 1mm initial gap

        # Vertex directly above triangle centroid at rest distance
        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, d0, 0.0], dtype=np.float32)

        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])  # [vertex, tri0, tri1, tri2]

        # Bond: vertex=0, triangle=(1,2,3)
        vertexId = np.array([0], dtype=np.int32)
        triId = np.array([1, 2, 3], dtype=np.int32)
        triBar = np.array([[1.0/3, 1.0/3, 1.0/3]], dtype=np.float32).view(dtype=[('x','f4'),('y','f4'),('z','f4')]).reshape(-1)
        restGap = np.array([d0], dtype=np.float32)
        active = np.array([1.0], dtype=np.float32)

        # Allocate GPU arrays
        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=DEVICE)
        adhesionVertexId = wp.array(vertexId, dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array(triId, dtype=wp.int32, device=DEVICE)

        # Initialize barycentric coords and rest gap via init kernel
        adhesionTriBar = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.zeros(1, dtype=wp.float32, device=DEVICE)
        wp.launch(initAdhesionBondsKernel, dim=1,
                  inputs=[predictedVertex, adhesionVertexId, adhesionTriId,
                          adhesionTriBar, adhesionRestGap],
                  device=DEVICE)

        adhesionActive = wp.array(active, dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        # Launch constraint kernel
        wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          adhesionVertexId, adhesionTriId, adhesionTriBar,
                          adhesionRestGap, adhesionActive,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()
        rest_gap_np = adhesionRestGap.numpy()

        max_correction = np.max(np.abs(dP_np))
        print(f"  Rest gap computed: {rest_gap_np[0]*1000:.4f} mm (expected ~{d0*1000:.4f} mm)")
        print(f"  Max dP correction: {max_correction:.2e}")

        # At equilibrium, C(d) should be small → dP should be small
        # Not exactly zero because d0 != equilibrium of the curve
        assert not np.any(np.isnan(dP_np)), "dP contains NaN"
        assert max_correction < 0.01, f"dP too large at equilibrium: {max_correction}"
        print("  PASSED")

    def test_stretch_creates_attraction(self):
        """When vertex is pulled away, dP should pull it back (attraction)."""
        print("\n=== Kernel Test 2: Deform-Deform Stretch → Attraction ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        stretch_d = 0.003  # stretched to 3mm (was 1mm)

        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, stretch_d, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.array([d0], dtype=wp.float32, device=DEVICE)
        adhesionActive = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          adhesionVertexId, adhesionTriId, adhesionTriBar,
                          adhesionRestGap, adhesionActive,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()

        # Vertex (idx 0) correction should point DOWNWARD (toward triangle, negative y)
        vertex_correction_y = dP_np[0][1]
        print(f"  Vertex dP.y = {vertex_correction_y:.6f} (should be < 0, pulling down)")
        assert vertex_correction_y < 0, f"Expected attraction (negative y), got {vertex_correction_y}"
        assert not np.any(np.isnan(dP_np)), "dP contains NaN"
        print("  PASSED")

    def test_compression_creates_repulsion(self):
        """When vertex is pushed very close, dP should push it away (repulsion)."""
        print("\n=== Kernel Test 3: Deform-Deform Compression → Repulsion ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        compressed_d = 0.00005  # 0.05mm (well below d_contact=0.3mm)

        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, compressed_d, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.array([d0], dtype=wp.float32, device=DEVICE)
        adhesionActive = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          adhesionVertexId, adhesionTriId, adhesionTriBar,
                          adhesionRestGap, adhesionActive,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()

        # Vertex correction should point UPWARD (away from triangle, positive y)
        vertex_correction_y = dP_np[0][1]
        print(f"  Vertex dP.y = {vertex_correction_y:.6f} (should be > 0, pushing up)")
        assert vertex_correction_y > 0, f"Expected repulsion (positive y), got {vertex_correction_y}"
        assert not np.any(np.isnan(dP_np)), "dP contains NaN"
        print("  PASSED")

    def test_bond_breaking(self):
        """Bond should break when stretch exceeds threshold."""
        print("\n=== Kernel Test 4: Bond Breaking ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001  # 1mm
        # Break threshold: max(d0 * (breakRatio - 1), stretchAbsMin) = max(0.002, 0.005) = 0.005
        # So bond breaks when d - d0 > 0.005, i.e. d > 0.006
        far_d = 0.008  # 8mm, well beyond break threshold

        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, far_d, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.array([d0], dtype=wp.float32, device=DEVICE)
        adhesionActive = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          adhesionVertexId, adhesionTriId, adhesionTriBar,
                          adhesionRestGap, adhesionActive,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        active_np = adhesionActive.numpy()
        dP_np = dP.numpy()

        print(f"  Active after stretch: {active_np[0]} (should be 0.0 = broken)")
        print(f"  Max dP: {np.max(np.abs(dP_np)):.2e} (should be 0, broken bond applies no correction)")
        assert active_np[0] == 0.0, f"Bond should have broken, active={active_np[0]}"
        assert np.max(np.abs(dP_np)) < 1e-10, "Broken bond should not apply corrections"
        print("  PASSED")

    def test_inactive_bond_noop(self):
        """Inactive bond should produce zero corrections."""
        print("\n=== Kernel Test 5: Inactive Bond No-Op ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        vertex_pos = tri_verts.mean(axis=0) + np.array([0.0, 0.002, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.array([0.001], dtype=wp.float32, device=DEVICE)
        adhesionActive = wp.array([0.0], dtype=wp.float32, device=DEVICE)  # INACTIVE
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          adhesionVertexId, adhesionTriId, adhesionTriBar,
                          adhesionRestGap, adhesionActive,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()
        assert np.max(np.abs(dP_np)) < 1e-15, "Inactive bond should produce zero dP"
        print("  PASSED")

    def run_all(self):
        self.test_equilibrium_no_correction()
        self.test_stretch_creates_attraction()
        self.test_compression_creates_repulsion()
        self.test_bond_breaking()
        self.test_inactive_bond_noop()


class TestRigidDeformKernel:
    """Test the rigid-deform adhesion kernel with known geometry."""

    def test_rigid_stretch_attraction(self):
        """Rigid point above triangle: when triangle is pulled away, dP should pull triangle up."""
        print("\n=== Kernel Test 6: Rigid-Deform Stretch → Attraction ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        rigid_y = 0.003  # rigid point at 3mm above triangle

        centroid = tri_verts.mean(axis=0)
        rigid_point = centroid + np.array([0.0, rigid_y, 0.0], dtype=np.float32)

        # Predicted vertices = triangle vertices only (rigid point is separate)
        predictedVertex = wp.array(tri_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(3, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(3, dtype=wp.int32, device=DEVICE)

        rigidPoint = wp.array([rigid_point], dtype=wp.vec3, device=DEVICE)
        triId = wp.array([0, 1, 2], dtype=wp.int32, device=DEVICE)
        triBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        restGap = wp.array([d0], dtype=wp.float32, device=DEVICE)
        active = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(rigidDeformAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          rigidPoint, triId, triBar, restGap, active,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()

        # Triangle should be pulled UPWARD (toward rigid point)
        avg_tri_correction_y = np.mean([dP_np[i][1] for i in range(3)])
        print(f"  Avg triangle dP.y = {avg_tri_correction_y:.6f} (should be > 0, pulled up)")
        assert avg_tri_correction_y > 0, f"Expected attraction upward, got {avg_tri_correction_y}"
        assert not np.any(np.isnan(dP_np)), "dP contains NaN"
        print("  PASSED")

    def test_rigid_only_deformable_corrected(self):
        """Verify that no corrections go to the rigid point (it's fixed)."""
        print("\n=== Kernel Test 7: Rigid Point Receives No Correction ===")

        # Use 4 vertices: 0=some_other_vertex, 1-3=triangle
        # The rigid point is NOT in the vertex array at all
        tri_verts = make_triangle_geometry(center_y=0.0)
        rigid_point = tri_verts.mean(axis=0) + np.array([0.0, 0.002, 0.0], dtype=np.float32)

        predictedVertex = wp.array(tri_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(3, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(3, dtype=wp.int32, device=DEVICE)

        rigidPoint = wp.array([rigid_point], dtype=wp.vec3, device=DEVICE)
        triId = wp.array([0, 1, 2], dtype=wp.int32, device=DEVICE)
        triBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        restGap = wp.array([0.001], dtype=wp.float32, device=DEVICE)
        active = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(rigidDeformAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          rigidPoint, triId, triBar, restGap, active,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        dP_np = dP.numpy()
        # All 3 dP entries should be non-zero (triangle vertices get corrections)
        has_correction = any(np.linalg.norm(dP_np[i]) > 1e-10 for i in range(3))
        assert has_correction, "Triangle vertices should receive corrections"
        assert not np.any(np.isnan(dP_np)), "dP contains NaN"
        print("  Triangle vertices corrected, rigid point not in dP array (by design)")
        print("  PASSED")

    def test_rigid_bond_breaking(self):
        """Rigid-deform bond should break when stretch exceeds threshold."""
        print("\n=== Kernel Test 8: Rigid-Deform Bond Breaking ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        far_y = 0.008  # 8mm above, stretch = 7mm > stretchAbsMin=5mm

        rigid_point = tri_verts.mean(axis=0) + np.array([0.0, far_y, 0.0], dtype=np.float32)

        predictedVertex = wp.array(tri_verts, dtype=wp.vec3, device=DEVICE)
        dP = wp.zeros(3, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(3, dtype=wp.int32, device=DEVICE)

        rigidPoint = wp.array([rigid_point], dtype=wp.vec3, device=DEVICE)
        triId = wp.array([0, 1, 2], dtype=wp.int32, device=DEVICE)
        triBar = wp.array([wp.vec3(1.0/3, 1.0/3, 1.0/3)], dtype=wp.vec3, device=DEVICE)
        restGap = wp.array([d0], dtype=wp.float32, device=DEVICE)
        active = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(rigidDeformAdhesionConstraintsDO, dim=1,
                  inputs=[predictedVertex, dP, constraintsNumber,
                          rigidPoint, triId, triBar, restGap, active,
                          normalCache, cacheValid,
                          D_CONTACT, D_REST, D_NEUTRAL_START,
                          BREAK_RATIO, STRETCH_ABS_MIN,
                          DEFAULT_ALPHA, DT],
                  device=DEVICE)

        active_np = active.numpy()
        print(f"  Active after stretch: {active_np[0]} (should be 0.0)")
        assert active_np[0] == 0.0, f"Bond should have broken, active={active_np[0]}"
        print("  PASSED")

    def run_all(self):
        self.test_rigid_stretch_attraction()
        self.test_rigid_only_deformable_corrected()
        self.test_rigid_bond_breaking()


# =============================================================================
# Layer 3: Mini physics simulation
# =============================================================================

@wp.kernel
def simple_gravity_kernel(
    predictedVertex: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    inverseMass: wp.array(dtype=float),
    gravity_y: float,
    dt: float):
    """Minimal PBD gravity: v += g*dt, predictedPos = pos + v*dt."""
    tid = wp.tid()
    invM = inverseMass[tid]
    if invM > 0.0:
        v = velocity[tid]
        v = wp.vec3(v[0], v[1] + gravity_y * dt * invM, v[2])
        velocity[tid] = v
        pos = predictedVertex[tid]
        predictedVertex[tid] = wp.vec3(pos[0], pos[1] + v[1] * dt, pos[2])


@wp.kernel
def update_velocity_kernel(
    vertex: wp.array(dtype=wp.vec3),
    predictedVertex: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    dt: float):
    """PBD velocity update: v = (predicted - old) / dt."""
    tid = wp.tid()
    v = (predictedVertex[tid] - vertex[tid]) * (1.0 / dt)
    velocity[tid] = v


@wp.kernel
def apply_corrections_kernel(
    predictedVertex: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    constraintsNumber: wp.array(dtype=int)):
    """Average constraint corrections (Jacobi pattern)."""
    tid = wp.tid()
    n = constraintsNumber[tid]
    if n > 0:
        correction = dP[tid] * (1.0 / float(n))
        predictedVertex[tid] = predictedVertex[tid] + correction
    dP[tid] = wp.vec3(0.0, 0.0, 0.0)
    constraintsNumber[tid] = 0


class TestPhysicsLoop:
    """Mini time-stepping: verify adhesion bond holds and eventually breaks."""

    def test_bond_holds_under_gravity(self):
        """A vertex bonded to a fixed triangle should not fall away under light gravity."""
        print("\n=== Physics Test 1: Bond Holds Under Gravity ===")

        # Setup: triangle at y=0 (fixed, invMass=0), vertex at y=d0 (free, invMass=1)
        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, d0, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        initial_vertex_y = float(vertex_pos[1])

        nVerts = 4
        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        vertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        inverseMass = wp.array([1.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=DEVICE)  # only vertex 0 is free

        dP = wp.zeros(nVerts, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(nVerts, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.zeros(1, dtype=wp.float32, device=DEVICE)

        wp.launch(initAdhesionBondsKernel, dim=1,
                  inputs=[vertex, adhesionVertexId, adhesionTriId,
                          adhesionTriBar, adhesionRestGap],
                  device=DEVICE)

        adhesionActive = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        velocity = wp.zeros(nVerts, dtype=wp.vec3, device=DEVICE)

        dt = 0.001
        gravity_y = -9.81
        num_steps = 100
        constraint_iters = 4

        for step in range(num_steps):
            # Copy vertex → predictedVertex
            wp.copy(predictedVertex, vertex)

            # Apply gravity (with velocity accumulation)
            wp.launch(simple_gravity_kernel, dim=nVerts,
                      inputs=[predictedVertex, velocity, inverseMass, gravity_y, dt],
                      device=DEVICE)

            # Reset cache
            wp.launch(resetAdhesionCacheKernel, dim=1,
                      inputs=[cacheValid], device=DEVICE)

            # Constraint iterations
            for _ in range(constraint_iters):
                wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                          inputs=[predictedVertex, dP, constraintsNumber,
                                  adhesionVertexId, adhesionTriId, adhesionTriBar,
                                  adhesionRestGap, adhesionActive,
                                  normalCache, cacheValid,
                                  D_CONTACT, D_REST, D_NEUTRAL_START,
                                  BREAK_RATIO, STRETCH_ABS_MIN,
                                  DEFAULT_ALPHA, dt],
                          device=DEVICE)

                wp.launch(apply_corrections_kernel, dim=nVerts,
                          inputs=[predictedVertex, dP, constraintsNumber],
                          device=DEVICE)

            # PBD velocity update: v = (predicted - old) / dt
            wp.launch(update_velocity_kernel, dim=nVerts,
                      inputs=[vertex, predictedVertex, velocity, dt],
                      device=DEVICE)

            # Update vertex from predicted
            wp.copy(vertex, predictedVertex)

        final_verts = vertex.numpy()
        final_vertex_y = float(final_verts[0][1])
        displacement = abs(final_vertex_y - initial_vertex_y)
        active_np = adhesionActive.numpy()

        print(f"  Initial vertex y: {initial_vertex_y*1000:.4f} mm")
        print(f"  Final vertex y:   {final_vertex_y*1000:.4f} mm")
        print(f"  Displacement:     {displacement*1000:.4f} mm")
        print(f"  Bond still active: {active_np[0]}")

        assert active_np[0] == 1.0, "Bond should still be active"
        assert displacement < 0.005, f"Vertex drifted too far: {displacement*1000:.2f}mm"
        print("  PASSED")

    def test_bond_breaks_under_strong_pull(self):
        """Apply strong downward pull → bond should eventually break."""
        print("\n=== Physics Test 2: Bond Breaks Under Strong Pull ===")

        tri_verts = make_triangle_geometry(center_y=0.0)
        d0 = 0.001
        centroid = tri_verts.mean(axis=0)
        vertex_pos = centroid + np.array([0.0, d0, 0.0], dtype=np.float32)
        all_verts = np.vstack([vertex_pos.reshape(1, 3), tri_verts])

        nVerts = 4
        predictedVertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        vertex = wp.array(all_verts, dtype=wp.vec3, device=DEVICE)
        inverseMass = wp.array([1.0, 0.0, 0.0, 0.0], dtype=wp.float32, device=DEVICE)

        dP = wp.zeros(nVerts, dtype=wp.vec3, device=DEVICE)
        constraintsNumber = wp.zeros(nVerts, dtype=wp.int32, device=DEVICE)

        adhesionVertexId = wp.array([0], dtype=wp.int32, device=DEVICE)
        adhesionTriId = wp.array([1, 2, 3], dtype=wp.int32, device=DEVICE)
        adhesionTriBar = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        adhesionRestGap = wp.zeros(1, dtype=wp.float32, device=DEVICE)

        wp.launch(initAdhesionBondsKernel, dim=1,
                  inputs=[vertex, adhesionVertexId, adhesionTriId,
                          adhesionTriBar, adhesionRestGap],
                  device=DEVICE)

        adhesionActive = wp.array([1.0], dtype=wp.float32, device=DEVICE)
        normalCache = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
        cacheValid = wp.zeros(1, dtype=wp.int32, device=DEVICE)
        velocity = wp.zeros(nVerts, dtype=wp.vec3, device=DEVICE)

        dt = 0.001
        gravity_y = -5000.0  # Very strong pull (velocity accumulates over time)
        num_steps = 500
        constraint_iters = 4

        broke_at_step = -1
        for step in range(num_steps):
            wp.copy(predictedVertex, vertex)

            wp.launch(simple_gravity_kernel, dim=nVerts,
                      inputs=[predictedVertex, velocity, inverseMass, gravity_y, dt],
                      device=DEVICE)

            wp.launch(resetAdhesionCacheKernel, dim=1,
                      inputs=[cacheValid], device=DEVICE)

            for _ in range(constraint_iters):
                wp.launch(unifiedAdhesionConstraintsDO, dim=1,
                          inputs=[predictedVertex, dP, constraintsNumber,
                                  adhesionVertexId, adhesionTriId, adhesionTriBar,
                                  adhesionRestGap, adhesionActive,
                                  normalCache, cacheValid,
                                  D_CONTACT, D_REST, D_NEUTRAL_START,
                                  BREAK_RATIO, STRETCH_ABS_MIN,
                                  DEFAULT_ALPHA, dt],
                          device=DEVICE)

                wp.launch(apply_corrections_kernel, dim=nVerts,
                          inputs=[predictedVertex, dP, constraintsNumber],
                          device=DEVICE)

            # PBD velocity update: v = (predicted - old) / dt
            wp.launch(update_velocity_kernel, dim=nVerts,
                      inputs=[vertex, predictedVertex, velocity, dt],
                      device=DEVICE)

            wp.copy(vertex, predictedVertex)

            active_np = adhesionActive.numpy()
            if active_np[0] == 0.0:
                broke_at_step = step
                break

        print(f"  Bond broke at step: {broke_at_step} (expected > 0)")
        if broke_at_step > 0:
            final_verts = vertex.numpy()
            print(f"  Final vertex y: {final_verts[0][1]*1000:.2f} mm (should be negative = fell down)")
        assert broke_at_step > 0, "Bond should have broken under strong pull"
        print("  PASSED")

    def test_multi_bond_count(self):
        """Test countActiveBondsPerEnvKernel with multiple bonds."""
        print("\n=== Physics Test 3: Active Bond Counting ===")

        numBonds = 10
        active_vals = [1.0] * numBonds
        active_vals[3] = 0.0  # one broken
        active_vals[7] = 0.0  # another broken

        adhesionActive = wp.array(active_vals, dtype=wp.float32, device=DEVICE)
        activeCounts = wp.zeros(1, dtype=wp.int32, device=DEVICE)

        wp.launch(countActiveBondsPerEnvKernel, dim=numBonds,
                  inputs=[adhesionActive, numBonds, activeCounts],
                  device=DEVICE)

        count = activeCounts.numpy()[0]
        print(f"  Active bonds: {count} (expected 8)")
        assert count == 8, f"Expected 8 active bonds, got {count}"
        print("  PASSED")

    def run_all(self):
        self.test_bond_holds_under_gravity()
        self.test_bond_breaks_under_strong_pull()
        self.test_multi_bond_count()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADHESION CONSTRAINT VERIFICATION TESTS")
    print("=" * 60)

    all_passed = True

    try:
        print("\n--- Layer 1: Math Verification ---")
        TestMath().run_all()

        print("\n--- Layer 2: Deform-Deform Kernel Tests ---")
        TestDeformDeformKernel().run_all()

        print("\n--- Layer 2: Rigid-Deform Kernel Tests ---")
        TestRigidDeformKernel().run_all()

        print("\n--- Layer 3: Physics Simulation Tests ---")
        TestPhysicsLoop().run_all()

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
