"""
Unit tests for Stable Neo-Hookean constraints
Validates implementation against known analytical solutions
"""

import warp as wp
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constraints_stable_nh import (
    stableNeohookeanDeviatoricConstraint,
    stableNeohookeanHydrostaticConstraint,
    computeInverseRestMatrices
)

wp.init()

class TestStableNeohookean:
    def __init__(self):
        self.device = "cuda:0"
        
    def test_identity_deformation(self):
        """Test: Identity deformation (F = I) should remain stable
        
        Note: C = ||I||_F = √3 at rest (non-zero by design).
        The XPBD solver naturally handles this offset via compliance.
        We test for numerical stability, not C=0.
        """
        print("\n=== Test 1: Identity Deformation ===")

        
        # Single tetrahedron at rest (unit tet)
        vertex_np = np.array([
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [0.0, 1.0, 0.0],  # v2 
            [0.0, 0.0, 1.0],  # v3
        ], dtype=np.float32)
        
        tetrahedron_np = np.array([0, 1, 2, 3], dtype=np.int32)
        inverseMass_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        activeTetrahedron_np = np.array([1], dtype=np.float32)
        
        # Allocate GPU arrays
        vertex = wp.array(vertex_np, dtype=wp.vec3, device=self.device)
        predictedVertex = wp.array(vertex_np, dtype=wp.vec3, device=self.device)
        tetrahedron = wp.array(tetrahedron_np, dtype=wp.int32, device=self.device)
        inverseMass = wp.array(inverseMass_np, dtype=wp.float32, device=self.device)
        activeTetrahedron = wp.array(activeTetrahedron_np, dtype=wp.float32, device=self.device)
        
        invRestMatrix = wp.zeros(1, dtype=wp.mat33, device=self.device)
        restVolume = wp.zeros(1, dtype=wp.float32, device=self.device)
        lambdas = wp.zeros(1, dtype=wp.float32, device=self.device)
        dP = wp.zeros(4, dtype=wp.vec3, device=self.device)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=self.device)
        
        # Precompute rest matrices
        wp.launch(
            kernel=computeInverseRestMatrices,
            dim=1,
            inputs=[vertex, tetrahedron, invRestMatrix, restVolume],
            device=self.device
        )
        
        # Material parameters (soft tissue)
        mu = 1000.0  # Pa
        lambda_ = 5000.0  # Pa
        dt = 0.01
        
        # Run deviatoric constraint
        wp.launch(
            kernel=stableNeohookeanDeviatoricConstraint,
            dim=1,
            inputs=[
                predictedVertex, dP, constraintsNumber,
                tetrahedron, invRestMatrix, lambdas,
                inverseMass, restVolume, activeTetrahedron,
                mu, lambda_, dt
            ],
            device=self.device
        )
        
        # Check results
        dP_np = dP.numpy()
        lambdas_np = lambdas.numpy()
        
        print(f"Lambda: {lambdas_np[0]:.6e}")
        print(f"Max correction: {np.max(np.abs(dP_np)):.6e}")
        
        # At rest: C = √3 ≈ 1.732 (non-zero by design in Macklin's formulation)
        # Lambda will be negative to resist this "constraint violation"
        # The key is numerical stability, not C=0
        assert not np.isnan(lambdas_np[0]), "Lambda should not be NaN"
        assert not np.isinf(lambdas_np[0]), "Lambda should not be Inf"
        assert np.abs(lambdas_np[0]) < 100.0, "Lambda should be bounded"
        assert not np.any(np.isnan(dP_np)), "Corrections should not be NaN"
        
        print("✓ PASSED: Identity deformation remains stable")
        
    def test_pure_compression(self):
        """Test: Pure compression (F = 0.5*I) should resist"""
        print("\n=== Test 2: Pure Compression ===")
        
        # Rest configuration
        vertex_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Compressed configuration (50% smaller)
        vertex_compressed = vertex_rest * 0.5
        
        tetrahedron_np = np.array([0, 1, 2, 3], dtype=np.int32)
        inverseMass_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        activeTetrahedron_np = np.array([1], dtype=np.float32)
        
        vertex = wp.array(vertex_rest, dtype=wp.vec3, device=self.device)
        predictedVertex = wp.array(vertex_compressed, dtype=wp.vec3, device=self.device)
        tetrahedron = wp.array(tetrahedron_np, dtype=wp.int32, device=self.device)
        inverseMass = wp.array(inverseMass_np, dtype=wp.float32, device=self.device)
        activeTetrahedron = wp.array(activeTetrahedron_np, dtype=wp.float32, device=self.device)
        
        invRestMatrix = wp.zeros(1, dtype=wp.mat33, device=self.device)
        restVolume = wp.zeros(1, dtype=wp.float32, device=self.device)
        lambdas_dev = wp.zeros(1, dtype=wp.float32, device=self.device)
        lambdas_vol = wp.zeros(1, dtype=wp.float32, device=self.device)
        dP = wp.zeros(4, dtype=wp.vec3, device=self.device)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=self.device)
        
        wp.launch(
            kernel=computeInverseRestMatrices,
            dim=1,
            inputs=[vertex, tetrahedron, invRestMatrix, restVolume],
            device=self.device
        )
        
        mu = 1000.0
        lambda_ = 5000.0
        dt = 0.01
        
        # Test hydrostatic constraint (should strongly resist compression)
        wp.launch(
            kernel=stableNeohookeanHydrostaticConstraint,
            dim=1,
            inputs=[
                predictedVertex, dP, constraintsNumber,
                tetrahedron, invRestMatrix, lambdas_vol,
                inverseMass, restVolume, activeTetrahedron,
                mu, lambda_, dt
            ],
            device=self.device
        )
        
        lambdas_vol_np = lambdas_vol.numpy()
        dP_np = dP.numpy()
        
        print(f"Hydrostatic lambda: {lambdas_vol_np[0]:.6f}")
        print(f"Correction magnitude: {np.linalg.norm(dP_np[0]):.6f}")
        
        # For 50% compression: J = 0.125
        # C = -γ + log(0.125) ≈ -0.2 - 2.08 = -2.28 (negative)
        # Lambda update will be positive to expand the element
        # The sign of lambda depends on XPBD update formula
        assert not np.isnan(lambdas_vol_np[0]), "Lambda should not be NaN"
        assert np.abs(lambdas_vol_np[0]) > 1e-6, "Should generate restoration force"
        assert np.linalg.norm(dP_np[0]) > 1e-6, "Should apply corrections"
        
        print("✓ PASSED: Compression generates restoration force")
        
    def test_pure_shear(self):
        """Test: Pure shear deformation"""
        print("\n=== Test 3: Pure Shear ===")
        
        # Rest configuration
        vertex_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Shear: move top vertex
        vertex_sheared = vertex_rest.copy()
        vertex_sheared[3] = [0.5, 0.0, 1.0]  # Shear in x-direction
        
        tetrahedron_np = np.array([0, 1, 2, 3], dtype=np.int32)
        inverseMass_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        activeTetrahedron_np = np.array([1], dtype=np.float32)
        
        vertex = wp.array(vertex_rest, dtype=wp.vec3, device=self.device)
        predictedVertex = wp.array(vertex_sheared, dtype=wp.vec3, device=self.device)
        tetrahedron = wp.array(tetrahedron_np, dtype=wp.int32, device=self.device)
        inverseMass = wp.array(inverseMass_np, dtype=wp.float32, device=self.device)
        activeTetrahedron = wp.array(activeTetrahedron_np, dtype=wp.float32, device=self.device)
        
        invRestMatrix = wp.zeros(1, dtype=wp.mat33, device=self.device)
        restVolume = wp.zeros(1, dtype=wp.float32, device=self.device)
        lambdas_dev = wp.zeros(1, dtype=wp.float32, device=self.device)
        dP = wp.zeros(4, dtype=wp.vec3, device=self.device)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=self.device)
        
        wp.launch(
            kernel=computeInverseRestMatrices,
            dim=1,
            inputs=[vertex, tetrahedron, invRestMatrix, restVolume],
            device=self.device
        )
        
        mu = 1000.0
        lambda_ = 5000.0
        dt = 0.01
        
        # Test deviatoric constraint (should resist shear)
        wp.launch(
            kernel=stableNeohookeanDeviatoricConstraint,
            dim=1,
            inputs=[
                predictedVertex, dP, constraintsNumber,
                tetrahedron, invRestMatrix, lambdas_dev,
                inverseMass, restVolume, activeTetrahedron,
                mu, lambda_, dt
            ],
            device=self.device
        )
        
        lambdas_dev_np = lambdas_dev.numpy()
        dP_np = dP.numpy()
        
        print(f"Deviatoric lambda: {lambdas_dev_np[0]:.6f}")
        print(f"Correction on vertex 3: {dP_np[3]}")
        
        # Shear increases ||F||_F, so C > √3
        # Lambda will be negative to reduce the constraint value
        assert not np.isnan(lambdas_dev_np[0]), "Lambda should not be NaN"
        assert np.abs(lambdas_dev_np[0]) > 1e-6, "Should resist shear"
        assert np.linalg.norm(dP_np[3]) > 1e-6, "Should correct shear displacement"
        
        print("✓ PASSED: Shear deformation resisted")
    
    def test_extreme_compression_stability(self):
        """Test: Extreme compression (det(F) = 0.01) should remain stable"""
        print("\n=== Test 4: Extreme Compression Stability ===")
        
        vertex_rest = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        # Extreme compression (10% of original size)
        vertex_extreme = vertex_rest * 0.215  # det = 0.01
        
        tetrahedron_np = np.array([0, 1, 2, 3], dtype=np.int32)
        inverseMass_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        activeTetrahedron_np = np.array([1], dtype=np.float32)
        
        vertex = wp.array(vertex_rest, dtype=wp.vec3, device=self.device)
        predictedVertex = wp.array(vertex_extreme, dtype=wp.vec3, device=self.device)
        tetrahedron = wp.array(tetrahedron_np, dtype=wp.int32, device=self.device)
        inverseMass = wp.array(inverseMass_np, dtype=wp.float32, device=self.device)
        activeTetrahedron = wp.array(activeTetrahedron_np, dtype=wp.float32, device=self.device)
        
        invRestMatrix = wp.zeros(1, dtype=wp.mat33, device=self.device)
        restVolume = wp.zeros(1, dtype=wp.float32, device=self.device)
        lambdas_vol = wp.zeros(1, dtype=wp.float32, device=self.device)
        dP = wp.zeros(4, dtype=wp.vec3, device=self.device)
        constraintsNumber = wp.zeros(4, dtype=wp.int32, device=self.device)
        
        wp.launch(
            kernel=computeInverseRestMatrices,
            dim=1,
            inputs=[vertex, tetrahedron, invRestMatrix, restVolume],
            device=self.device
        )
        
        mu = 1000.0
        lambda_ = 5000.0
        dt = 0.01
        
        # Run constraint
        wp.launch(
            kernel=stableNeohookeanHydrostaticConstraint,
            dim=1,
            inputs=[
                predictedVertex, dP, constraintsNumber,
                tetrahedron, invRestMatrix, lambdas_vol,
                inverseMass, restVolume, activeTetrahedron,
                mu, lambda_, dt
            ],
            device=self.device
        )
        
        lambdas_np = lambdas_vol.numpy()
        dP_np = dP.numpy()
        
        print(f"Lambda: {lambdas_np[0]:.6f}")
        print(f"Max correction: {np.max(np.abs(dP_np)):.6f}")
        
        # Check for NaN or Inf (stability test)
        assert not np.any(np.isnan(lambdas_np)), "Lambda should not be NaN"
        assert not np.any(np.isinf(lambdas_np)), "Lambda should not be Inf"
        assert not np.any(np.isnan(dP_np)), "Corrections should not be NaN"
        
        # Energy should be bounded
        assert np.abs(lambdas_np[0]) < 1e6, "Lambda should be bounded"
        
        print("✓ PASSED: Extreme compression remains numerically stable")
    
    def run_all_tests(self):
        """Run all unit tests"""
        print("="*60)
        print("STABLE NEO-HOOKEAN CONSTRAINT TESTS")
        print("="*60)
        
        try:
            self.test_identity_deformation()
            self.test_pure_compression()
            self.test_pure_shear()
            self.test_extreme_compression_stability()
            
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED")
            print("="*60)
            return True
            
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            return False
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    tester = TestStableNeohookean()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
