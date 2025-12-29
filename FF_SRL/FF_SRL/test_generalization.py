"""
Test SAC policy generalization with different initial positions.

This script tests whether the trained policy can still reach the target
when starting from positions different from the training distribution.

Usage:
    python test_generalization.py --checkpoint ./output/sac_20251228_112720/policy_best.pth
    python test_generalization.py --checkpoint policy_best.pth --variations 10
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

from train_sac import Actor
from rl_env import FFSRLEnv


class GeneralizationTester:
    """Test policy generalization across different initial conditions."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Create base environment
        print("Creating test environment...")
        self.env = FFSRLEnv(
            device=device,
            max_episode_steps=128,
            action_strength=0.4,
            render=False
        )
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Load policy
        print(f"Loading policy from: {checkpoint_path}")
        self.actor = Actor(obs_dim, action_dim, hidden_dim=256).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
        else:
            self.actor.load_state_dict(checkpoint)
        
        self.actor.eval()
        print("✓ Policy loaded!")
        
        # Store original initial position
        self.original_init_pos = np.array([0.0, 10.0, 4.0])
        self.target_pos = np.array([0.31, 10.11, 1.61])
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get deterministic action from policy."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actor.get_action(obs_tensor, deterministic=True)
            return action.squeeze(0).cpu().numpy()
    
    def test_position(self, init_pos: np.ndarray, verbose: bool = False) -> dict:
        """
        Test policy with a specific initial position.
        
        Args:
            init_pos: Initial tool position [x, y, z]
            verbose: Print step-by-step info
        
        Returns:
            dict: Test results
        """
        # Calculate delta from original position
        delta = init_pos - self.original_init_pos
        
        # Convert to flat float32 array for cartesian action
        cartesian_action = wp.array(delta.astype(np.float32), dtype=wp.float32, device=self.env.device)
        
        # Apply position change and update initial state
        self.env.simModel.setEffectorInitialPosition(cartesian_action)
        
        obs, info = self.env.reset()
        
        initial_distance = np.linalg.norm(init_pos - self.target_pos)
        
        if verbose:
            print(f"\nInitial position: [{init_pos[0]:.3f}, {init_pos[1]:.3f}, {init_pos[2]:.3f}]")
            print(f"Target position:  [{self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, {self.target_pos[2]:.3f}]")
            print(f"Initial distance: {initial_distance:.4f} units ({initial_distance*10:.2f}mm)")
        
        episode_reward = 0.0
        step = 0
        done = False
        success = False
        distances = [initial_distance]
        
        while not done and step < self.env.max_episode_steps:
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            current_distance = np.linalg.norm(next_obs[:3] - self.target_pos)
            distances.append(current_distance)
            
            if terminated:
                success = True
                break
            
            obs = next_obs
        
        final_distance = distances[-1]
        
        return {
            'init_pos': init_pos.copy(),
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'min_distance': min(distances),
            'success': success,
            'steps': step,
            'reward': episode_reward,
            'distances': distances
        }
    
    def test_grid_variations(self, delta: float = 0.2, num_samples: int = 5):
        """
        Test policy on a grid of positions around the original.
        
        Args:
            delta: Maximum deviation from original position (units)
            num_samples: Number of samples per dimension
        """
        print("\n" + "="*70)
        print("GRID VARIATION TEST")
        print("="*70)
        print(f"Original position: {self.original_init_pos}")
        print(f"Variation range: ±{delta} units (±{delta*10:.1f}mm)")
        print(f"Samples per dim: {num_samples}")
        print(f"Total tests: {num_samples**3}")
        
        # Generate grid
        x_range = np.linspace(self.original_init_pos[0] - delta, 
                             self.original_init_pos[0] + delta, num_samples)
        y_range = np.linspace(self.original_init_pos[1] - delta, 
                             self.original_init_pos[1] + delta, num_samples)
        z_range = np.linspace(self.original_init_pos[2] - delta, 
                             self.original_init_pos[2] + delta, num_samples)
        
        results = []
        total_tests = num_samples ** 3
        
        print("\nRunning tests...")
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                for k, z in enumerate(z_range):
                    init_pos = np.array([x, y, z])
                    result = self.test_position(init_pos, verbose=False)
                    results.append(result)
                    
                    # Progress
                    progress = len(results) / total_tests * 100
                    if len(results) % 10 == 0:
                        print(f"  Progress: {progress:.1f}% ({len(results)}/{total_tests})")
        
        self._print_results(results)
        return results
    
    def test_random_variations(self, delta: float = 0.3, num_tests: int = 50):
        """
        Test policy on random positions around the original.
        
        Args:
            delta: Maximum deviation from original position (units)
            num_tests: Number of random tests
        """
        print("\n" + "="*70)
        print("RANDOM VARIATION TEST")
        print("="*70)
        print(f"Original position: {self.original_init_pos}")
        print(f"Variation range: ±{delta} units (±{delta*10:.1f}mm)")
        print(f"Number of tests: {num_tests}")
        
        results = []
        
        print("\nRunning tests...")
        for i in range(num_tests):
            # Random offset
            offset = np.random.uniform(-delta, delta, size=3)
            init_pos = self.original_init_pos + offset
            
            result = self.test_position(init_pos, verbose=False)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {(i+1)/num_tests*100:.1f}% ({i+1}/{num_tests})")
        
        self._print_results(results)
        return results
    
    def test_specific_positions(self, positions: list):
        """Test specific positions provided by user."""
        print("\n" + "="*70)
        print("SPECIFIC POSITION TEST")
        print("="*70)
        print(f"Number of positions: {len(positions)}")
        
        results = []
        for i, pos in enumerate(positions):
            print(f"\nTest {i+1}/{len(positions)}:")
            result = self.test_position(np.array(pos), verbose=True)
            results.append(result)
            
            print(f"  Result: {'✓ SUCCESS' if result['success'] else '✗ FAILED'}")
            print(f"  Steps: {result['steps']}")
            print(f"  Final distance: {result['final_distance']*10:.2f}mm")
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: list):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        successes = sum(r['success'] for r in results)
        success_rate = successes / len(results) * 100
        
        print(f"\nOverall Performance:")
        print(f"  Total tests:      {len(results)}")
        print(f"  Successes:        {successes}")
        print(f"  Failures:         {len(results) - successes}")
        print(f"  Success rate:     {success_rate:.1f}%")
        
        # Statistics for successful trials
        if successes > 0:
            successful = [r for r in results if r['success']]
            print(f"\nSuccessful Trials:")
            print(f"  Mean steps:       {np.mean([r['steps'] for r in successful]):.1f}")
            print(f"  Mean reward:      {np.mean([r['reward'] for r in successful]):.2f}")
            print(f"  Mean final dist:  {np.mean([r['final_distance'] for r in successful])*10:.2f}mm")
        
        # Statistics for failed trials
        if successes < len(results):
            failed = [r for r in results if not r['success']]
            print(f"\nFailed Trials:")
            print(f"  Mean steps:       {np.mean([r['steps'] for r in failed]):.1f}")
            print(f"  Mean min dist:    {np.mean([r['min_distance'] for r in failed])*10:.2f}mm")
            print(f"  Best min dist:    {min([r['min_distance'] for r in failed])*10:.2f}mm")
        
        # Initial distance analysis
        print(f"\nInitial Distance Analysis:")
        print(f"  Mean init dist:   {np.mean([r['initial_distance'] for r in results])*10:.2f}mm")
        print(f"  Min init dist:    {min([r['initial_distance'] for r in results])*10:.2f}mm")
        print(f"  Max init dist:    {max([r['initial_distance'] for r in results])*10:.2f}mm")
        
        # Distance vs success correlation
        print(f"\nDistance vs Success:")
        for threshold in [15, 20, 25, 30, 35]:
            close = [r for r in results if r['initial_distance']*10 <= threshold]
            if close:
                close_success = sum(r['success'] for r in close) / len(close) * 100
                print(f"  ≤{threshold}mm: {close_success:.1f}% ({sum(r['success'] for r in close)}/{len(close)})")
    
    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Test SAC policy generalization")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to policy checkpoint")
    parser.add_argument("--mode", type=str, default="random",
                       choices=["grid", "random", "specific"],
                       help="Test mode")
    parser.add_argument("--delta", type=float, default=0.3,
                       help="Max deviation from original position (units)")
    parser.add_argument("--variations", type=int, default=50,
                       help="Number of test variations")
    parser.add_argument("--grid-samples", type=int, default=5,
                       help="Samples per dimension for grid mode")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print("\n" + "="*70)
    print("SAC POLICY GENERALIZATION TEST")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test mode:  {args.mode}")
    print(f"Device:     {args.device}")
    
    # Create tester
    tester = GeneralizationTester(
        checkpoint_path=str(checkpoint_path),
        device=args.device
    )
    
    try:
        # Run tests based on mode
        if args.mode == "grid":
            results = tester.test_grid_variations(
                delta=args.delta,
                num_samples=args.grid_samples
            )
        
        elif args.mode == "random":
            results = tester.test_random_variations(
                delta=args.delta,
                num_tests=args.variations
            )
        
        elif args.mode == "specific":
            # Example specific positions to test
            positions = [
                [0.0, 10.0, 4.0],    # Original
                [0.2, 10.0, 4.0],    # +X
                [-0.2, 10.0, 4.0],   # -X
                [0.0, 10.2, 4.0],    # +Y
                [0.0, 9.8, 4.0],     # -Y
                [0.0, 10.0, 4.3],    # +Z
                [0.0, 10.0, 3.7],    # -Z
                [0.3, 10.3, 4.3],    # All positive
                [-0.3, 9.7, 3.7],    # All negative
            ]
            results = tester.test_specific_positions(positions)
        
        # Save results
        save_path = checkpoint_path.parent / f"generalization_{args.mode}.npz"
        np.savez(
            save_path,
            results=[r for r in results],
            mode=args.mode,
            delta=args.delta,
            num_tests=len(results)
        )
        print(f"\n✓ Results saved to: {save_path}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    
    finally:
        tester.close()
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()
