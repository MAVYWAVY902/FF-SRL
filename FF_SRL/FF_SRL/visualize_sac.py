"""
Visualize SAC policy execution with interactive rendering.

This script loads a trained SAC policy and visualizes:
- Tool movement trajectory
- Tissue deformation
- Target reaching behavior
- Action sequences

Usage:
    python visualize_sac.py --checkpoint ./output/sac_20251228_112720/policy_best.pth
    python visualize_sac.py --checkpoint policy_best.pth --episodes 5 --speed 1.0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from train_sac import Actor
from rl_env import FFSRLEnv


class SAC_Visualizer:
    """Visualizer for trained SAC policies."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initialize visualizer.
        
        Args:
            checkpoint_path: Path to trained policy checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Create environment with rendering
        print("Creating environment with rendering enabled...")
        self.env = FFSRLEnv(
            device=device,
            max_episode_steps=128,
            action_strength=0.4,
            render=True  # Enable rendering
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
        print(f"✓ Policy loaded successfully!")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {action_dim}")
    
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from policy."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action = self.actor.get_action(obs_tensor, deterministic=deterministic)
            return action.squeeze(0).cpu().numpy()
    
    def visualize_episode(
        self, 
        episode_num: int = 1,
        speed: float = 1.0,
        deterministic: bool = True,
        save_trajectory: bool = True
    ):
        """
        Visualize a single episode.
        
        Args:
            episode_num: Episode number for logging
            speed: Playback speed (1.0 = normal, 2.0 = 2x faster)
            deterministic: Use deterministic policy
            save_trajectory: Save trajectory data
        
        Returns:
            dict: Episode statistics
        """
        print(f"\n{'='*70}")
        print(f"Episode {episode_num}")
        print(f"{'='*70}")
        
        obs, info = self.env.reset()
        
        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        trajectory = {
            'observations': [obs.copy()],
            'actions': [],
            'rewards': [],
            'tool_positions': [],
            'distances': []
        }
        
        # Initial info
        initial_distance = info.get('distance', np.linalg.norm(obs[:3] - obs[3:6]))
        print(f"\nInitial State:")
        print(f"  Tool position:   {obs[:3]}")
        print(f"  Target position: {obs[3:6]}")
        print(f"  Distance:        {initial_distance:.4f} units ({initial_distance*10:.2f}mm)")
        
        # Render initial state
        self.env.render()
        time.sleep(1.0 / speed)  # Pause to see initial state
        
        step = 0
        done = False
        success = False
        
        print(f"\n{'Step':<6} {'Action':<30} {'Distance':<12} {'Reward':<10}")
        print("-" * 70)
        
        while not done and step < self.env.max_episode_steps:
            # Get action from policy
            action = self.get_action(obs, deterministic=deterministic)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            current_distance = info.get('distance', np.linalg.norm(next_obs[:3] - next_obs[3:6]))
            
            # Store trajectory
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
            trajectory['tool_positions'].append(next_obs[:3].copy())
            trajectory['distances'].append(current_distance)
            trajectory['observations'].append(next_obs.copy())
            
            # Print step info
            action_str = f"[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}]"
            print(f"{step+1:<6} {action_str:<30} {current_distance:.4f} ({current_distance*10:.2f}mm)  {reward:>8.2f}")
            
            # Render
            self.env.render()
            time.sleep(0.1 / speed)  # Control playback speed
            
            obs = next_obs
            step += 1
            
            if terminated:
                success = True
                break
        
        # Final statistics
        print("\n" + "="*70)
        print("Episode Summary:")
        print("="*70)
        print(f"  Success:         {'✓' if success else '✗'}")
        print(f"  Total Steps:     {episode_length}")
        print(f"  Total Reward:    {episode_reward:.2f}")
        print(f"  Initial Dist:    {initial_distance:.4f} units ({initial_distance*10:.2f}mm)")
        print(f"  Final Dist:      {current_distance:.4f} units ({current_distance*10:.2f}mm)")
        print(f"  Min Dist:        {min(trajectory['distances']):.4f} units")
        
        # Action statistics
        actions = np.array(trajectory['actions'])
        print(f"\nAction Statistics:")
        print(f"  Mean |action|:   {np.linalg.norm(actions, axis=1).mean():.4f}")
        print(f"  Max |action_i|:  {np.abs(actions).max():.4f}")
        print(f"  Action range:    [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Save trajectory
        if save_trajectory:
            save_path = Path(self.checkpoint_path).parent / f"trajectory_episode_{episode_num}.npz"
            np.savez(
                save_path,
                observations=np.array(trajectory['observations']),
                actions=np.array(trajectory['actions']),
                rewards=np.array(trajectory['rewards']),
                tool_positions=np.array(trajectory['tool_positions']),
                distances=np.array(trajectory['distances']),
                success=success,
                episode_reward=episode_reward,
                episode_length=episode_length
            )
            print(f"\n✓ Trajectory saved to: {save_path}")
        
        return {
            'success': success,
            'reward': episode_reward,
            'length': episode_length,
            'initial_distance': initial_distance,
            'final_distance': current_distance,
            'min_distance': min(trajectory['distances']),
            'trajectory': trajectory
        }
    
    def visualize_multiple_episodes(
        self,
        num_episodes: int = 3,
        speed: float = 1.0,
        deterministic: bool = True
    ):
        """Visualize multiple episodes."""
        print("\n" + "="*70)
        print(f"Visualizing {num_episodes} episodes")
        print("="*70)
        
        results = []
        for i in range(num_episodes):
            result = self.visualize_episode(
                episode_num=i+1,
                speed=speed,
                deterministic=deterministic,
                save_trajectory=True
            )
            results.append(result)
            
            if i < num_episodes - 1:
                print(f"\nPress Enter to continue to next episode...")
                input()
        
        # Summary statistics
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        successes = sum(r['success'] for r in results)
        print(f"Success Rate:    {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
        print(f"Mean Reward:     {np.mean([r['reward'] for r in results]):.2f}")
        print(f"Mean Length:     {np.mean([r['length'] for r in results]):.1f} steps")
        print(f"Mean Final Dist: {np.mean([r['final_distance'] for r in results]):.4f} units")
        
    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize trained SAC policy")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy checkpoint (.pth file)")
    
    # Visualization
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to visualize")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0=normal, 2.0=2x faster)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save trajectory data")
    
    # System
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print("\n" + "="*70)
    print("SAC Policy Visualization")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {args.episodes}")
    print(f"Speed:      {args.speed}x")
    print(f"Policy:     {'Stochastic' if args.stochastic else 'Deterministic'}")
    print(f"Device:     {args.device}")
    print("="*70)
    
    # Create visualizer
    visualizer = SAC_Visualizer(
        checkpoint_path=str(checkpoint_path),
        device=args.device
    )
    
    try:
        # Visualize episodes
        if args.episodes == 1:
            visualizer.visualize_episode(
                episode_num=1,
                speed=args.speed,
                deterministic=not args.stochastic,
                save_trajectory=not args.no_save
            )
        else:
            visualizer.visualize_multiple_episodes(
                num_episodes=args.episodes,
                speed=args.speed,
                deterministic=not args.stochastic
            )
        
        print("\n✓ Visualization complete!")
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    
    finally:
        visualizer.close()
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()
