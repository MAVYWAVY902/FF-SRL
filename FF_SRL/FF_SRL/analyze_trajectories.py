"""
Analyze and visualize saved trajectory data from SAC policy.

This script loads saved trajectory files and creates:
- Action sequence plots
- Distance convergence curves
- Reward accumulation
- 3D trajectory visualization
- Action magnitude analysis

Usage:
    python analyze_trajectories.py --dir ./output/sac_20251228_112720
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_trajectory(path: str) -> dict:
    """Load trajectory from .npz file."""
    data = np.load(path)
    return {
        'observations': data['observations'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'tool_positions': data['tool_positions'],
        'distances': data['distances'],
        'success': bool(data['success']),
        'episode_reward': float(data['episode_reward']),
        'episode_length': int(data['episode_length'])
    }


def plot_action_sequences(trajectories: list, save_path: Path = None):
    """Plot action sequences for all dimensions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Action Sequences (All Episodes)', fontsize=16, fontweight='bold')
    
    dims = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, (dim, color) in enumerate(zip(dims, colors)):
        ax = axes[i]
        
        for ep_idx, traj in enumerate(trajectories):
            actions = traj['actions']
            steps = np.arange(1, len(actions) + 1)
            
            ax.plot(steps, actions[:, i], 
                   marker='o', linewidth=2, markersize=6,
                   alpha=0.7, label=f'Episode {ep_idx+1}',
                   color=color if len(trajectories) == 1 else None)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(f'Action {dim}', fontsize=12)
        ax.set_title(f'{dim}-axis Actions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add bounds
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Action limit')
        ax.axhline(y=-1.0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'action_sequences.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path / 'action_sequences.png'}")
    
    plt.show()


def plot_distance_convergence(trajectories: list, save_path: Path = None):
    """Plot distance to target over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ep_idx, traj in enumerate(trajectories):
        distances = traj['distances']
        steps = np.arange(1, len(distances) + 1)
        
        ax.plot(steps, distances * 10,  # Convert to mm
               marker='o', linewidth=2, markersize=6,
               label=f"Episode {ep_idx+1} ({'✓' if traj['success'] else '✗'})",
               alpha=0.8)
    
    # Success threshold
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=2, 
              label='Success threshold (3mm)', alpha=0.7)
    
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Distance to Target (mm)', fontsize=14)
    ax.set_title('Distance Convergence Over Time', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'distance_convergence.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path / 'distance_convergence.png'}")
    
    plt.show()


def plot_reward_accumulation(trajectories: list, save_path: Path = None):
    """Plot cumulative reward over steps."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ep_idx, traj in enumerate(trajectories):
        rewards = traj['rewards']
        cumulative_rewards = np.cumsum(rewards)
        steps = np.arange(1, len(rewards) + 1)
        
        ax.plot(steps, cumulative_rewards,
               marker='o', linewidth=2, markersize=6,
               label=f"Episode {ep_idx+1} (Total: {traj['episode_reward']:.2f})",
               alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Cumulative Reward', fontsize=14)
    ax.set_title('Reward Accumulation', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'reward_accumulation.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path / 'reward_accumulation.png'}")
    
    plt.show()


def plot_3d_trajectory(trajectories: list, save_path: Path = None):
    """Plot 3D tool trajectory."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    for ep_idx, traj in enumerate(trajectories):
        positions = traj['tool_positions']
        
        # Trajectory line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               linewidth=2, alpha=0.7, label=f'Episode {ep_idx+1}')
        
        # Start point
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                  s=200, c='green', marker='o', edgecolors='black', linewidths=2,
                  label='Start' if ep_idx == 0 else '')
        
        # End point
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                  s=200, c='blue', marker='s', edgecolors='black', linewidths=2,
                  label='End' if ep_idx == 0 else '')
        
        # Target (from observations)
        target_pos = traj['observations'][0][3:6]
        if ep_idx == 0:
            ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                      s=300, c='red', marker='*', edgecolors='black', linewidths=2,
                      label='Target')
    
    ax.set_xlabel('X (units)', fontsize=12)
    ax.set_ylabel('Y (units)', fontsize=12)
    ax.set_zlabel('Z (units)', fontsize=12)
    ax.set_title('3D Tool Trajectory', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    max_range = np.array([
        np.ptp([traj['tool_positions'][:, i] for traj in trajectories])
        for i in range(3)
    ]).max() / 2.0
    
    mid_x = np.mean([traj['tool_positions'][:, 0] for traj in trajectories])
    mid_y = np.mean([traj['tool_positions'][:, 1] for traj in trajectories])
    mid_z = np.mean([traj['tool_positions'][:, 2] for traj in trajectories])
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / '3d_trajectory.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path / '3d_trajectory.png'}")
    
    plt.show()


def plot_action_magnitudes(trajectories: list, save_path: Path = None):
    """Plot action magnitude statistics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Total action magnitude
    ax1 = axes[0]
    for ep_idx, traj in enumerate(trajectories):
        actions = traj['actions']
        magnitudes = np.linalg.norm(actions, axis=1)
        steps = np.arange(1, len(magnitudes) + 1)
        
        ax1.plot(steps, magnitudes,
                marker='o', linewidth=2, markersize=6,
                label=f'Episode {ep_idx+1}',
                alpha=0.8)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Action Magnitude ||a||', fontsize=12)
    ax1.set_title('Action Magnitude Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Per-dimension magnitude
    ax2 = axes[1]
    for ep_idx, traj in enumerate(trajectories):
        actions = traj['actions']
        abs_actions = np.abs(actions)
        steps = np.arange(1, len(actions) + 1)
        
        for i, dim in enumerate(['X', 'Y', 'Z']):
            ax2.plot(steps, abs_actions[:, i],
                    linewidth=2, alpha=0.6,
                    label=f'Episode {ep_idx+1} - {dim}' if len(trajectories) <= 2 else None)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Absolute Action Value', fontsize=12)
    ax2.set_title('Per-Dimension Action Magnitudes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(trajectories) <= 2:
        ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'action_magnitudes.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path / 'action_magnitudes.png'}")
    
    plt.show()


def print_statistics(trajectories: list):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)
    
    # Overall statistics
    success_rate = sum(t['success'] for t in trajectories) / len(trajectories)
    mean_reward = np.mean([t['episode_reward'] for t in trajectories])
    mean_length = np.mean([t['episode_length'] for t in trajectories])
    
    print(f"\nOverall Statistics ({len(trajectories)} episodes):")
    print(f"  Success Rate:     {success_rate*100:.1f}%")
    print(f"  Mean Reward:      {mean_reward:.2f}")
    print(f"  Mean Length:      {mean_length:.1f} steps")
    
    # Per-episode details
    print(f"\nPer-Episode Analysis:")
    for i, traj in enumerate(trajectories):
        print(f"\n  Episode {i+1}:")
        print(f"    Success:        {'✓' if traj['success'] else '✗'}")
        print(f"    Steps:          {traj['episode_length']}")
        print(f"    Total Reward:   {traj['episode_reward']:.2f}")
        
        distances = traj['distances']
        print(f"    Initial Dist:   {distances[0]*10:.2f} mm")
        print(f"    Final Dist:     {distances[-1]*10:.2f} mm")
        print(f"    Min Dist:       {distances.min()*10:.2f} mm")
        print(f"    Distance Δ:     {(distances[0] - distances[-1])*10:.2f} mm")
        
        actions = traj['actions']
        action_norms = np.linalg.norm(actions, axis=1)
        print(f"    Mean ||a||:     {action_norms.mean():.4f}")
        print(f"    Max ||a||:      {action_norms.max():.4f}")
        print(f"    Max |a_i|:      {np.abs(actions).max():.4f}")
        
        # Action direction analysis
        action_signs = np.sign(actions.mean(axis=0))
        print(f"    Action tendency: X={'+' if action_signs[0]>0 else '-'}, "
              f"Y={'+' if action_signs[1]>0 else '-'}, "
              f"Z={'+' if action_signs[2]>0 else '-'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAC trajectory data")
    parser.add_argument("--dir", type=str, required=True,
                       help="Directory containing trajectory_episode_*.npz files")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of episodes to analyze (default: all)")
    parser.add_argument("--save", action="store_true",
                       help="Save plots to directory")
    
    args = parser.parse_args()
    
    # Find trajectory files
    traj_dir = Path(args.dir)
    if not traj_dir.exists():
        print(f"Error: Directory not found: {traj_dir}")
        return
    
    traj_files = sorted(traj_dir.glob("trajectory_episode_*.npz"))
    if not traj_files:
        print(f"Error: No trajectory files found in {traj_dir}")
        return
    
    if args.episodes:
        traj_files = traj_files[:args.episodes]
    
    print(f"\nLoading {len(traj_files)} trajectory files from: {traj_dir}")
    
    # Load trajectories
    trajectories = []
    for traj_file in traj_files:
        traj = load_trajectory(str(traj_file))
        trajectories.append(traj)
        print(f"  ✓ Loaded: {traj_file.name}")
    
    # Print statistics
    print_statistics(trajectories)
    
    # Create plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    save_path = traj_dir if args.save else None
    
    plot_distance_convergence(trajectories, save_path)
    plot_action_sequences(trajectories, save_path)
    plot_reward_accumulation(trajectories, save_path)
    plot_3d_trajectory(trajectories, save_path)
    plot_action_magnitudes(trajectories, save_path)
    
    if args.save:
        print(f"\n✓ All plots saved to: {traj_dir}")
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
