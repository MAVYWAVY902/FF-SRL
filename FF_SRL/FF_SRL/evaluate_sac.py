"""
Evaluate a trained SAC policy on FF-SRL environment.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from rl_env import VectorizedFFSRLEnv


# Copy Actor network definition from train_sac.py
class Actor(nn.Module):
    """Gaussian policy network with state-dependent std."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds for tanh squashing
        self.action_scale = 1.0
        self.action_bias = 0.0
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for evaluation (no log_prob needed)."""
        features = self.net(obs)
        mean = self.mean_head(features)
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            from torch.distributions import Normal
            dist = Normal(mean, std)
            x_t = dist.rsample()
            action = torch.tanh(x_t)
        
        return action * self.action_scale + self.action_bias


def evaluate_policy(
    checkpoint_path: str,
    episodes: int = 128,
    max_steps: int = 128,
    action_strength: float = 0.4,
    device: str = "cuda:0",
):
    """Evaluate SAC policy."""
    
    device = torch.device(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create environment
    print("Creating environment...")
    env = VectorizedFFSRLEnv(
        num_envs=1,
        device=device,
        max_episode_steps=max_steps,
        action_strength=action_strength,
    )
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create actor
    actor = Actor(obs_dim, action_dim, hidden_dim=256).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    
    print(f"\nEvaluating for {episodes} episodes...")
    print("=" * 70)
    
    # Evaluation loop
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    min_distances = []
    final_distances = []
    workspace_violations = []
    
    obs, _ = env.reset()
    current_reward = 0.0
    current_length = 0
    current_min_dist = float('inf')
    initial_dist = None
    workspace_viol_count = 0
    
    total_actions = []
    
    while episode_count < episodes:
        # Get action
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(device)
            action_tensor = actor.get_action(obs_tensor, deterministic=True)
            action = action_tensor.cpu().numpy()
        
        total_actions.append(action[0])
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated[0] or truncated[0]
        
        current_reward += reward[0]
        current_length += 1
        
        # Track statistics
        distance = info["distance_to_target"][0].item()
        current_min_dist = min(current_min_dist, distance)
        
        if initial_dist is None:
            initial_dist = distance
        
        # Check workspace violation (assuming info contains this)
        effector_pos = info["effector_position"][0].cpu().numpy()
        workspace_low = env.env.workspace_low_np
        workspace_high = env.env.workspace_high_np
        
        if np.any(effector_pos < workspace_low) or np.any(effector_pos > workspace_high):
            workspace_viol_count += 1
        
        if done:
            # Record episode
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episode_successes.append(float(terminated[0]))
            min_distances.append(current_min_dist)
            final_distances.append(distance)
            workspace_violations.append(workspace_viol_count)
            
            episode_count += 1
            
            # Print episode info
            print(f"\nEpisode {episode_count}/{episodes}:")
            print(f"  Initial distance: {initial_dist:.4f}")
            print(f"  Final distance: {distance:.4f}")
            print(f"  Min distance:   {current_min_dist:.4f}")
            print(f"  Reward: {current_reward:.2f}")
            print(f"  Steps: {current_length}")
            print(f"  Success: {'✓' if terminated[0] else '✗'}")
            
            # Action statistics
            actions_array = np.array(total_actions)
            mean_action_norm = np.linalg.norm(actions_array, axis=1).mean()
            max_action_component = np.abs(actions_array).max()
            
            print(f"  Mean |a|:       {mean_action_norm:.4f}")
            print(f"  Max |a_i|:      {max_action_component:.4f}")
            print(f"  Workspace viol: {workspace_viol_count}")
            
            # Reset
            current_reward = 0.0
            current_length = 0
            current_min_dist = float('inf')
            initial_dist = None
            workspace_viol_count = 0
            total_actions = []
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Episodes:        {episodes}")
    print(f"Mean Reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length:     {np.mean(episode_lengths):.1f} steps")
    print(f"Mean Final Dist: {np.mean(final_distances):.4f} units")
    print(f"Success Rate:    {sum(episode_successes)}/{episodes} ({100*np.mean(episode_successes):.1f}%)")
    print("=" * 70)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=128, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=128, help="Max steps per episode")
    parser.add_argument("--action_strength", type=float, default=0.4, help="Action scaling")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    evaluate_policy(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        action_strength=args.action_strength,
        device=args.device,
    )


if __name__ == "__main__":
    main()
