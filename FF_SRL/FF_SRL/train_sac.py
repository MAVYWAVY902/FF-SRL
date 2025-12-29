"""
SAC (Soft Actor-Critic) trainer for FF-SRL environment.

SAC is an off-policy algorithm that:
1. Uses a replay buffer for sample efficiency
2. Maximizes entropy for better exploration
3. Learns Q-functions and policy simultaneously
4. More stable than PPO for continuous control

Reference: https://arxiv.org/abs/1801.01290
"""

import argparse
import os
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from rl_env import VectorizedFFSRLEnv


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Preallocate memory
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, 
            next_obs: np.ndarray, done: np.ndarray):
        """Add a batch of transitions to the buffer."""
        batch_size = obs.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr % self.capacity
            
            self.observations[idx] = torch.from_numpy(obs[i]).to(self.device)
            self.actions[idx] = torch.from_numpy(action[i]).to(self.device)
            self.rewards[idx] = torch.tensor(reward[i], dtype=torch.float32, device=self.device)
            self.next_observations[idx] = torch.from_numpy(next_obs[i]).to(self.device)
            self.dones[idx] = torch.tensor(float(done[i]), dtype=torch.float32, device=self.device)
            
            self.ptr += 1
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size


# ============================================================================
# Neural Networks
# ============================================================================

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
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Sampled action (after tanh squashing)
            log_prob: Log probability of the action
        """
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        std = torch.exp(log_std)
        
        # Sample from Gaussian
        dist = Normal(mean, std)
        x_t = dist.rsample()  # Reparameterization trick
        
        # Tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale to action space
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob
    
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
            dist = Normal(mean, std)
            x_t = dist.rsample()
            action = torch.tanh(x_t)
        
        return action * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Twin Q-networks for stability (reduce overestimation)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both Q-values."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 value (for policy update)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


# ============================================================================
# SAC Agent
# ============================================================================

class SAC:
    """Soft Actor-Critic agent."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        hidden_dim: int = 256,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        
        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        if auto_tune_alpha:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = alpha  # Use the provided initial alpha
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for interaction."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            action = self.actor.get_action(obs_tensor, deterministic)
            return action.cpu().numpy()
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update networks with a batch of transitions."""
        obs, actions, rewards, next_obs, dones = batch
        
        # ========================================
        # Update Critic
        # ========================================
        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_probs = self.actor(next_obs)
            
            # Compute target Q-values (using target networks)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # Add entropy bonus
            target_q = rewards + (1 - dones) * self.gamma * (min_q_target - self.alpha * next_log_probs)
        
        # Current Q-values
        q1, q2 = self.critic(obs, actions)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========================================
        # Update Actor
        # ========================================
        # Sample new actions from current policy
        new_actions, log_probs = self.actor(obs)
        
        # Q-value of new actions
        q1_new = self.critic.q1_forward(obs, new_actions)
        
        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (self.alpha * log_probs - q1_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========================================
        # Update Alpha (entropy coefficient)
        # ========================================
        alpha_loss = 0.0
        if self.auto_tune_alpha:
            with torch.no_grad():
                _, log_probs = self.actor(obs)
            
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ========================================
        # Update Target Networks (soft update)
        # ========================================
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": self.alpha,
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
            "log_alpha": self.log_alpha if self.auto_tune_alpha else None,
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha = checkpoint["alpha"]
        if self.auto_tune_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha.data = checkpoint["log_alpha"].data


# ============================================================================
# Training Loop
# ============================================================================

def train_sac(args):
    """Main training loop."""
    
    # Setup
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"sac_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join("./output/tensorboard", f"sac_{timestamp}"))
    
    # Environment
    print("Creating environment...")
    env = VectorizedFFSRLEnv(
        num_envs=args.num_envs,
        device=device,
        max_episode_steps=args.max_episode_steps,
        action_strength=args.action_strength,
    )
    
    # Store randomization settings
    randomize_init_pos = args.randomize_init_pos
    init_pos_range = args.init_pos_range
    original_init_pos = np.array([0.0, 10.0, 4.0])  # Default training position
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    
    # Agent
    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_tune_alpha=args.auto_tune_alpha,
        hidden_dim=args.hidden_dim,
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=args.buffer_size,
        device=device,
    )
    
    # Training
    print(f"\nStarting SAC training for {args.total_steps:,} steps...")
    print(f"Save directory: {save_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Replay buffer size: {args.buffer_size:,}")
    print(f"Updates per step: {args.updates_per_step}")
    print(f"\n🎯 Early Stopping Settings:")
    print(f"  - Stop if success rate ≥ {args.early_stop_threshold:.0%} for {args.early_stop_patience} consecutive evals")
    print(f"  - Alpha capped at {args.max_alpha:.2f} (prevents over-exploration)")
    if randomize_init_pos:
        print(f"\n🔀 Generalization Training Enabled:")
        print(f"  - Randomizing initial position each episode")
        print(f"  - Position range: ±{init_pos_range:.3f} units (±{init_pos_range*10:.1f}mm)")
        print(f"  - This will improve policy generalization!")
    print("=" * 70)
    
    obs, _ = env.reset()
    episode_rewards = np.zeros(args.num_envs)
    episode_lengths = np.zeros(args.num_envs, dtype=int)
    episode_count = 0  # Track total episodes for logging
    
    start_time = time.time()
    best_reward = -float('inf')
    best_success_rate = 0.0
    consecutive_success_count = 0  # Track consecutive high success evals
    
    for step in range(args.total_steps):
        # 🔀 Randomize initial position BEFORE action/step (CRITICAL TIMING FIX!)
        # Must happen before env.step() so auto-reset uses the new position
        if randomize_init_pos and step > 0:
            # Generate random offset within range
            offset = np.random.uniform(-init_pos_range, init_pos_range, size=3)
            new_init_pos = original_init_pos + offset
            
            # Apply to simulation model - will be used by next auto-reset
            import warp as wp
            cartesian_action = wp.array(offset.astype(np.float32), dtype=wp.float32, device=env.env.device)
            env.env.simModel.setEffectorInitialPosition(cartesian_action)
            
            # Log randomization (occasionally)
            if step % 1000 == 0:
                writer.add_scalar("train/init_pos_x", new_init_pos[0], step)
                writer.add_scalar("train/init_pos_y", new_init_pos[1], step)
                writer.add_scalar("train/init_pos_z", new_init_pos[2], step)
        
        # Collect experience
        if step < args.start_steps:
            # Random actions for initial exploration
            action = np.random.uniform(-1, 1, (args.num_envs, action_dim))
        else:
            # Sample from policy
            action = agent.select_action(obs, deterministic=False)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        
        # Store in replay buffer
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Track statistics
        episode_rewards += reward
        episode_lengths += 1
        
        # Log episodes
        if done.any():
            for i in range(args.num_envs):
                if done[i]:
                    writer.add_scalar("train/episode_reward", episode_rewards[i], step)
                    writer.add_scalar("train/episode_length", episode_lengths[i], step)
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_count += 1
        
        obs = next_obs
        
        # Update networks
        if step >= args.start_steps and len(replay_buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = replay_buffer.sample(args.batch_size)
                update_info = agent.update(batch)
            
            # 🔥 Cap alpha to prevent over-exploration
            if args.auto_tune_alpha and agent.alpha > args.max_alpha:
                # Manually clamp log_alpha to keep alpha <= max_alpha
                with torch.no_grad():
                    agent.log_alpha.data.clamp_(max=np.log(args.max_alpha))
            
            # Log update info (only last update)
            if step % args.log_interval == 0:
                writer.add_scalar("train/critic_loss", update_info["critic_loss"], step)
                writer.add_scalar("train/actor_loss", update_info["actor_loss"], step)
                writer.add_scalar("train/alpha", update_info["alpha"], step)
                writer.add_scalar("train/q1_mean", update_info["q1_mean"], step)
        
        # Logging
        if step % args.log_interval == 0 and step > 0:
            elapsed = time.time() - start_time
            fps = step / elapsed
            
            print(f"Step {step:,}/{args.total_steps:,} | "
                  f"Buffer: {len(replay_buffer):,} | "
                  f"FPS: {fps:.0f} | "
                  f"Alpha: {agent.alpha:.3f}")
        
        # Evaluation
        if step % args.eval_interval == 0 and step > 0:
            eval_reward, eval_success = evaluate(env, agent, args.eval_episodes)
            writer.add_scalar("eval/mean_reward", eval_reward, step)
            writer.add_scalar("eval/success_rate", eval_success, step)
            
            print(f"Evaluation: Reward={eval_reward:.2f}, Success={eval_success:.2%}")
            
            # Save best model (by reward)
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(os.path.join(save_dir, "policy_best.pth"))
                print(f"  ✓ New best model saved! (reward: {best_reward:.2f})")
            
            # Save best model (by success rate)
            if eval_success > best_success_rate:
                best_success_rate = eval_success
                agent.save(os.path.join(save_dir, f"policy_success_{eval_success:.0%}_step_{step}.pth"))
                print(f"  ⭐ New best success rate: {eval_success:.2%}")
            
            # 🎯 Early stopping check
            if eval_success >= args.early_stop_threshold:
                consecutive_success_count += 1
                print(f"  🔥 High success rate achieved! ({consecutive_success_count}/{args.early_stop_patience})")
                
                if consecutive_success_count >= args.early_stop_patience:
                    print("\n" + "=" * 70)
                    print("🎉 EARLY STOPPING TRIGGERED!")
                    print(f"Success rate ≥ {args.early_stop_threshold:.0%} for {args.early_stop_patience} consecutive evaluations.")
                    print(f"Final step: {step:,}")
                    print(f"Final success rate: {eval_success:.2%}")
                    print(f"Final reward: {eval_reward:.2f}")
                    print("=" * 70 + "\n")
                    
                    # Save final model
                    agent.save(os.path.join(save_dir, f"policy_final_step_{step}.pth"))
                    env.close()
                    writer.close()
                    
                    print(f"✓ Training stopped early to preserve good policy!")
                    print(f"Best model: {save_dir}/policy_best.pth")
                    print(f"Final model: {save_dir}/policy_final_step_{step}.pth")
                    return
            else:
                consecutive_success_count = 0  # Reset counter
        
        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            agent.save(os.path.join(save_dir, f"policy_step_{step}.pth"))
    
    # Final save
    agent.save(os.path.join(save_dir, "policy_final.pth"))
    env.close()
    writer.close()
    
    print("\n" + "=" * 70)
    print("✓ Training completed!")
    print(f"Final model saved to: {save_dir}/policy_final.pth")
    print(f"Best model saved to: {save_dir}/policy_best.pth")
    print(f"View training curves with:")
    print(f"  tensorboard --logdir=./output/tensorboard/sac_{timestamp}")


def evaluate(env, agent, num_episodes: int) -> Tuple[float, float]:
    """Evaluate agent."""
    obs, _ = env.reset()
    episode_rewards = []
    episode_successes = []
    
    current_rewards = np.zeros(env.num_envs)
    episodes_done = 0
    
    while episodes_done < num_episodes:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        
        current_rewards += reward
        
        if done.any():
            for i in range(env.num_envs):
                if done[i] and episodes_done < num_episodes:
                    episode_rewards.append(current_rewards[i])
                    episode_successes.append(float(terminated[i]))
                    current_rewards[i] = 0
                    episodes_done += 1
    
    return np.mean(episode_rewards), np.mean(episode_successes)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAC on FF-SRL")
    
    # Environment
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--max_episode_steps", type=int, default=128, help="Max steps per episode")
    parser.add_argument("--action_strength", type=float, default=0.4, help="Action scaling factor")
    
    # Generalization (Initial Position Randomization)
    parser.add_argument("--randomize_init_pos", action="store_true", help="Randomize initial tool position each episode")
    parser.add_argument("--init_pos_range", type=float, default=0.2, help="Range for initial position randomization (±units, e.g., 0.2 = ±2mm)")
    
    # SAC hyperparameters
    parser.add_argument("--total_steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--start_steps", type=int, default=10000, help="Random exploration steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer capacity")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network soft update")
    parser.add_argument("--alpha", type=float, default=0.2, help="Initial entropy coefficient")
    parser.add_argument("--auto_tune_alpha", action="store_true", help="Auto-tune alpha")
    parser.add_argument("--max_alpha", type=float, default=0.8, help="Maximum alpha value (prevents over-exploration)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per env step")
    
    # Early stopping
    parser.add_argument("--early_stop_threshold", type=float, default=0.95, help="Success rate threshold for early stopping")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="Number of consecutive high-success evals before stopping")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=1000, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Evaluate every N steps")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--save_interval", type=int, default=50000, help="Save checkpoint every N steps")
    
    # System
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="./output", help="Save directory")
    
    args = parser.parse_args()
    
    # Set default: auto-tune alpha
    if not hasattr(args, 'auto_tune_alpha') or args.auto_tune_alpha is None:
        args.auto_tune_alpha = True
    
    train_sac(args)


if __name__ == "__main__":
    main()
