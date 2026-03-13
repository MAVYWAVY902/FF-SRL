"""
Train SAC agent for tumor peeling task.

Usage:
    python -m FF_SRL.train_peeling
    python -m FF_SRL.train_peeling --total_timesteps 200000 --max_steps 150
"""

import argparse
import os
import sys
import time

# Initialize warp before anything else
import warp as wp
wp.init()

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList)
from stable_baselines3.common.monitor import Monitor

from FF_SRL.env_tumor_peeling import TumorPeelingEnv


def make_env(args):
    """Create and wrap the environment."""
    env = TumorPeelingEnv(
        device=args.device,
        sim_substeps=args.sim_substeps,
        action_strength=args.action_strength,
        adhesion_stretch_abs_min=args.stretch_abs_min,
        adhesion_break_ratio=args.break_ratio,
        target_break_ratio=args.target_break_ratio,
        max_steps=args.max_steps,
        grab_radius=args.grab_radius,
        reward_break_weight=args.reward_break_weight,
        reward_deform_penalty=args.reward_deform_penalty,
    )
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train SAC for tumor peeling")

    # Environment
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sim_substeps", type=int, default=16)
    parser.add_argument("--action_strength", type=float, default=0.04,
                        help="Pull displacement per step (cm)")
    parser.add_argument("--stretch_abs_min", type=float, default=0.5,
                        help="Adhesion break threshold (cm)")
    parser.add_argument("--break_ratio", type=float, default=1.27,
                        help="Adhesion break strain ratio")
    parser.add_argument("--target_break_ratio", type=float, default=0.6,
                        help="Fraction of bonds to break for success")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--grab_radius", type=float, default=0.5,
                        help="Grab radius (cm)")
    parser.add_argument("--reward_break_weight", type=float, default=1.0)
    parser.add_argument("--reward_deform_penalty", type=float, default=0.01)

    # SAC
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent_coef", type=str, default="auto",
                        help="'auto' for automatic entropy tuning")

    # Output
    parser.add_argument("--save_dir", type=str, default="./output/peeling_sac")
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--checkpoint_freq", type=int, default=10000)

    args = parser.parse_args()

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("TUMOR PEELING SAC TRAINING")
    print("=" * 60)
    print(f"Save dir: {save_dir}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Action strength: {args.action_strength} cm/step")
    print(f"Target break ratio: {args.target_break_ratio}")
    print(f"Max steps/episode: {args.max_steps}")
    print()

    # Create environments
    print("Creating training environment...")
    train_env = make_env(args)

    print("Creating eval environment...")
    eval_env = make_env(args)

    # Create SAC agent
    print("\nCreating SAC agent...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        verbose=1,
        device="cpu",  # SAC networks on CPU (small), sim on GPU
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="sac_peeling",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Train
    print("\nStarting training...")
    print("-" * 60)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
    )

    # Save final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
