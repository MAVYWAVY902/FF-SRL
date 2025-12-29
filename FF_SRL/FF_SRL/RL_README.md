# FF-SRL Reinforcement Learning Training

This directory contains the reinforcement learning (RL) training code for the FF-SRL surgical simulation environment, reproducing the experiments from the paper:

**"FF-SRL: High Performance GPU-Based Surgical Simulation For Robot Learning"**

[![arXiv](https://img.shields.io/badge/arXiv-2503.18616-b31b1b.svg)](https://arxiv.org/abs/2503.18616)

## Overview

The RL training infrastructure enables training surgical robot manipulation policies using the Proximal Policy Optimization (PPO) algorithm entirely on GPU. This achieves:

- **Fast training**: Complete tissue retraction task training in <2 minutes (32 environments)
- **GPU-accelerated**: Both simulation and RL training on single GPU
- **Scalable**: Support for 1-1000+ parallel environments
- **Efficient**: CUDA graphs and vectorized operations

## Features

✅ **Gymnasium-compatible environment wrapper** (`rl_env.py`)
✅ **PPO training (self-contained)** (`train_simple_ppo.py`)
✅ **Configurable hyperparameters** matching paper Table I (`rl_config.py`)
✅ **Evaluation and visualization tools** (`evaluate.py`)
✅ **Tissue retraction task** - reaching target on deformable tissue
✅ **Reward shaping** with distance-based rewards
✅ **Multi-environment parallelization**

## Installation

### Requirements

```bash
# Core dependencies (already in main setup.py)
- Python >= 3.7
- PyTorch >= 1.10
- warp-lang == 0.10.1
- numpy
- usd-core

# Additional RL dependencies
- rl-games >= 1.6.0
- gymnasium >= 0.28.0
- matplotlib
- pyyaml
```

### Install RL Dependencies

```bash
# Install rl_games for PPO training
pip install rl-games

# Install gymnasium (modern fork of OpenAI Gym)
pip install gymnasium

# Install visualization dependencies
pip install matplotlib pyyaml tensorboard
```

### Verify Installation

```bash
cd FF_SRL/FF_SRL
python -c "import rl_games; import gymnasium; print('RL dependencies OK')"
```

## Quick Start

### 1. Basic Training (8 environments)

Train a PPO agent on the tissue retraction task with default parameters:

```bash
cd FF_SRL/FF_SRL
python train_simple_ppo.py --num_envs 8 --total_steps 500000
```

This will:
- Train for 500K simulation steps (~10-15 minutes on RTX 2060)
- Save checkpoints every 100 epochs
- Log training metrics to `./output/runs/`

### 2. Fast Training (32 environments)

For faster training as reported in the paper (<2 minutes):

```bash
python train_simple_ppo.py --num_envs 32 --total_steps 500000
```

**Note**: Requires GPU with sufficient VRAM (≥6GB for 32 envs)

### 3. Training with Visualization

Enable rendering during training (slower but useful for debugging):

```bash
python train_ppo.py --num_envs 1 --visualize
```

Note: This branch does not include `train_ppo.py`. Use `train_simple_ppo.py` for PPO training.

### 4. Resume Training

Resume from a checkpoint:

```bash
python train_simple_ppo.py --resume ./output/<run_dir>/checkpoint_epoch_<N>.pth
```

## Training Configuration

### Hyperparameters (from Paper Table I)

The default configuration in `rl_config.py` matches the paper:

| Hyperparameter | Value |
|----------------|-------|
| Total simulation steps | 500,000 |
| Steps before PPO update | 1024 (summed over envs) |
| Minibatch size | 256 |
| Update epochs | 4 |
| Discount factor (γ) | 0.995 |
| GAE lambda (λ) | 0.95 |
| Learning rate | 2.5e-4 → 0 (linear) |
| Clip range | 0.1 → 0 (linear) |
| Clip range (value) | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.0 |
| Max gradient norm | 0.5 |

### Reward Function

```python
R = wl * l + wd * d + ws * s

where:
  l = distance to target (meters)
  d = change in distance from previous step
  s = success flag (1 if distance < 3mm, else 0)
  
Default weights: wl=-1, wd=-10, ws=100
```

### Observation Space

6D vector: `[effector_x, effector_y, effector_z, target_x, target_y, target_z]`

### Action Space

3D continuous: `[dx, dy, dz]` - Cartesian movements in range [-1, 1]

## Evaluation

### Evaluate Trained Model

```bash
python evaluate.py \
    --checkpoint ./output/runs/ff_srl_20250127/model.pth \
    --num_episodes 100 \
    --visualize
```

### Record Trajectories

```bash
python evaluate.py \
    --checkpoint ./output/runs/ff_srl_20250127/model.pth \
    --num_episodes 50 \
    --record \
    --save_dir ./output/eval
```

### Plot Training Progress

```bash
python evaluate.py --plot_logs ./output/runs/ff_srl_20250127/
```

### Analyze Trajectories

```bash
python evaluate.py --analyze_trajectories ./output/eval/trajectories.npy
```

## Advanced Usage

### Custom Configuration

Create a custom config file:

```python
# my_config.py
from rl_config import params

custom_params = {
    **params,
    "config": {
        **params["config"],
        "learning_rate": 1e-4,
        "num_actors": 16,
        "gamma": 0.99,
    }
}
```

Then use it:

```bash
python train_ppo.py --config my_config.py
```

### Modify Reward Function

Edit `rl_env.py`:

```python
def _compute_reward(self):
    # Custom reward implementation
    wl, wd, ws = (-2.0, -20.0, 200.0)  # Stronger penalties
    ...
```

### Change Task Parameters

```bash
python train_ppo.py \
    --scene_path ./scenes/custom_scene.usd \
    --success_threshold 0.005 \
    --max_episode_steps 300
```

## Performance Benchmarks

Results on NVIDIA RTX 2060 Mobile (6GB VRAM):

| Num Envs | FPS (Sim Only) | FPS (RL Training) | Training Time (500K steps) |
|----------|----------------|-------------------|----------------------------|
| 1        | 2965 ± 41      | 434 ± 8          | ~24 minutes               |
| 4        | 9093 ± 133     | 1527 ± 34        | ~6.8 minutes              |
| 8        | 14021 ± 193    | 2637 ± 41        | ~3.9 minutes              |
| 16       | 19033 ± 98     | 3955 ± 202       | ~2.6 minutes              |
| 32       | 21662 ± 145    | 4992 ± 336       | **~1.9 minutes**          |

## File Structure

```
FF_SRL/FF_SRL/
├── rl_env.py           # Gymnasium environment wrapper
├── rl_config.py        # rl_games-style PPO hyperparameter configurations
├── train_simple_ppo.py # PPO trainer (self-contained)
├── evaluate.py         # Evaluation and visualization
├── scenes/
│   └── liverRetraction.usd  # Tissue retraction scene
└── output/
    ├── runs/           # Training checkpoints and logs
    └── eval/           # Evaluation results
```

## Troubleshooting

### Out of Memory (OOM)

Reduce number of environments or batch size:

```bash
python train_ppo.py --num_envs 4  # Use fewer environments
```

Or modify config:

```python
config["config"]["minibatch_size"] = 128  # Smaller batches
```

### Slow Training

1. Enable CUDA graphs (default):
   ```python
   use_graph=True  # In rl_env.py
   ```

2. Increase parallel environments:
   ```bash
   python train_ppo.py --num_envs 32
   ```

3. Disable visualization:
   ```bash
   # Remove --visualize flag
   ```

### Import Errors

```bash
# If rl_games not found
pip install rl-games

# If gymnasium not found
pip install gymnasium

# If FF_SRL not found
cd FF-SRL
pip install -e .
```

### CUDA Errors

Ensure NVIDIA drivers and CUDA are properly installed:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check warp CUDA initialization
python -c "import warp as wp; wp.init(); print('Warp initialized')"
```

## Citation

If you use this RL training code, please cite:

```bibtex
@article{dallalba2025ffsrl,
  title={FF-SRL: High Performance GPU-Based Surgical Simulation For Robot Learning},
  author={Dall'Alba, Diego and Naskr{\k{e}}t, Michał and Kamińska, Sabina and Korzeniowski, Przemysław},
  journal={arXiv preprint arXiv:2503.18616},
  year={2025}
}
```

## Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/SanoScience/FF-SRL)
- Check the [paper](https://arxiv.org/abs/2503.18616) for details
- Contact: p.korzeniowski@sanoscience.org

## License

See main repository LICENSE file.

## Acknowledgments

This work was supported by:
- Sano Centre for Computational Medicine
- EU Horizon 2020 (Grant No. 857533)
- Foundation for Polish Science
- Polish high-performance computing infrastructure PLGrid
