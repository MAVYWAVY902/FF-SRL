"""
Visualize generalization test results.

This script loads saved generalization test results and creates various
visualizations to analyze policy performance across different initial positions.

Usage:
    python visualize_generalization.py --results ./output/sac_20251228_112720/generalization_grid.npz
    python visualize_generalization.py --results ./output/sac_20251228_112720/generalization_random.npz --plot all
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


class GeneralizationVisualizer:
    """Visualize generalization test results."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        
        # Load results
        print(f"Loading results from: {results_path}")
        data = np.load(results_path, allow_pickle=True)
        
        self.results = data['results']
        self.mode = str(data['mode'])
        self.delta = float(data['delta'])
        self.num_tests = int(data['num_tests'])
        
        print(f"✓ Loaded {self.num_tests} test results")
        print(f"  Mode: {self.mode}")
        print(f"  Delta: ±{self.delta} units (±{self.delta*10:.1f}mm)")
        
        # Extract data
        self._extract_data()
    
    def _extract_data(self):
        """Extract data from results."""
        self.init_positions = np.array([r['init_pos'] for r in self.results])
        self.initial_distances = np.array([r['initial_distance'] for r in self.results])
        self.final_distances = np.array([r['final_distance'] for r in self.results])
        self.min_distances = np.array([r['min_distance'] for r in self.results])
        self.successes = np.array([r['success'] for r in self.results])
        self.steps = np.array([r['steps'] for r in self.results])
        self.rewards = np.array([r['reward'] for r in self.results])
        
        # Target position
        self.target_pos = np.array([0.31, 10.11, 1.61])
        
        # Original training position
        self.original_pos = np.array([0.0, 10.0, 4.0])
        
        print(f"\nData Summary:")
        print(f"  Success rate: {self.successes.mean()*100:.1f}%")
        print(f"  Mean steps: {self.steps.mean():.1f}")
        print(f"  Mean reward: {self.rewards.mean():.2f}")
        print(f"  Mean final distance: {self.final_distances.mean()*10:.2f}mm")
    
    def plot_3d_heatmap(self, save_path: str = None):
        """
        Create 3D scatter plot showing success rate across positions.
        Since all points may have 100% success, also show distance-based coloring.
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create 2 subplots: success rate and initial distance
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Extract coordinates
        x = self.init_positions[:, 0]
        y = self.init_positions[:, 1]
        z = self.init_positions[:, 2]
        
        # ========== Plot 1: Success Rate ==========
        success_values = self.successes.astype(float)
        
        # If all successes, use gradient for visibility
        if success_values.min() == success_values.max() == 1.0:
            # Color by distance to original position for better visualization
            dist_to_original = np.linalg.norm(self.init_positions - self.original_pos, axis=1)
            colors1 = dist_to_original
            cmap1 = plt.cm.Greens
            vmin1, vmax1 = dist_to_original.min(), dist_to_original.max()
            cbar_label1 = 'Distance from Original (units)'
        else:
            colors1 = success_values
            cmap1 = LinearSegmentedColormap.from_list('success', ['red', 'yellow', 'green'], N=100)
            vmin1, vmax1 = 0, 1
            cbar_label1 = 'Success Rate'
        
        scatter1 = ax1.scatter(x, y, z, 
                              c=colors1, 
                              cmap=cmap1,
                              s=40,
                              alpha=0.7,
                              edgecolors='black',
                              linewidth=0.3,
                              vmin=vmin1, vmax=vmax1)
        
        # Mark target position
        ax1.scatter(*self.target_pos, 
                   color='red', 
                   s=400, 
                   marker='*', 
                   edgecolors='black',
                   linewidth=2.5,
                   label='Target',
                   zorder=10)
        
        # Mark original training position
        ax1.scatter(*self.original_pos,
                   color='navy',
                   s=250,
                   marker='X',
                   edgecolors='black',
                   linewidth=2,
                   label='Training Start',
                   zorder=10)
        
        # Draw line from original to target
        ax1.plot([self.original_pos[0], self.target_pos[0]],
                [self.original_pos[1], self.target_pos[1]],
                [self.original_pos[2], self.target_pos[2]],
                'r--', linewidth=2, alpha=0.5, label='Training Direction')
        
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.7)
        cbar1.set_label(cbar_label1, fontsize=11)
        
        ax1.set_xlabel('X Position (units)', fontsize=11, labelpad=8)
        ax1.set_ylabel('Y Position (units)', fontsize=11, labelpad=8)
        ax1.set_zlabel('Z Position (units)', fontsize=11, labelpad=8)
        
        success_rate = self.successes.mean() * 100
        ax1.set_title(f'All Test Positions (100% Success)\n'
                     f'Colored by distance from training start',
                     fontsize=12, fontweight='bold', pad=15)
        
        ax1.legend(fontsize=9, loc='upper left')
        ax1.view_init(elev=20, azim=45)
        ax1.grid(True, alpha=0.3)
        
        # ========== Plot 2: Initial Distance to Target ==========
        initial_distances_mm = self.initial_distances * 10  # Convert to mm
        
        scatter2 = ax2.scatter(x, y, z,
                              c=initial_distances_mm,
                              cmap='plasma',
                              s=40,
                              alpha=0.7,
                              edgecolors='black',
                              linewidth=0.3)
        
        # Mark target
        ax2.scatter(*self.target_pos,
                   color='red',
                   s=400,
                   marker='*',
                   edgecolors='black',
                   linewidth=2.5,
                   label='Target',
                   zorder=10)
        
        # Mark original
        ax2.scatter(*self.original_pos,
                   color='navy',
                   s=250,
                   marker='X',
                   edgecolors='black',
                   linewidth=2,
                   label='Training Start',
                   zorder=10)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.7)
        cbar2.set_label('Initial Distance to Target (mm)', fontsize=11)
        
        ax2.set_xlabel('X Position (units)', fontsize=11, labelpad=8)
        ax2.set_ylabel('Y Position (units)', fontsize=11, labelpad=8)
        ax2.set_zlabel('Z Position (units)', fontsize=11, labelpad=8)
        
        ax2.set_title(f'Test Coverage Area\n'
                     f'Range: {initial_distances_mm.min():.1f}-{initial_distances_mm.max():.1f}mm from target',
                     fontsize=12, fontweight='bold', pad=15)
        
        ax2.legend(fontsize=9, loc='upper left')
        ax2.view_init(elev=20, azim=45)
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Policy Generalization: 3D Test Results\n'
                    f'{self.num_tests} Tests | Success Rate: {success_rate:.1f}% | '
                    f'Variation: ±{self.delta*10:.1f}mm',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 3D heatmap to: {save_path}")
        
        return fig
    
    def plot_distance_analysis(self, save_path: str = None):
        """
        Plot relationship between initial distance and success.
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Initial distance vs success rate
        ax = axes[0, 0]
        
        # Bin by distance
        distance_bins = np.linspace(self.initial_distances.min(), 
                                    self.initial_distances.max(), 20)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        success_rates = []
        counts = []
        for i in range(len(distance_bins) - 1):
            mask = (self.initial_distances >= distance_bins[i]) & \
                   (self.initial_distances < distance_bins[i+1])
            if mask.sum() > 0:
                success_rates.append(self.successes[mask].mean() * 100)
                counts.append(mask.sum())
            else:
                success_rates.append(np.nan)
                counts.append(0)
        
        ax.bar(bin_centers * 10, success_rates, 
               width=(distance_bins[1] - distance_bins[0]) * 10,
               alpha=0.7, color='green', edgecolor='black')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100%')
        ax.set_xlabel('Initial Distance to Target (mm)', fontsize=11)
        ax.set_ylabel('Success Rate (%)', fontsize=11)
        ax.set_title('Success Rate vs Initial Distance', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Initial distance distribution
        ax = axes[0, 1]
        ax.hist(self.initial_distances * 10, bins=30, 
               alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=self.initial_distances.mean() * 10, 
                  color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {self.initial_distances.mean()*10:.2f}mm')
        ax.set_xlabel('Initial Distance (mm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Initial Distances', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # 3. Final distance for successful trials
        ax = axes[1, 0]
        successful_final = self.final_distances[self.successes] * 10
        ax.hist(successful_final, bins=20,
               alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(x=successful_final.mean(),
                  color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {successful_final.mean():.2f}mm')
        ax.axvline(x=3.0, color='orange', linestyle=':', linewidth=2,
                  label='Success threshold: 3mm')
        ax.set_xlabel('Final Distance (mm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Final Distance Distribution (Successful: {self.successes.sum()}/{len(self.successes)})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # 4. Distance improvement
        ax = axes[1, 1]
        distance_improvement = (self.initial_distances - self.final_distances) * 10
        
        colors = ['green' if s else 'red' for s in self.successes]
        ax.scatter(self.initial_distances * 10, distance_improvement,
                  c=colors, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect improvement line
        x_range = np.array([self.initial_distances.min(), self.initial_distances.max()]) * 10
        ax.plot(x_range, x_range - 1.09, 'b--', linewidth=2,
               label='Perfect (→1.09mm)')
        
        ax.set_xlabel('Initial Distance (mm)', fontsize=11)
        ax.set_ylabel('Distance Improvement (mm)', fontsize=11)
        ax.set_title('Distance Improvement vs Initial Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with statistics
        stats_text = f"Statistics:\n"
        stats_text += f"Tests: {self.num_tests}\n"
        stats_text += f"Success: {self.successes.sum()}/{self.num_tests} ({self.successes.mean()*100:.1f}%)\n"
        stats_text += f"Dist range: {self.initial_distances.min()*10:.1f}-{self.initial_distances.max()*10:.1f}mm"
        
        fig.text(0.99, 0.01, stats_text, 
                fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='bottom',
                horizontalalignment='right',
                fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distance analysis to: {save_path}")
        
        return fig
    
    def plot_performance_heatmap(self, save_path: str = None):
        """
        Plot 2D heatmaps for different metrics (if grid mode).
        
        Args:
            save_path: Path to save figure (optional)
        """
        if self.mode != 'grid':
            print("⚠ Performance heatmap only available for grid mode")
            return None
        
        # Determine grid structure
        unique_x = np.unique(self.init_positions[:, 0])
        unique_y = np.unique(self.init_positions[:, 1])
        unique_z = np.unique(self.init_positions[:, 2])
        
        nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
        
        # Create figure with multiple heatmaps (different Z slices)
        n_slices = min(5, nz)  # Show up to 5 Z slices
        slice_indices = np.linspace(0, nz-1, n_slices, dtype=int)
        
        fig, axes = plt.subplots(2, n_slices, figsize=(4*n_slices, 8))
        if n_slices == 1:
            axes = axes.reshape(2, 1)
        
        for idx, z_idx in enumerate(slice_indices):
            z_val = unique_z[z_idx]
            
            # Get data for this Z slice
            mask = np.abs(self.init_positions[:, 2] - z_val) < 1e-6
            slice_data = self.init_positions[mask]
            slice_success = self.successes[mask]
            slice_final_dist = self.final_distances[mask]
            
            # Create grid
            success_grid = np.full((ny, nx), np.nan)
            dist_grid = np.full((ny, nx), np.nan)
            
            for i, y_val in enumerate(unique_y):
                for j, x_val in enumerate(unique_x):
                    point_mask = (np.abs(slice_data[:, 0] - x_val) < 1e-6) & \
                                (np.abs(slice_data[:, 1] - y_val) < 1e-6)
                    if point_mask.any():
                        success_grid[i, j] = slice_success[point_mask][0]
                        dist_grid[i, j] = slice_final_dist[point_mask][0] * 10
            
            # Plot success rate
            ax = axes[0, idx]
            im1 = ax.imshow(success_grid, cmap='RdYlGn', vmin=0, vmax=1,
                          extent=[unique_x.min(), unique_x.max(),
                                 unique_y.min(), unique_y.max()],
                          origin='lower', aspect='auto')
            
            # Mark target and original if in this slice
            if np.abs(z_val - self.target_pos[2]) < 0.1:
                ax.scatter(*self.target_pos[:2], color='red', s=200, 
                         marker='*', edgecolors='black', linewidth=2, zorder=10)
            if np.abs(z_val - self.original_pos[2]) < 0.1:
                ax.scatter(*self.original_pos[:2], color='blue', s=150,
                         marker='X', edgecolors='black', linewidth=2, zorder=10)
            
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f'Success Rate\nZ={z_val:.2f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if idx == n_slices - 1:
                cbar1 = plt.colorbar(im1, ax=ax)
                cbar1.set_label('Success Rate', fontsize=10)
            
            # Plot final distance
            ax = axes[1, idx]
            im2 = ax.imshow(dist_grid, cmap='RdYlGn_r', vmin=0, vmax=3,
                          extent=[unique_x.min(), unique_x.max(),
                                 unique_y.min(), unique_y.max()],
                          origin='lower', aspect='auto')
            
            # Mark target and original
            if np.abs(z_val - self.target_pos[2]) < 0.1:
                ax.scatter(*self.target_pos[:2], color='red', s=200,
                         marker='*', edgecolors='black', linewidth=2, zorder=10)
            if np.abs(z_val - self.original_pos[2]) < 0.1:
                ax.scatter(*self.original_pos[:2], color='blue', s=150,
                         marker='X', edgecolors='black', linewidth=2, zorder=10)
            
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f'Final Distance (mm)\nZ={z_val:.2f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if idx == n_slices - 1:
                cbar2 = plt.colorbar(im2, ax=ax)
                cbar2.set_label('Final Distance (mm)', fontsize=10)
        
        plt.suptitle(f'Policy Performance Heatmaps - {self.mode.upper()} Test\n'
                    f'Success Rate: {self.successes.mean()*100:.1f}%',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved performance heatmap to: {save_path}")
        
        return fig
    
    def plot_all(self, output_dir: str = None):
        """
        Generate all visualizations.
        
        Args:
            output_dir: Directory to save figures (optional)
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving figures to: {output_dir}")
        
        # 3D heatmap
        print("\n1. Generating 3D heatmap...")
        save_path = output_dir / "generalization_3d_heatmap.png" if output_dir else None
        self.plot_3d_heatmap(save_path)
        
        # Distance analysis
        print("\n2. Generating distance analysis...")
        save_path = output_dir / "generalization_distance_analysis.png" if output_dir else None
        self.plot_distance_analysis(save_path)
        
        # Performance heatmap (grid only)
        if self.mode == 'grid':
            print("\n3. Generating performance heatmaps...")
            save_path = output_dir / "generalization_performance_heatmap.png" if output_dir else None
            self.plot_performance_heatmap(save_path)
        
        print("\n✓ All visualizations complete!")
        
        if not output_dir:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize generalization test results")
    
    parser.add_argument("--results", type=str, required=True,
                       help="Path to generalization results .npz file")
    parser.add_argument("--plot", type=str, default="all",
                       choices=["3d", "distance", "heatmap", "all"],
                       help="Which plot(s) to generate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for saved figures")
    parser.add_argument("--show", action="store_true",
                       help="Show plots interactively (in addition to saving)")
    
    args = parser.parse_args()
    
    # Check results file
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    print("\n" + "="*70)
    print("GENERALIZATION TEST VISUALIZATION")
    print("="*70)
    
    # Create visualizer
    visualizer = GeneralizationVisualizer(str(results_path))
    
    # Generate plots
    if args.plot == "all":
        visualizer.plot_all(output_dir=args.output)
    elif args.plot == "3d":
        save_path = Path(args.output) / "generalization_3d_heatmap.png" if args.output else None
        visualizer.plot_3d_heatmap(save_path)
    elif args.plot == "distance":
        save_path = Path(args.output) / "generalization_distance_analysis.png" if args.output else None
        visualizer.plot_distance_analysis(save_path)
    elif args.plot == "heatmap":
        save_path = Path(args.output) / "generalization_performance_heatmap.png" if args.output else None
        visualizer.plot_performance_heatmap(save_path)
    
    # Show plots if requested
    if args.show or not args.output:
        plt.show()
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
