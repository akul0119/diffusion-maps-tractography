"""
Visualization Functions for Diffusion Maps Analysis
====================================================

Author: Akul Sharma
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_fiber_properties(stats, fiber_lengths, curvatures, dispersions,
                               output_dir, bundle_name):
    """
    Create visualizations of basic fiber properties.

    Parameters
    ----------
    stats : dict
        Bundle statistics from analyze_fiber_properties.
    fiber_lengths : list
        Fiber length values.
    curvatures : list
        Fiber curvature values.
    dispersions : list
        Fiber dispersion values.
    output_dir : Path
        Directory to save plots.
    bundle_name : str
        Bundle name for plot titles.
    """
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(15, 10))

    # 1. Fiber length distribution
    plt.subplot(221)
    plt.hist(fiber_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(stats['length_mean'], color='red', linestyle='--',
                label=f"Mean: {stats['length_mean']:.2f}")
    plt.title(f'{bundle_name}\nFiber Length Distribution')
    plt.xlabel('Length (mm)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Curvature distribution
    plt.subplot(222)
    plt.hist(curvatures, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.axvline(stats['curvature_mean'], color='red', linestyle='--',
                label=f"Mean: {stats['curvature_mean']:.4f}")
    plt.title('Fiber Curvature Distribution')
    plt.xlabel('Mean Curvature')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)

    # 3. Dispersion distribution
    plt.subplot(223)
    plt.hist(dispersions, bins=30, alpha=0.7, edgecolor='black', color='green')
    plt.axvline(stats['dispersion_mean'], color='red', linestyle='--',
                label=f"Mean: {stats['dispersion_mean']:.2f}")
    plt.title('Fiber Dispersion Distribution')
    plt.xlabel('Mean Distance from Center (mm)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.3)

    # 4. Length vs Curvature scatter
    plt.subplot(224)
    plt.scatter(fiber_lengths, curvatures, alpha=0.5, s=20)
    plt.title('Length vs Curvature Relationship')
    plt.xlabel('Fiber Length (mm)')
    plt.ylabel('Mean Curvature')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{bundle_name}_fiber_properties.png', dpi=150)
    plt.close()


def visualize_diffusion_maps(embedding, eigenvalues, features, output_dir, bundle_name):
    """
    Create comprehensive visualizations for diffusion map analysis.

    Parameters
    ----------
    embedding : ndarray
        Diffusion map embedding coordinates.
    eigenvalues : ndarray
        Eigenvalues from diffusion maps.
    features : ndarray
        Original feature matrix.
    output_dir : Path
        Directory to save plots.
    bundle_name : str
        Bundle name for plot titles.
    """
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(18, 6))

    # 1. 3D embedding colored by fiber length
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                         c=features[:, 0], cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(scatter, ax=ax1, label='Fiber Length (mm)', shrink=0.8)
    ax1.set_title(f'{bundle_name}\nDiffusion Map Embedding\n(colored by Length)')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_zlabel('Dimension 3')

    # 2. 3D embedding colored by curvature
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                         c=features[:, 1], cmap='plasma', s=10, alpha=0.6)
    plt.colorbar(scatter, ax=ax2, label='Curvature', shrink=0.8)
    ax2.set_title('Diffusion Map Embedding\n(colored by Curvature)')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.set_zlabel('Dimension 3')

    # 3. Eigenvalue spectrum
    ax3 = fig.add_subplot(133)
    ax3.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2, markersize=8)
    ax3.set_title('Eigenvalue Spectrum')
    ax3.set_xlabel('Component Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Add variance explained
    variance_explained = eigenvalues / eigenvalues.sum() * 100
    for i, (val, var) in enumerate(zip(eigenvalues, variance_explained)):
        ax3.text(i+1, val, f'{var:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'{bundle_name}_diffusion_maps.png', dpi=150)
    plt.close()

    # Create 2D pairwise plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Dim 1 vs 2
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1],
                             c=features[:, 0], cmap='viridis', s=20, alpha=0.5)
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].set_title(f'{bundle_name}\nDimensions 1 vs 2')
    plt.colorbar(scatter, ax=axes[0], label='Fiber Length')

    # Dim 1 vs 3
    scatter = axes[1].scatter(embedding[:, 0], embedding[:, 2],
                             c=features[:, 0], cmap='viridis', s=20, alpha=0.5)
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 3')
    axes[1].set_title('Dimensions 1 vs 3')
    plt.colorbar(scatter, ax=axes[1], label='Fiber Length')

    # Dim 2 vs 3
    scatter = axes[2].scatter(embedding[:, 1], embedding[:, 2],
                             c=features[:, 0], cmap='viridis', s=20, alpha=0.5)
    axes[2].set_xlabel('Dimension 2')
    axes[2].set_ylabel('Dimension 3')
    axes[2].set_title('Dimensions 2 vs 3')
    plt.colorbar(scatter, ax=axes[2], label='Fiber Length')

    plt.tight_layout()
    plt.savefig(output_dir / f'{bundle_name}_embedding_2d.png', dpi=150)
    plt.close()
