"""
Example Usage of Diffusion Maps Tractography Framework
=======================================================

This script demonstrates how to use the framework for analyzing
fiber tract bundles.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    load_streamlines,
    extract_fiber_features,
    compute_diffusion_maps,
    analyze_fiber_properties
)
from visualization import visualize_fiber_properties, visualize_diffusion_maps


def example_single_bundle():
    """
    Example: Analyze a single bundle
    """
    print("Example 1: Single Bundle Analysis")
    print("="*50)

    # Path to your tractography data
    vtk_file = "path/to/your/curves.vtk.gz"
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)

    # Load streamlines
    print("Loading streamlines...")
    fibers = load_streamlines(vtk_file)

    if fibers is None:
        print("Error: Could not load fibers")
        return

    print(f"Loaded {len(fibers)} fibers")

    # Analyze basic properties
    print("\nAnalyzing fiber properties...")
    stats, lengths, curvatures, dispersions = analyze_fiber_properties(fibers)

    print(f"Mean length: {stats['length_mean']:.2f} mm")
    print(f"Mean curvature: {stats['curvature_mean']:.4f}")
    print(f"Bundle center: {stats['bundle_center']}")

    # Visualize properties
    visualize_fiber_properties(stats, lengths, curvatures, dispersions,
                               output_dir, "example_bundle")

    # Extract features
    print("\nExtracting features...")
    features = extract_fiber_features(fibers)
    print(f"Feature matrix shape: {features.shape}")

    # Compute diffusion maps
    print("\nComputing diffusion maps...")
    embedding, eigenvalues = compute_diffusion_maps(features, n_components=3)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Eigenvalues: {eigenvalues}")

    # Visualize results
    visualize_diffusion_maps(embedding, eigenvalues, features,
                            output_dir, "example_bundle")

    print(f"\nResults saved to: {output_dir}")


def example_custom_features():
    """
    Example: Extract features with custom parameters
    """
    print("\nExample 2: Custom Feature Extraction")
    print("="*50)

    vtk_file = "path/to/your/curves.vtk.gz"
    fibers = load_streamlines(vtk_file)

    if fibers is None:
        return

    # Extract features
    features = extract_fiber_features(fibers)

    # Access individual feature columns
    lengths = features[:, 0]
    curvatures = features[:, 1]
    dispersions = features[:, 2]
    start_points = features[:, 3:6]
    end_points = features[:, 6:9]
    mid_points = features[:, 9:12]

    print(f"Extracted features for {len(fibers)} fibers:")
    print(f"  Lengths: min={lengths.min():.2f}, max={lengths.max():.2f}")
    print(f"  Curvatures: min={curvatures.min():.4f}, max={curvatures.max():.4f}")
    print(f"  Dispersions: min={dispersions.min():.2f}, max={dispersions.max():.2f}")


def example_parameter_tuning():
    """
    Example: Tune diffusion maps parameters
    """
    print("\nExample 3: Parameter Tuning")
    print("="*50)

    import numpy as np
    from core import compute_kernel_bandwidth

    vtk_file = "path/to/your/curves.vtk.gz"
    fibers = load_streamlines(vtk_file)

    if fibers is None:
        return

    features = extract_fiber_features(fibers)

    # Try different bandwidth selection methods
    epsilon_median = compute_kernel_bandwidth(features, method='median')
    epsilon_p25 = compute_kernel_bandwidth(features, method='percentile', percentile=25)
    epsilon_p75 = compute_kernel_bandwidth(features, method='percentile', percentile=75)

    print(f"Bandwidth options:")
    print(f"  Median: {epsilon_median:.4f}")
    print(f"  25th percentile: {epsilon_p25:.4f}")
    print(f"  75th percentile: {epsilon_p75:.4f}")

    # Compute with different parameters
    print("\nComputing with different n_components...")
    for n_comp in [2, 3, 5]:
        embedding, eigenvalues = compute_diffusion_maps(features, n_components=n_comp)
        variance_explained = eigenvalues / eigenvalues.sum() * 100
        print(f"  n_components={n_comp}: variance={variance_explained.sum():.2f}%")


if __name__ == "__main__":
    print("Diffusion Maps Tractography - Example Usage")
    print("="*50)
    print("\nNote: Update file paths in the examples before running!\n")

    # Run examples
    # Uncomment to run specific examples:

    # example_single_bundle()
    # example_custom_features()
    # example_parameter_tuning()

    print("\nTo run examples, update the file paths and uncomment the example calls.")
