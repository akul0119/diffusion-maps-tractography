"""
Bundle-Level Diffusion Maps Analysis
=====================================

Main pipeline for analyzing individual fiber bundles using diffusion maps.

Author: Akul Sharma
License: MIT
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    load_streamlines,
    extract_fiber_features,
    compute_diffusion_maps,
    analyze_fiber_properties,
    read_bundle_list
)
from visualization import (
    visualize_fiber_properties,
    visualize_diffusion_maps
)


def analyze_single_bundle(bundle_path, output_dir, bundle_name, n_components=3):
    """
    Run complete diffusion maps analysis for a single fiber bundle.

    Parameters
    ----------
    bundle_path : Path
        Path to bundle VTK file (curves.vtk.gz).
    output_dir : Path
        Directory to save results.
    bundle_name : str
        Name of the bundle for labeling outputs.
    n_components : int, default=3
        Number of diffusion map components to compute.

    Returns
    -------
    results : dict
        Dictionary containing all analysis results including:
        - stats: Bundle statistics
        - embedding: Diffusion map coordinates
        - eigenvalues: Eigenvalues from diffusion maps
        - features: Extracted fiber features
    """
    print(f"\nAnalyzing bundle: {bundle_name}")
    print(f"Loading streamlines from: {bundle_path}")

    # Load streamlines
    fibers = load_streamlines(bundle_path)

    if fibers is None or len(fibers) == 0:
        print(f"ERROR: No fibers found in {bundle_path}")
        return None

    print(f"Loaded {len(fibers)} fibers")

    # Analyze basic fiber properties
    print("Computing fiber properties...")
    stats, lengths, curvatures, dispersions = analyze_fiber_properties(fibers)

    # Visualize basic properties
    print("Creating fiber property visualizations...")
    visualize_fiber_properties(stats, lengths, curvatures, dispersions,
                               output_dir, bundle_name)

    # Extract features for diffusion maps
    print("Extracting features for diffusion maps...")
    features = extract_fiber_features(fibers)
    print(f"Feature matrix shape: {features.shape}")

    # Compute diffusion maps
    print("Computing diffusion maps embedding...")
    embedding, eigenvalues = compute_diffusion_maps(features,
                                                    n_components=n_components)

    # Visualize diffusion map results
    print("Creating diffusion map visualizations...")
    visualize_diffusion_maps(embedding, eigenvalues, features,
                            output_dir, bundle_name)

    # Save numerical results
    print("Saving results...")
    np.save(output_dir / f'{bundle_name}_embedding.npy', embedding)
    np.save(output_dir / f'{bundle_name}_eigenvalues.npy', eigenvalues)
    np.save(output_dir / f'{bundle_name}_features.npy', features)
    np.save(output_dir / f'{bundle_name}_fiber_metrics.npy', {
        'lengths': lengths,
        'curvatures': curvatures,
        'dispersions': dispersions
    })

    # Print summary
    print(f"\nAnalysis complete for {bundle_name}:")
    print(f"  Number of fibers: {len(fibers)}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Results saved to: {output_dir}")

    return {
        'stats': stats,
        'embedding': embedding,
        'eigenvalues': eigenvalues,
        'features': features
    }


def process_subject(subject_id, base_dir, output_base, bundle_list_file, n_components=3):
    """
    Process all bundles for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    base_dir : Path
        Base directory containing subject data.
    output_base : Path
        Base directory for output.
    bundle_list_file : Path
        Path to file listing bundles to analyze.
    n_components : int, default=3
        Number of diffusion map components.

    Returns
    -------
    success : bool
        True if processing completed successfully.
    """
    print(f"\n{'='*60}")
    print(f"Processing subject: {subject_id}")
    print(f"{'='*60}")

    # Create subject-specific output directory
    subject_output = output_base / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)

    # Get subject's tract directory
    subject_tract_dir = base_dir / subject_id / 'diff.tract' / 'bundles'

    if not subject_tract_dir.exists():
        print(f"ERROR: Tract directory not found: {subject_tract_dir}")
        return False

    # Read bundle list
    bundles = read_bundle_list(bundle_list_file)
    print(f"Processing {len(bundles)} bundles")

    # Process each bundle
    successful = 0
    failed = []

    for bundle_name in bundles:
        # Create bundle-specific output directory
        bundle_output = subject_output / bundle_name
        bundle_output.mkdir(exist_ok=True)

        # Find bundle file
        bundle_path = subject_tract_dir / bundle_name / 'curves.vtk.gz'

        if not bundle_path.exists():
            print(f"WARNING: Bundle not found: {bundle_path}")
            failed.append(bundle_name)
            continue

        try:
            result = analyze_single_bundle(bundle_path, bundle_output,
                                          bundle_name, n_components)
            if result is not None:
                successful += 1
            else:
                failed.append(bundle_name)

        except Exception as e:
            print(f"ERROR processing {bundle_name}: {str(e)}")
            failed.append(bundle_name)

    # Write summary
    summary_file = subject_output / 'analysis_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Diffusion Maps Analysis Summary\n")
        f.write(f"Subject: {subject_id}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Successfully analyzed: {successful}/{len(bundles)} bundles\n")
        if failed:
            f.write(f"\nFailed bundles:\n")
            for bundle in failed:
                f.write(f"  - {bundle}\n")

    print(f"\n{'='*60}")
    print(f"Subject {subject_id} complete:")
    print(f"  Successful: {successful}/{len(bundles)}")
    print(f"  Summary saved to: {summary_file}")
    print(f"{'='*60}")

    return True


def main():
    """
    Main entry point for bundle analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Diffusion maps analysis for fiber tract bundles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single subject
  python bundle_analysis.py --subject TBIAA027KLW --base_dir /path/to/data

  # Specify custom bundle list
  python bundle_analysis.py --subject TBIAA027KLW \\
      --bundle_list custom_bundles.txt

  # Change number of components
  python bundle_analysis.py --subject TBIAA027KLW --n_components 5
        """
    )

    parser.add_argument("--subject", type=str, required=True,
                       help="Subject ID to process")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory containing subject data")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--bundle_list", type=str, default="bundle_list.txt",
                       help="Path to file listing bundles to analyze")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Number of diffusion map components")

    args = parser.parse_args()

    # Convert to Path objects
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    bundle_list = Path(args.bundle_list)

    # Validate inputs
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        sys.exit(1)

    if not bundle_list.exists():
        print(f"ERROR: Bundle list file not found: {bundle_list}")
        sys.exit(1)

    # Run analysis
    success = process_subject(args.subject, base_dir, output_dir,
                             bundle_list, args.n_components)

    if success:
        print("\nAnalysis completed successfully!")
        sys.exit(0)
    else:
        print("\nAnalysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
