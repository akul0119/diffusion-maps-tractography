"""
Post-hoc Analysis for Diffusion Maps Results
============================================

Performs clustering, correlation analysis, and validation on diffusion map embeddings.

Author: Akul Sharma
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import read_bundle_list


def analyze_feature_correlations(features, embedding, bundle_name):
    """
    Analyze correlations between original features and embedding dimensions.

    Parameters
    ----------
    features : ndarray
        Original feature matrix (n_fibers, 12).
    embedding : ndarray
        Diffusion map embedding (n_fibers, n_components).
    bundle_name : str
        Bundle name for plot title.

    Returns
    -------
    correlations : ndarray
        Correlation matrix (n_features, n_components).
    p_values : ndarray
        P-values for correlations.
    """
    feature_names = ['Length', 'Curvature', 'Dispersion',
                     'Start_x', 'Start_y', 'Start_z',
                     'End_x', 'End_y', 'End_z',
                     'Mid_x', 'Mid_y', 'Mid_z']

    correlations = np.zeros((len(feature_names), embedding.shape[1]))
    p_values = np.zeros_like(correlations)

    for i, name in enumerate(feature_names):
        for j in range(embedding.shape[1]):
            corr, p_val = pearsonr(features[:, i], embedding[:, j])
            correlations[i, j] = corr
            p_values[i, j] = p_val

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=[f'Dim {i+1}' for i in range(embedding.shape[1])],
                yticklabels=feature_names, cbar_kws={'label': 'Pearson R'})
    plt.title(f'{bundle_name}\nFeature-Embedding Correlations')
    plt.tight_layout()

    return correlations, p_values


def analyze_clustering(embedding, bundle_name, k_range=range(2, 8)):
    """
    Analyze clustering quality in embedding space.

    Parameters
    ----------
    embedding : ndarray
        Diffusion map embedding.
    bundle_name : str
        Bundle name for reporting.
    k_range : range
        Range of cluster numbers to test for k-means.

    Returns
    -------
    results : dict
        Dictionary containing clustering results.
    """
    results = {}

    print(f"\nClustering analysis for {bundle_name}:")

    # K-means analysis
    silhouette_scores = []
    best_k = 2
    best_score = -1

    for n_clusters in k_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)
        score = silhouette_score(embedding, labels)
        silhouette_scores.append(score)

        print(f"  K={n_clusters}: silhouette={score:.3f}")

        if score > best_score:
            best_score = score
            best_k = n_clusters

    print(f"  Best K={best_k} with score={best_score:.3f}")

    results['kmeans'] = {
        'n_clusters_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'best_k': best_k,
        'best_score': best_score
    }

    # DBSCAN analysis
    eps_range = np.linspace(0.0001, 0.001, 10)
    dbscan_scores = []
    n_clusters_found = []

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(embedding)
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_clusters_found.append(n_clusters)

        if n_clusters > 1 and len(np.unique(labels)) > 1:
            try:
                score = silhouette_score(embedding, labels)
                dbscan_scores.append(score)
            except:
                dbscan_scores.append(0)
        else:
            dbscan_scores.append(0)

    results['dbscan'] = {
        'eps_range': eps_range.tolist(),
        'n_clusters': n_clusters_found,
        'scores': dbscan_scores
    }

    return results


def analyze_eigenvalue_spectrum(eigenvalues, bundle_name):
    """
    Detailed analysis of eigenvalue spectrum.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues from diffusion maps.
    bundle_name : str
        Bundle name for plot title.

    Returns
    -------
    relative_importance : ndarray
        Proportion of variance explained by each component.
    cumulative_importance : ndarray
        Cumulative variance explained.
    """
    relative_importance = eigenvalues / np.sum(eigenvalues)
    cumulative_importance = np.cumsum(relative_importance)

    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.bar(range(1, len(eigenvalues) + 1), relative_importance * 100)
    plt.xlabel('Component')
    plt.ylabel('Variance Explained (%)')
    plt.title(f'{bundle_name}\nVariance per Component')
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(122)
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_importance * 100, 'ro-', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.title('Cumulative Variance')
    plt.grid(alpha=0.3)

    plt.tight_layout()

    return relative_importance, cumulative_importance


def process_bundle(bundle_dir):
    """
    Run post-hoc analysis for a single bundle.

    Parameters
    ----------
    bundle_dir : Path
        Directory containing bundle results.

    Returns
    -------
    success : bool
        True if analysis completed successfully.
    """
    bundle_name = bundle_dir.name

    try:
        # Load data
        embedding = np.load(bundle_dir / f'{bundle_name}_embedding.npy')
        eigenvalues = np.load(bundle_dir / f'{bundle_name}_eigenvalues.npy')
        features = np.load(bundle_dir / f'{bundle_name}_features.npy')

        print(f"\nProcessing bundle: {bundle_name}")

        # Feature correlations
        correlations, p_values = analyze_feature_correlations(features, embedding, bundle_name)
        plt.savefig(bundle_dir / f'{bundle_name}_feature_correlations.png', dpi=150)
        plt.close()

        # Clustering analysis
        clustering_results = analyze_clustering(embedding, bundle_name)

        # Eigenvalue analysis
        rel_importance, cum_importance = analyze_eigenvalue_spectrum(eigenvalues, bundle_name)
        plt.savefig(bundle_dir / f'{bundle_name}_eigenvalue_analysis.png', dpi=150)
        plt.close()

        # Save results
        np.savez(bundle_dir / f'{bundle_name}_posthoc_analysis.npz',
                correlations=correlations,
                p_values=p_values,
                relative_importance=rel_importance,
                cumulative_importance=cum_importance,
                clustering_results=clustering_results)

        print(f"Post-hoc analysis complete for {bundle_name}")
        return True

    except Exception as e:
        print(f"ERROR analyzing {bundle_name}: {str(e)}")
        return False


def process_subject(subject_id, results_dir, bundle_list_file):
    """
    Process post-hoc analysis for all bundles of a subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    results_dir : Path
        Base results directory.
    bundle_list_file : Path
        Path to bundle list file.

    Returns
    -------
    success : bool
        True if processing completed successfully.
    """
    print(f"\n{'='*60}")
    print(f"Post-hoc analysis for subject: {subject_id}")
    print(f"{'='*60}")

    subject_dir = results_dir / subject_id
    if not subject_dir.exists():
        print(f"ERROR: Results directory not found: {subject_dir}")
        return False

    # Read bundle list
    bundles = read_bundle_list(bundle_list_file)
    print(f"Processing {len(bundles)} bundles")

    successful = 0
    failed = []

    for bundle_name in bundles:
        bundle_dir = subject_dir / bundle_name
        if not bundle_dir.exists():
            print(f"WARNING: Bundle directory not found: {bundle_dir}")
            failed.append(bundle_name)
            continue

        if process_bundle(bundle_dir):
            successful += 1
        else:
            failed.append(bundle_name)

    # Write summary
    summary_file = subject_dir / 'posthoc_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Post-hoc Analysis Summary\n")
        f.write(f"Subject: {subject_id}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Successfully analyzed: {successful}/{len(bundles)} bundles\n")
        if failed:
            f.write(f"\nFailed bundles:\n")
            for bundle in failed:
                f.write(f"  - {bundle}\n")

    print(f"\n{'='*60}")
    print(f"Post-hoc analysis complete for {subject_id}")
    print(f"  Successful: {successful}/{len(bundles)}")
    print(f"{'='*60}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of diffusion maps results"
    )

    parser.add_argument("--subject", type=str, required=True,
                       help="Subject ID to process")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory containing diffusion maps results")
    parser.add_argument("--bundle_list", type=str, default="bundle_list.txt",
                       help="Path to bundle list file")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    bundle_list = Path(args.bundle_list)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    if not bundle_list.exists():
        print(f"ERROR: Bundle list file not found: {bundle_list}")
        sys.exit(1)

    success = process_subject(args.subject, results_dir, bundle_list)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
