"""
Diffusion Maps Analysis for Tractography Data
==============================================

Core module for computing diffusion map embeddings from fiber tract features.

This module implements the diffusion maps algorithm to find low-dimensional
representations of high-dimensional fiber tract data.

Author: Akul Sharma
License: MIT
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def compute_diffusion_maps(features, n_components=3, epsilon='auto'):
    """
    Compute diffusion map embedding from feature matrix.

    The diffusion maps algorithm finds the intrinsic geometry of high-dimensional
    data by constructing a Markov transition matrix and analyzing its eigenvectors.

    Parameters
    ----------
    features : ndarray, shape (n_samples, n_features)
        Feature matrix where each row represents a fiber and each column
        represents a feature.
    n_components : int, default=3
        Number of diffusion map components to compute.
    epsilon : float or 'auto', default='auto'
        Kernel bandwidth parameter. If 'auto', uses median of pairwise distances.

    Returns
    -------
    embedding : ndarray, shape (n_samples, n_components)
        Diffusion map embedding coordinates.
    eigenvalues : ndarray, shape (n_components,)
        Eigenvalues corresponding to the embedding dimensions.

    Notes
    -----
    The algorithm follows these steps:
    1. Compute pairwise Euclidean distances
    2. Apply Gaussian kernel: K(x,y) = exp(-||x-y||^2 / (2*epsilon^2))
    3. Normalize to create Markov matrix
    4. Compute eigendecomposition
    5. Scale eigenvectors by eigenvalues for final embedding

    References
    ----------
    Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
    computational harmonic analysis, 21(1), 5-30.
    """
    # Compute pairwise distances
    distances = pdist(features)
    distances = squareform(distances)

    # Auto-select epsilon if needed
    if epsilon == 'auto':
        epsilon = np.median(distances)
        print(f"Auto-selected epsilon: {epsilon:.4f}")

    # Compute Gaussian kernel matrix
    K = np.exp(-distances**2 / (2 * epsilon**2))

    # Normalize (anisotropic diffusion)
    D = np.sum(K, axis=1)
    K_norm = K / np.sqrt(D[:, None] * D[None, :])

    # Convert to sparse matrix for efficient eigendecomposition
    K_sparse = csr_matrix(K_norm)

    # Compute eigendecomposition
    # Request n_components+1 because first eigenvector is trivial (constant)
    eigenvalues, eigenvectors = eigsh(K_sparse, k=n_components+1, which='LM')

    # Sort by eigenvalue magnitude (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][1:]  # Skip first (trivial) eigenvalue
    eigenvectors = eigenvectors[:, idx][:, 1:]  # Skip first eigenvector

    # Scale eigenvectors by sqrt of eigenvalues for embedding
    embedding = eigenvectors * np.sqrt(eigenvalues)

    return embedding, eigenvalues


def compute_kernel_bandwidth(features, method='median', percentile=50):
    """
    Compute appropriate kernel bandwidth for diffusion maps.

    Parameters
    ----------
    features : ndarray, shape (n_samples, n_features)
        Feature matrix.
    method : str, default='median'
        Method to compute bandwidth. Options: 'median', 'percentile', 'silverman'
    percentile : float, default=50
        Percentile to use if method='percentile'.

    Returns
    -------
    epsilon : float
        Suggested kernel bandwidth parameter.
    """
    distances = pdist(features)

    if method == 'median':
        epsilon = np.median(distances)
    elif method == 'percentile':
        epsilon = np.percentile(distances, percentile)
    elif method == 'silverman':
        # Silverman's rule of thumb adapted for distance matrix
        n = len(features)
        epsilon = np.std(distances) * (n ** (-1./(len(features[0]) + 4)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return epsilon
