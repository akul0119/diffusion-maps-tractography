"""
Fiber Tract Feature Extraction
===============================

Module for extracting geometric and spatial features from fiber streamlines.

Author: Akul Sharma
License: MIT
"""

import numpy as np


def extract_fiber_features(fibers):
    """
    Extract comprehensive features from fiber streamlines for diffusion map analysis.

    Parameters
    ----------
    fibers : list of ndarray
        List of fiber streamlines, where each fiber is an array of shape (n_points, 3)
        representing 3D coordinates along the streamline.

    Returns
    -------
    features : ndarray, shape (n_fibers, 12)
        Feature matrix with the following columns:
        - [0] Length: Total fiber length
        - [1] Curvature: Mean curvature along fiber
        - [2] Dispersion: Mean distance from fiber center
        - [3:6] Start point (x, y, z)
        - [6:9] End point (x, y, z)
        - [9:12] Midpoint (x, y, z)

    Notes
    -----
    Features are designed to capture both geometric properties (length, curvature,
    dispersion) and spatial embedding (endpoints, midpoint) of each fiber.
    """
    features = []

    for fiber in fibers:
        # Geometric features
        length = compute_fiber_length(fiber)
        curvature = compute_fiber_curvature(fiber)
        dispersion = compute_fiber_dispersion(fiber)

        # Spatial features
        start_point = fiber[0]
        end_point = fiber[-1]
        mid_point = np.mean(fiber, axis=0)

        # Combine features
        feature_vector = np.concatenate([
            [length, curvature, dispersion],
            start_point, end_point, mid_point
        ])
        features.append(feature_vector)

    return np.array(features)


def compute_fiber_length(fiber):
    """
    Compute total length of a fiber streamline.

    Parameters
    ----------
    fiber : ndarray, shape (n_points, 3)
        Fiber streamline coordinates.

    Returns
    -------
    length : float
        Total length in millimeters.
    """
    segments = np.diff(fiber, axis=0)
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    return np.sum(segment_lengths)


def compute_fiber_curvature(fiber):
    """
    Compute mean curvature along a fiber streamline.

    Curvature is approximated as the rate of change of tangent vectors.

    Parameters
    ----------
    fiber : ndarray, shape (n_points, 3)
        Fiber streamline coordinates.

    Returns
    -------
    curvature : float
        Mean curvature value.
    """
    if len(fiber) < 3:
        return 0.0

    # Compute tangent vectors
    tangents = np.diff(fiber, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1)

    # Normalize tangents (avoid division by zero)
    tangent_norms[tangent_norms == 0] = 1e-10
    tangents_normalized = tangents / tangent_norms[:, None]

    # Curvature as rate of change of tangent direction
    tangent_changes = np.diff(tangents_normalized, axis=0)
    curvatures = np.sqrt(np.sum(tangent_changes**2, axis=1))

    return np.mean(curvatures)


def compute_fiber_dispersion(fiber):
    """
    Compute dispersion of a fiber (mean distance from fiber center).

    Parameters
    ----------
    fiber : ndarray, shape (n_points, 3)
        Fiber streamline coordinates.

    Returns
    -------
    dispersion : float
        Mean distance from fiber center in millimeters.
    """
    center = np.mean(fiber, axis=0)
    distances = np.linalg.norm(fiber - center, axis=1)
    return np.mean(distances)


def analyze_fiber_properties(fibers):
    """
    Analyze basic statistical properties of fiber bundle.

    Parameters
    ----------
    fibers : list of ndarray
        List of fiber streamlines.

    Returns
    -------
    stats : dict
        Dictionary containing statistical summaries:
        - n_fibers: Number of fibers
        - length_mean, length_std: Length statistics
        - curvature_mean, curvature_std: Curvature statistics
        - dispersion_mean, dispersion_std: Dispersion statistics
        - bundle_center: 3D center of mass
        - bundle_extent: Spatial extent in each dimension
    """
    # Calculate properties for all fibers
    fiber_lengths = [compute_fiber_length(fiber) for fiber in fibers]
    curvatures = [compute_fiber_curvature(fiber) for fiber in fibers]
    dispersions = [compute_fiber_dispersion(fiber) for fiber in fibers]

    # Bundle-level statistics
    bundle_center = np.mean([np.mean(fiber, axis=0) for fiber in fibers], axis=0)
    all_points = np.vstack(fibers)
    bundle_extent = np.ptp(all_points, axis=0)  # Range in each dimension

    stats = {
        'n_fibers': len(fibers),
        'length_mean': np.mean(fiber_lengths),
        'length_std': np.std(fiber_lengths),
        'curvature_mean': np.mean(curvatures),
        'curvature_std': np.std(curvatures),
        'dispersion_mean': np.mean(dispersions),
        'dispersion_std': np.std(dispersions),
        'bundle_center': bundle_center,
        'bundle_extent': bundle_extent
    }

    return stats, fiber_lengths, curvatures, dispersions
