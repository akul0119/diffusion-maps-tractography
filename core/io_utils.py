"""
Input/Output Utilities for Tractography Data
=============================================

Module for loading and saving tractography data in various formats.

Author: Akul Sharma
License: MIT
"""

import os
import gzip
import shutil
import pyvista as pv
from pathlib import Path


def decompress_vtk_gz(file_path):
    """
    Decompress a .vtk.gz file to a temporary .vtk file.

    Parameters
    ----------
    file_path : str or Path
        Path to the .vtk.gz file.

    Returns
    -------
    decompressed_path : str
        Path to the decompressed .vtk file.
    """
    file_path = str(file_path)
    if file_path.endswith('.gz'):
        decompressed_path = file_path[:-3]
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_path
    return file_path


def get_streamlines(mesh):
    """
    Extract streamlines from VTK mesh object.

    Parameters
    ----------
    mesh : pyvista.PolyData
        PyVista mesh object containing streamline data.

    Returns
    -------
    streamlines : list of ndarray
        List of streamlines, each as array of shape (n_points, 3).
    """
    points = mesh.points
    lines = mesh.lines

    streamlines = []
    i = 0
    while i < len(lines):
        num_points = lines[i]
        streamline = points[lines[i + 1:i + 1 + num_points]]

        # Validity check
        if streamline.shape[0] > 1 and streamline.shape[1] == 3:
            streamlines.append(streamline)

        i += 1 + num_points

    return streamlines


def load_streamlines(vtk_file_path):
    """
    Load streamlines from VTK or VTK.GZ file.

    Parameters
    ----------
    vtk_file_path : str or Path
        Path to VTK or VTK.GZ file containing tractography data.

    Returns
    -------
    streamlines : list of ndarray
        List of streamlines, each as array of shape (n_points, 3).
        Returns None if loading fails.

    Examples
    --------
    >>> fibers = load_streamlines('path/to/curves.vtk.gz')
    >>> print(f"Loaded {len(fibers)} fibers")
    """
    try:
        # Decompress if needed
        decompressed_path = decompress_vtk_gz(vtk_file_path)

        # Read the mesh using pyvista
        mesh = pv.read(decompressed_path)

        # Get streamlines
        streamlines = get_streamlines(mesh)

        # Clean up decompressed file if it was created
        if decompressed_path != str(vtk_file_path):
            os.remove(decompressed_path)

        return streamlines

    except Exception as e:
        print(f"Error loading streamlines from {vtk_file_path}: {str(e)}")
        return None


def read_bundle_list(bundle_list_file):
    """
    Read list of bundle names from text file.

    Parameters
    ----------
    bundle_list_file : str or Path
        Path to text file with one bundle name per line.

    Returns
    -------
    bundles : list of str
        List of bundle names.
    """
    with open(bundle_list_file, 'r') as f:
        bundles = [line.strip() for line in f.readlines() if line.strip()]
    return bundles


def read_subject_list(subject_list_file):
    """
    Read list of subject IDs from text file.

    Parameters
    ----------
    subject_list_file : str or Path
        Path to text file with one subject ID per line.

    Returns
    -------
    subjects : list of str
        List of subject IDs.
    """
    with open(subject_list_file, 'r') as f:
        subjects = [line.strip() for line in f.readlines() if line.strip()]
    return subjects
