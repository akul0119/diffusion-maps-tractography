import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import eigs
from dipy.tracking.distances import approx_polygon_track
import pyvista as pv
import gzip
import shutil


# --------------- Configuration ---------------
# Users should set these paths accordingly
BASE_DIR = "<BASE_DIRECTORY_PATH>"  # Example: "/path/to/data"
SUBJECT_LIST_FILE = os.path.join(BASE_DIR, "subject_list.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "diff_map_output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------- Functions ---------------
def decompress_vtk_gz(file_path):
    """ Decompress a .vtk.gz file to a temporary .vtk file. """
    if file_path.endswith('.gz'):
        decompressed_path = file_path[:-3]
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_path
    return file_path


def get_streamlines(mesh):
    """ Extract streamlines from the VTK mesh. """
    points = mesh.points
    lines = mesh.lines
    streamlines = []
    i = 0
    while i < len(lines):
        num_points = lines[i]
        streamline = points[lines[i + 1:i + 1 + num_points]]
        if streamline.shape[0] > 1 and streamline.shape[1] == 3:  # Validity check
            streamlines.append(streamline)
        i += 1 + num_points
    return streamlines


def compute_features(streamlines):
    """ Compute streamline lengths and centroids. """
    lengths = [np.sum(np.sqrt(np.sum(np.diff(sl, axis=0) ** 2, axis=1))) for sl in streamlines]
    centroids = [np.mean(sl, axis=0) for sl in streamlines]
    return np.column_stack([lengths, np.array(centroids)])


def compute_similarity_matrix(features, sigma=10.0):
    """ Compute a similarity matrix using Gaussian weights. """
    distances = euclidean_distances(features)
    return np.exp(-distances ** 2 / (2 * sigma ** 2))


def diffusion_maps(similarity_matrix, n_components=3, alpha=0.5):
    """ Perform Diffusion Maps dimensionality reduction. """
    D = np.sum(similarity_matrix, axis=1)
    D_alpha = np.diag(D ** (-alpha))
    diffusion_matrix = D_alpha @ similarity_matrix @ D_alpha
    eigenvalues, eigenvectors = eigs(diffusion_matrix, k=n_components + 1, which='LR')
    return eigenvectors[:, 1:n_components + 1].real


def process_subject(subject_name, file_path, output_dir):
    """ Process a single subject. """
    print(f"Processing subject: {subject_name}")

    # Decompress and load VTK file
    decompressed_file = decompress_vtk_gz(file_path)
    mesh = pv.read(decompressed_file)
    streamlines = get_streamlines(mesh)

    # Compute features and similarity matrix
    features = compute_features(streamlines)
    similarity_matrix = compute_similarity_matrix(features)
    embedding = diffusion_maps(similarity_matrix)

    # Save outputs
    pd.DataFrame(features, columns=["Length", "Centroid_X", "Centroid_Y", "Centroid_Z"]).to_csv(
        os.path.join(output_dir, f"{subject_name}_features.csv"), index=False)
    np.save(os.path.join(output_dir, f"{subject_name}_similarity.npy"), similarity_matrix)
    np.save(os.path.join(output_dir, f"{subject_name}_diffusion_maps.npy"), embedding)

    print(f"Finished processing {subject_name}.")


# --------------- Main Execution Script ---------------
# Validate input paths
if not os.path.exists(SUBJECT_LIST_FILE):
    raise FileNotFoundError(f"Subject list file not found at {SUBJECT_LIST_FILE}")

# Load subjects
with open(SUBJECT_LIST_FILE, 'r') as file:
    subjects = [line.strip() for line in file if line.strip()]

print(f"Found {len(subjects)} subjects to process.")

# Process each subject
for subject in subjects:
    subject_file_path = os.path.join(BASE_DIR, subject, "tractography/curves.vtk.gz")
    if not os.path.exists(subject_file_path):
        print(f"Warning: File not found for subject {subject}. Skipping...")
        continue
    try:
        process_subject(subject, subject_file_path, OUTPUT_DIR)
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")

print("Processing complete.")
