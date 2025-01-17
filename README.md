# diff-maps
Diffusion Map Analysis for Tractography Data

This repository contains scripts for applying diffusion maps to streamline tractography data, extracting features, and performing dimensionality reduction for classification. The workflow is designed to process diffusion MRI data, compute similarity matrices, and visualize results in a streamlined manner.

Motivation

Traumatic brain injury (TBI) can lead to widespread structural changes in white matter. Diffusion MRI-based tractography provides a way to map and analyze these changes, offering insights into TBI-induced alterations. This repository implements a novel pipeline for analyzing tractography data using diffusion maps, enabling the detection of subtle patterns in streamline features that could be relevant for classification tasks.

Features

Decompresses and processes .vtk.gz tractography files.

Extracts and computes streamline features, including lengths and centroids.

Constructs similarity matrices using Gaussian weights.

Reduces data dimensionality with diffusion maps.

Visualizes streamline clusters and dimensionality-reduced embeddings.

Installation

Dependencies

The following Python libraries are required:

numpy

pandas

matplotlib

pyvista

scikit-learn

scipy

dipy

Install dependencies using:

pip install numpy pandas matplotlib pyvista scikit-learn scipy dipy

Setup

Clone this repository:

git clone https://github.com/username/diffusion-map-tractography.git
cd diffusion-map-tractography

Configure the paths:

Open the main script and update the BASE_DIR variable to point to the directory containing your data.

Ensure the subject list file (subject_list.txt) is placed in the base directory.

Create the output directory (if not automatically created):

mkdir -p <BASE_DIRECTORY_PATH>/diff_map_output

Usage

Follow these steps to process your data:

1. Prepare Data

Place the .vtk.gz files for each subject in the respective directories within BASE_DIR.

Create a subject_list.txt file in BASE_DIR, listing the names of the subjects (one per line).

2. Run the Main Script

Execute the main script to process all subjects:

python main.py

3. Outputs

The script generates the following outputs for each subject in the diff_map_output directory:

<subject_name>_features.csv: Streamline features (lengths and centroids).

<subject_name>_similarity.npy: Similarity matrix.

<subject_name>_diffusion_maps.npy: Dimensionality-reduced embeddings.

<subject_name>_lengths.png: Histogram of streamline lengths.

<subject_name>_clustering.png: Visualization of diffusion map clustering.

Repository Structure

.
├── main.py                # Main script for processing subjects
├── subject_list.txt       # List of subjects to process (example placeholder)
├── diff_map_output/       # Directory for output files
├── README.md              # Project documentation
└── requirements.txt       # Dependency list

Example Workflow

Prepare your subject list file:

Subject1
Subject2
Subject3

Organize your data:

BASE_DIR/
├── Subject1/
│   └── tractography/curves.vtk.gz
├── Subject2/
│   └── tractography/curves.vtk.gz
└── Subject3/
    └── tractography/curves.vtk.gz

Run the script:

python main.py

Outputs will be saved in diff_map_output:

diff_map_output/
├── Subject1_features.csv
├── Subject1_similarity.npy
├── Subject1_diffusion_maps.npy
├── Subject1_lengths.png
├── Subject1_clustering.png
└── ...

Acknowledgments

This repository was developed as part of ongoing research in diffusion MRI-based tractography analysis.

Special thanks to the contributors and collaborators who supported this work.

Contact

For questions or feedback, please contact:

Name: Akul Sharma

Email: akulshar@usc.edu

