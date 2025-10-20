A Python framework for analyzing white matter fiber tracts using diffusion maps dimensionality reduction. This toolkit characterizes individual streamlines within fiber bundles and finds their intrinsic manifold structure in latent space. 

This repository implements the Diffusion Maps–based Tractography Analysis Framework described in: 
Sharma, A., Joshi, A., & Leahy, R. (2025) **A Tractography Analysis Framework using Diffusion Maps to Study Thalamic Connectivity in Traumatic Brain Injury.**
**_Proceedings of the IEEE Engineering in Medicine and Biology Society (EMBC)_.
The method identified significant associations between diffusion map embeddings and functional outcomes (GOSE scores), revealing potential biomarkers for injury severity and recovery trajectories using the TRACK-TBI dataset. 

## Overview

This framework implements a novel approach to tractography analysis:

1. **Feature Extraction**: Characterizes individual streamlines using geometric and spatial features
2. **Diffusion Maps**: Finds low-dimensional manifold representation of fiber organization
3. **Post-hoc Analysis**: Performs clustering, correlation analysis, and validation
4. **Visualization**: Creates comprehensive plots of results

### Key Features

- Automated processing of multiple subjects and bundles
- Robust feature extraction from tractography streamlines
- Adaptive kernel bandwidth selection for diffusion maps
- Clustering analysis (K-means, DBSCAN) in embedding space
- Feature-embedding correlation analysis
- SLURM-ready batch processing scripts
- Comprehensive visualizations

## Installation

### Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- pyvista

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion-maps-tractography.git
cd diffusion-maps-tractography

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Subject Analysis

```bash
# Run diffusion maps analysis
python analysis/bundle_analysis.py \
    --subject SUBJ001 \
    --base_dir /path/to/data \
    --output_dir ./results \
    --bundle_list thalamic_bundles.txt \
    --n_components 3

# Run post-hoc analysis
python analysis/posthoc_analysis.py \
    --subject SUBJ001 \
    --results_dir ./results \
    --bundle_list thalamic_bundles.txt
```

### Batch Processing with SLURM

```bash
# Edit configuration in SLURM scripts
cd slurm

# Submit bundle analysis jobs
sbatch run_bundle_analysis.sh

# After completion, submit post-hoc analysis
sbatch run_posthoc_analysis.sh
```

## Data Structure

### Input Data Format

Your data should be organized as:

```
base_dir/
├── SUBJECT_ID_1/
│   └── diff.tract/
│       └── bundles/
│           ├── bundle_name_1/
│           │   └── curves.vtk.gz
│           ├── bundle_name_2/
│           │   └── curves.vtk.gz
│           └── ...
├── SUBJECT_ID_2/
│   └── diff.tract/
│       └── bundles/
│           └── ...
```

### Bundle List File

Create a text file listing bundle names (one per line):

```
rh_thal_dlpfc
rh_thal_dmpfc
rh_thal_occip
...
```

### Output Structure

```
results/
└── SUBJECT_ID/
    ├── analysis_summary.txt
    ├── posthoc_summary.txt
    └── bundle_name/
        ├── bundle_name_embedding.npy
        ├── bundle_name_eigenvalues.npy
        ├── bundle_name_features.npy
        ├── bundle_name_fiber_properties.png
        ├── bundle_name_diffusion_maps.png
        ├── bundle_name_embedding_2d.png
        ├── bundle_name_feature_correlations.png
        ├── bundle_name_eigenvalue_analysis.png
        └── bundle_name_posthoc_analysis.npz
```

## Methods

### Feature Extraction

For each streamline, the following 12 features are extracted:

**Geometric Features (3)**:
- **Length**: Total fiber length (mm)
- **Curvature**: Mean rate of change of tangent vectors
- **Dispersion**: Mean distance from fiber center

**Spatial Features (9)**:
- Start point coordinates (x, y, z)
- End point coordinates (x, y, z)
- Midpoint coordinates (x, y, z)

### Diffusion Maps Algorithm

1. Compute pairwise Euclidean distances between fiber features
2. Construct Gaussian kernel: K(x,y) = exp(-||x-y||²/(2ε²))
3. Normalize to create Markov transition matrix
4. Perform eigendecomposition
5. Scale eigenvectors by sqrt(eigenvalues) for embedding

**Key Parameters**:
- `n_components`: Number of embedding dimensions (default: 3)
- `epsilon`: Kernel bandwidth (default: auto-selected as median distance)

### Post-hoc Analysis

**Clustering Analysis**:
- K-means with k=2 to 7 clusters
- DBSCAN with various epsilon values
- Silhouette scores for quality assessment

**Correlation Analysis**:
- Pearson correlations between original features and embedding dimensions
- Statistical significance testing

**Eigenvalue Analysis**:
- Variance explained per component
- Cumulative variance plots

## Module Reference

### Core Modules

#### `core/diffusion_maps.py`
- `compute_diffusion_maps()`: Main diffusion maps implementation
- `compute_kernel_bandwidth()`: Adaptive bandwidth selection

#### `core/fiber_features.py`
- `extract_fiber_features()`: Extract features from streamlines
- `compute_fiber_length()`: Calculate fiber length
- `compute_fiber_curvature()`: Calculate mean curvature
- `compute_fiber_dispersion()`: Calculate dispersion
- `analyze_fiber_properties()`: Statistical summary

#### `core/io_utils.py`
- `load_streamlines()`: Load VTK/VTK.GZ files
- `read_bundle_list()`: Read bundle configuration
- `read_subject_list()`: Read subject list

### Analysis Modules

#### `analysis/bundle_analysis.py`
Main pipeline for diffusion maps analysis

#### `analysis/posthoc_analysis.py`
Post-hoc clustering and correlation analysis

### Visualization

#### `visualization/plots.py`
- `visualize_fiber_properties()`: Plot fiber statistics
- `visualize_diffusion_maps()`: Plot embedding results

## Usage Examples

### Python API

```python
from core import load_streamlines, extract_fiber_features, compute_diffusion_maps
from visualization import visualize_diffusion_maps

# Load data
fibers = load_streamlines('path/to/curves.vtk.gz')

# Extract features
features = extract_fiber_features(fibers)

# Compute diffusion maps
embedding, eigenvalues = compute_diffusion_maps(features, n_components=3)

# Visualize
visualize_diffusion_maps(embedding, eigenvalues, features,
                         output_dir='./output', bundle_name='my_bundle')
```

### Custom Analysis

```python
import numpy as np
from core import compute_diffusion_maps, extract_fiber_features

# Your custom preprocessing
fibers = preprocess_my_data()

# Extract features with custom function
features = extract_fiber_features(fibers)

# Compute embedding with custom epsilon
embedding, eigenvalues = compute_diffusion_maps(
    features,
    n_components=5,
    epsilon=0.5  # Custom bandwidth
)

# Save results
np.save('my_embedding.npy', embedding)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{diffusion_maps_tractography,
  author = {Sharma, Akul},
  title = {Diffusion Maps for Tractography Analysis},
  year = {2024},
  url = {https://github.com/akul0119/diffusion-maps-tractography}
}
```

**Diffusion Maps Reference**:

```bibtex
@article{chamberland2019dimensionality,
  title={Dimensionality reduction of diffusion MRI measures for improved tractometry of the human brain},
  author={Chamberland, Marc and Raven, Emma P and Genc, Sermin and Duffy, Kaitlin and Descoteaux, Maxime and Parker, Geoffrey D and Jones, Derek K},
  journal={NeuroImage},
  volume={200},
  pages={89--100},
  year={2019},
  publisher={Elsevier}
}
@article{chamberland2019dimensionality,
  title={Dimensionality reduction of diffusion MRI measures for improved tractometry of the human brain},
  author={Chamberland, Marc and Raven, Emma P and Genc, Sermin and Duffy, Kaitlin and Descoteaux, Maxime and Parker, Geoffrey D and Jones, Derek K},
  journal={NeuroImage},
  volume={200},
  pages={89--100},
  year={2019},
  publisher={Elsevier}
}
@article{coifman2005geometric,
  title={Geometric diffusions as a tool for harmonic analysis and structure definition of data: Diffusion maps},
  author={Coifman, Ronald R and Lafon, St{\'e}phane and Lee, Ann B and Maggioni, Mauro and Nadler, Boaz and Warner, Frederick and Zucker, Steven W},
  journal={Proceedings of the National Academy of Sciences},
  volume={102},
  number={21},
  pages={7426--7431},
  year={2005},
  publisher={National Academy of Sciences}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: akul.s19@gmail.com

## Acknowledgments

- Diffusion maps algorithm based on Coifman & Lafon (2006)

## Version History

### v1.0.0 (2024)
- Initial release
- Core diffusion maps implementation
- Feature extraction from streamlines
- Post-hoc analysis tools
- SLURM batch processing scripts
- Comprehensive visualization suite
