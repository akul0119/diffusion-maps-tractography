# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diffusion-maps-tractography.git
cd diffusion-maps-tractography

# Install dependencies
pip install -r requirements.txt
```

## 1. Prepare Your Data

Organize your data as:
```
data/
├── SUBJECT_ID/
│   └── diff.tract/
│       └── bundles/
│           └── bundle_name/
│               └── curves.vtk.gz
```

Create a bundle list (`thalamic_bundles.txt`):
```
rh_thal_dlpfc
rh_thal_dmpfc
...
```

## 2. Run Analysis

### Single Subject

```bash
python analysis/bundle_analysis.py \
    --subject TBIAA027KLW \
    --base_dir ./data \
    --output_dir ./results \
    --bundle_list thalamic_bundles.txt
```

### Post-hoc Analysis

```bash
python analysis/posthoc_analysis.py \
    --subject TBIAA027KLW \
    --results_dir ./results \
    --bundle_list thalamic_bundles.txt
```

## 3. Batch Processing (SLURM)

Edit `slurm/run_bundle_analysis.sh`:
```bash
BASE_DIR="/path/to/your/data"
OUTPUT_DIR="/path/to/results"
```

Submit jobs:
```bash
sbatch slurm/run_bundle_analysis.sh
```

## 4. View Results

Results are saved in:
```
results/SUBJECT_ID/bundle_name/
├── bundle_name_embedding.npy           # Latent space coordinates
├── bundle_name_eigenvalues.npy         # Eigenvalues
├── bundle_name_features.npy            # Original features
├── bundle_name_fiber_properties.png    # Property histograms
├── bundle_name_diffusion_maps.png      # 3D embedding plots
└── bundle_name_feature_correlations.png # Feature-dimension correlations
```

## Common Issues

### ImportError for pyvista
```bash
pip install pyvista
```

### SLURM module not found
Edit SLURM scripts to load modules:
```bash
module load python/3.9
```

### Out of memory
Reduce number of components or increase memory allocation:
```bash
#SBATCH --mem=64G
```

## Next Steps

- See `examples/example_usage.py` for Python API usage
- Read full documentation in `README.md`
- Customize feature extraction in `core/fiber_features.py`
