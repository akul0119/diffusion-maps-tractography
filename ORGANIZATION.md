# Repository Organization

## Overview
This framework provides a complete pipeline for analyzing white matter fiber tracts using diffusion maps dimensionality reduction.

## Directory Structure

```
diffusion-maps-tractography/
├── core/                      # Core functionality
│   ├── __init__.py
│   ├── diffusion_maps.py      # Diffusion maps algorithm
│   ├── fiber_features.py      # Feature extraction from streamlines
│   └── io_utils.py            # Data loading utilities
│
├── analysis/                  # Analysis pipelines
│   ├── bundle_analysis.py     # Main diffusion maps analysis
│   └── posthoc_analysis.py    # Clustering & correlation analysis
│
├── visualization/             # Plotting functions
│   ├── __init__.py
│   └── plots.py               # Visualization utilities
│
├── slurm/                     # SLURM batch processing
│   ├── run_bundle_analysis.sh # Main analysis SLURM script
│   └── run_posthoc_analysis.sh # Post-hoc SLURM script
│
├── examples/                  # Example files and usage
│   ├── example_usage.py       # Python API examples
│   ├── thalamic_bundles.txt   # Example bundle list
│   └── subjects.txt           # Example subject list
│
├── docs/                      # Documentation
│   └── QUICKSTART.md          # Quick start guide
│
├── README.md                  # Main documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore patterns
└── ORGANIZATION.md           # This file
```

## Module Descriptions

### Core Modules

**core/diffusion_maps.py**
- `compute_diffusion_maps()`: Main algorithm implementation
- `compute_kernel_bandwidth()`: Adaptive epsilon selection
- Uses sparse eigendecomposition for efficiency

**core/fiber_features.py**
- `extract_fiber_features()`: Extract 12-dimensional features
- `compute_fiber_length()`: Fiber length calculation
- `compute_fiber_curvature()`: Curvature estimation
- `compute_fiber_dispersion()`: Dispersion metric
- `analyze_fiber_properties()`: Statistical summary

**core/io_utils.py**
- `load_streamlines()`: Load VTK/VTK.GZ files
- `decompress_vtk_gz()`: Handle compressed files
- `read_bundle_list()`: Parse bundle configuration
- `read_subject_list()`: Parse subject list

### Analysis Pipelines

**analysis/bundle_analysis.py**
- Complete pipeline for single subject
- Processes multiple bundles automatically
- Generates visualizations and saves results
- Command-line interface with argparse
- Can be run standalone or via SLURM

**analysis/posthoc_analysis.py**
- Clustering analysis (K-means, DBSCAN)
- Feature-embedding correlations
- Eigenvalue spectrum analysis
- Quality metrics (silhouette scores)
- Runs on existing diffusion maps results

### Visualization

**visualization/plots.py**
- `visualize_fiber_properties()`: Histograms of length, curvature, dispersion
- `visualize_diffusion_maps()`: 3D and 2D embedding plots
- Color-coded by features
- Eigenvalue spectra
- High-quality publication-ready figures

### SLURM Scripts

**slurm/run_bundle_analysis.sh**
- SLURM array job for batch processing
- Processes subjects in parallel
- Configurable resources (CPU, memory, time)
- Automatic logging

**slurm/run_posthoc_analysis.sh**
- Post-hoc analysis in parallel
- Runs after main analysis completes
- Lower resource requirements

## Workflow

### Standard Analysis Pipeline

1. **Data Preparation**
   - Organize tractography data in standard structure
   - Create bundle list file
   - Create subject list file

2. **Main Analysis** (`bundle_analysis.py`)
   - Load streamlines from VTK files
   - Extract geometric and spatial features
   - Compute diffusion maps embedding
   - Generate visualizations
   - Save numerical results

3. **Post-hoc Analysis** (`posthoc_analysis.py`)
   - Load embedding results
   - Perform clustering analysis
   - Calculate feature correlations
   - Analyze eigenvalue spectrum
   - Save additional visualizations

4. **Results Inspection**
   - Review summary files
   - Examine plots
   - Load .npy files for further analysis

### Execution Modes

**Single Subject (Local)**
```bash
python analysis/bundle_analysis.py --subject SUBJ001 --base_dir ./data
python analysis/posthoc_analysis.py --subject SUBJ001 --results_dir ./results
```

**Batch Processing (SLURM)**
```bash
sbatch slurm/run_bundle_analysis.sh
sbatch slurm/run_posthoc_analysis.sh
```

**Python API**
```python
from core import load_streamlines, extract_fiber_features, compute_diffusion_maps
# See examples/example_usage.py for full examples
```

## Data Flow

```
Input: curves.vtk.gz (tractography)
    ↓
Load Streamlines (io_utils)
    ↓
Extract Features (fiber_features)
    → 12D feature vector per fiber
    ↓
Compute Diffusion Maps (diffusion_maps)
    → 3D embedding coordinates
    ↓
Visualize & Save Results (visualization, analysis)
    ↓
Post-hoc Analysis (posthoc_analysis)
    → Clustering, correlations, metrics
    ↓
Output: .npy files, .png plots, summary .txt
```

## Output Files

For each bundle:
- `*_embedding.npy`: Diffusion map coordinates (n_fibers, n_components)
- `*_eigenvalues.npy`: Eigenvalues from decomposition
- `*_features.npy`: Original feature matrix (n_fibers, 12)
- `*_fiber_metrics.npy`: Raw length, curvature, dispersion values
- `*_fiber_properties.png`: Property distribution plots
- `*_diffusion_maps.png`: 3D embedding visualizations
- `*_embedding_2d.png`: 2D pairwise plots
- `*_feature_correlations.png`: Heatmap of correlations
- `*_eigenvalue_analysis.png`: Variance explained plots
- `*_posthoc_analysis.npz`: Comprehensive post-hoc results

## Customization

### Adding New Features
Edit `core/fiber_features.py`:
```python
def extract_fiber_features(fibers):
    # Add your custom features
    custom_feature = compute_my_feature(fiber)
    feature_vector = np.concatenate([
        [length, curvature, dispersion, custom_feature],
        start_point, end_point, mid_point
    ])
```

### Changing Diffusion Maps Parameters
```python
embedding, eigenvalues = compute_diffusion_maps(
    features,
    n_components=5,      # More dimensions
    epsilon=0.5          # Custom bandwidth
)
```

### Custom Visualizations
Use saved .npy files:
```python
embedding = np.load('bundle_embedding.npy')
features = np.load('bundle_features.npy')
# Create your own plots
```

## Dependencies

**Required:**
- numpy (≥1.20.0): Array operations
- scipy (≥1.7.0): Sparse eigendecomposition
- scikit-learn (≥1.0.0): Clustering algorithms
- matplotlib (≥3.4.0): Plotting
- seaborn (≥0.11.0): Statistical plots
- pyvista (≥0.32.0): VTK file reading

**Optional:**
- jupyter: Interactive analysis
- pandas: Data organization

## Best Practices

1. **Start Small**: Test with single subject before batch processing
2. **Check Data**: Verify VTK files load correctly
3. **Monitor Resources**: Watch memory usage for large bundles
4. **Save Intermediate**: Keep feature matrices for debugging
5. **Document Parameters**: Record epsilon and n_components used
6. **Version Control**: Track analysis versions and parameters

## Troubleshooting

**Empty fiber lists:**
- Check VTK file format
- Verify file paths
- Test with decompress_vtk_gz() manually

**Memory errors:**
- Reduce n_components
- Process fewer bundles at once
- Increase SLURM memory allocation

**Poor embeddings:**
- Try different epsilon values
- Check feature scaling
- Increase n_components
- Verify sufficient fibers (>100 recommended)

## Future Enhancements

Potential additions:
- Group-level analysis across subjects
- Supervised learning on embeddings
- Integration with graph neural networks
- Real-time visualization tools
- Configuration file support (YAML/JSON)
- Multi-scale analysis
- Temporal dynamics tracking

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Follow existing code style
4. Add tests for new features
5. Update documentation
6. Submit pull request

## Citation

When using this framework, cite both the software and the diffusion maps paper:
- Software: See README.md citation section
- Method: Coifman & Lafon (2006) - Diffusion maps
