#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-110%20
#SBATCH --output=logs/bundle_%A_%a.out
#SBATCH --error=logs/bundle_%A_%a.err

#================================================================
# Diffusion Maps Bundle Analysis - SLURM Array Job
#================================================================
# This script runs diffusion maps analysis for individual subjects
# as part of a SLURM array job.
#
# Usage:
#   sbatch run_bundle_analysis.sh
#
# Configuration:
#   - Edit BASE_DIR to point to your data directory
#   - Edit OUTPUT_DIR for results location
#   - Edit SUBJECTS_LIST for your subject list file
#   - Adjust --array parameter for your number of subjects
#================================================================

# Exit on error
set -e

# Configuration
BASE_DIR="/path/to/data"
OUTPUT_DIR="/path/to/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../analysis"
SUBJECTS_LIST="subjects.txt"
BUNDLE_LIST="thalamic_bundles.txt"

# Create logs directory if it doesn't exist
mkdir -p logs

# Get subject ID for this array task
SUBJECT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${SUBJECTS_LIST})

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing subject: ${SUBJECT_ID}"
echo "=========================================="

# Load required modules (adjust for your HPC environment)
# module load python/3.9
# module load gcc

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Run the analysis
python ${SCRIPT_DIR}/bundle_analysis.py \
    --subject ${SUBJECT_ID} \
    --base_dir ${BASE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --bundle_list ${BUNDLE_LIST} \
    --n_components 3

echo "=========================================="
echo "Completed: ${SUBJECT_ID}"
echo "=========================================="
