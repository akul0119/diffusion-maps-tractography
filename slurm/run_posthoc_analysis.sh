#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-110%20
#SBATCH --output=logs/posthoc_%A_%a.out
#SBATCH --error=logs/posthoc_%A_%a.err

#================================================================
# Post-hoc Analysis - SLURM Array Job
#================================================================
# This script runs post-hoc analysis (clustering, correlations)
# on diffusion maps results.
#
# Run this AFTER bundle_analysis has completed.
#================================================================

set -e

# Configuration
RESULTS_DIR="/path/to/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../analysis"
SUBJECTS_LIST="subjects.txt"
BUNDLE_LIST="thalamic_bundles.txt"

mkdir -p logs

# Get subject ID
SUBJECT_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${SUBJECTS_LIST})

echo "=========================================="
echo "Post-hoc analysis for: ${SUBJECT_ID}"
echo "=========================================="

# Run post-hoc analysis
python ${SCRIPT_DIR}/posthoc_analysis.py \
    --subject ${SUBJECT_ID} \
    --results_dir ${RESULTS_DIR} \
    --bundle_list ${BUNDLE_LIST}

echo "=========================================="
echo "Completed: ${SUBJECT_ID}"
echo "=========================================="
