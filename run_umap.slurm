#!/bin/bash
#SBATCH --job-name=proteome_umap
#SBATCH --nodelist=gpu2
#SBATCH --partition=ws-gpu2
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/umap_%j.log
#SBATCH --error=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/umap_%j.err

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python

echo "Starting combined UMAP on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG

$PYTHON create_combined_umap.py

echo "UMAP complete."
