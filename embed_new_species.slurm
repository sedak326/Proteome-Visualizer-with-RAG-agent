#!/bin/bash
#SBATCH --job-name=proteome_embed
#SBATCH --nodelist=gpu2
#SBATCH --partition=ws-gpu2
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/embed_%j.log
#SBATCH --error=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/embed_%j.err

mkdir -p /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python

echo "Starting ProtT5 embeddings on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"

cd /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG

$PYTHON embed_new_species.py

echo "Embeddings complete."
