#!/bin/bash
#SBATCH --job-name=anim_gen
#SBATCH --nodelist=gpu2
#SBATCH --partition=ws-gpu2
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/anim_%j.log
#SBATCH --error=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/anim_%j.err

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python

echo "Starting animation generation on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG

$PYTHON generate_new_animations.py

echo "Animation generation complete."
