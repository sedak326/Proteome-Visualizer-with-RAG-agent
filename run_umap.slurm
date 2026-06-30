#!/bin/bash
#SBATCH --job-name=proteome_umap
#SBATCH --nodelist=gpu
#SBATCH --partition=ws-gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/umap_%j.log
#SBATCH --error=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/logs/umap_%j.err

PYTHON=/home/skavlak/miniconda3/envs/autorag/bin/python
WORKDIR=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG

# cuml (RAPIDS) pip-installs nvidia libs under site-packages but does not add them
# to LD_LIBRARY_PATH automatically — this line fixes the missing libnvrtc.so.12 error.
NVIDIA_SITE=/home/skavlak/miniconda3/envs/autorag/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$(find "$NVIDIA_SITE" -name '*.so*' -path '*/lib*' \
    | sed 's|/[^/]*\.so.*||' | sort -u | tr '\n' ':')${LD_LIBRARY_PATH}"

mkdir -p "$WORKDIR/logs"

echo "Starting combined UMAP on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd "$WORKDIR"

echo "=== Step 1: UMAP ==="
$PYTHON create_combined_umap.py

echo "=== Step 2: K-means recluster + cluster naming ==="
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PYTHON recluster_kmeans.py

echo "=== Step 3: Species enrichment ==="
$PYTHON compute_enrichment.py

echo "=== Step 4: Group enrichment ==="
$PYTHON compute_group_enrichment.py

echo "=== Step 5: Cluster quality ==="
$PYTHON compute_cluster_quality.py

echo "All done."
