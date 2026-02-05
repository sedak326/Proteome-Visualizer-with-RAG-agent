#!/bin/bash
#SBATCH --job-name=prot5_embeddings               # Job name
#SBATCH --ntasks=1
#SBATCH --output=my_job_output.log       # Output file
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=10               # Request 20 CPUs
#SBATCH --mem=50G                       # Request 50 GB of RAM, note for Prot5 more memory might be needed
# run_all_t5_embeddings.sh
# Runs t5_embedder.py on all .fasta or .faa files in a folder

# ----------------------------
# 1️⃣ Set folder and model directory
# ----------------------------
INPUT_FOLDER="/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/fastafiles"
OUTPUT_FOLDER="/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/h5py_files_1"
MODEL_DIR="MODEL_DIR=/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/model_cache"   
# ----------------------------
# 2️⃣ Create output folder if it doesn't exist
# ----------------------------
mkdir -p "$OUTPUT_FOLDER"

# ----------------------------
# 3️⃣ Activate environment
# ----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prot5


# ----------------------------
# 4️⃣ Pick GPU
# ----------------------------
export CUDA_VISIBLE_DEVICES=1

# ----------------------------
# 5️⃣ Loop over all .fasta and .faa files
# ----------------------------
for file in "$INPUT_FOLDER"/*.{fasta,faa}; do
    # Check if any file exists
    if [ ! -f "$file" ]; then
        echo "No .fasta or .faa files found in $INPUT_FOLDER"
        break
    fi

    filename=$(basename "$file")
    output_file="$OUTPUT_FOLDER/${filename%.*}.h5"

    echo "Processing $file -> $output_file"
    python3 /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/h5py_files_1/prott5_embedder.py \
        -i "$file" \
        -o "$output_file" \
        ${MODEL_DIR:+--model "$MODEL_DIR"} \
        --per_protein 1
done

echo "All embeddings finished!"
