#!/bin/bash
# Start both the RAG API and the Proteome Explorer

source ~/miniconda3/etc/profile.d/conda.sh
conda activate autorag

echo "Starting RAG API on port 8000..."
autorag run_api --trial_dir /home/skavlak/AI_Agent/evals/eval_run_1/0 --port 8000 &
RAG_PID=$!

# Wait for the API to be ready
echo "Waiting for RAG API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/version > /dev/null 2>&1; then
        echo "RAG API is ready!"
        break
    fi
    sleep 1
done

echo "Starting Proteome Explorer on port 7860..."
python /home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG/proteome_explorer_ux2.py

# When Dash exits, also kill the RAG API
kill $RAG_PID 2>/dev/null
