#!/bin/bash
# Runs the full GLM pipeline across all four outcomes and both prompting conditions.
# Writes to ../parameter_estimates/ and ../plots/.
#
# Usage: bash run_pipeline.sh
set -e

for dataset in misinfo trust private extreme; do
    for prompting in 0 1; do

        echo "============================================"
        echo "Running $dataset prompting=$prompting: $(date)"
        echo "============================================"
        # The variant retaining the sub_compliance covariate:
        # python data_analysis_pipeline.py --dataset $dataset --prompting $prompting
        python data_analysis_pipeline_no_compliance.py --dataset $dataset --prompting $prompting

        # Each run is a separate process that releases its GPU memory on exit, so no
        # cleanup is needed here. An earlier version of this script killed every GPU
        # process reported by nvidia-smi, which on a shared machine would have killed
        # other users' jobs.
        sleep 10

        echo "GPU status:"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "  (no GPU)"
        echo ""
    done
done
echo "All complete: $(date)"
