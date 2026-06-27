#!/bin/bash
set -e

echo "=== BRONZE / SILVER / GOLD ==="
for date in 2023-01-01 2023-02-01 2023-03-01 2023-04-01 2023-05-01 2023-06-01 \
            2023-07-01 2023-08-01 2023-09-01 2023-10-01 2023-11-01 2023-12-01 \
            2024-01-01; do
    echo ">>> main.py $date"
    python main.py --snapshotdate $date
done

echo "=== MODEL TRAINING ==="
python scripts/model_training.py --snapshotdate 2024-01-01

echo "=== MODEL INFERENCE ==="
python scripts/model_inference.py --snapshotdate 2024-01-01

echo "=== MODEL MONITORING ==="
python scripts/model_monitoring.py --snapshotdate 2024-01-01

echo "=== PIPELINE COMPLETE ==="
