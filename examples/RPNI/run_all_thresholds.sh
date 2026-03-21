#!/bin/bash
# Run baseline experiments with different accuracy_thresholds
# Usage: bash examples/RPNI/run_all_thresholds.sh

set -e  # Exit if any command fails

cd /home/yihua/anchor-llm

echo "========================================================================"
echo "  Running baseline experiments with different accuracy_thresholds"
echo "========================================================================"

# Run with accuracy_threshold=0.8
echo -e "\n\n❯ Running experiment with accuracy_threshold=0.8 ..."
python examples/RPNI/run_baseline_experiment_parameterized.py --accuracy_threshold 0.8

echo -e "\n✓ Experiment with accuracy_threshold=0.8 completed!"
echo -e "  Results saved to: test_result/baseline_experiment_threshold_0.8/"

# Run with accuracy_threshold=0.9
echo -e "\n\n❯ Running experiment with accuracy_threshold=0.9 ..."
python examples/RPNI/run_baseline_experiment_parameterized.py --accuracy_threshold 0.9

echo -e "\n✓ Experiment with accuracy_threshold=0.9 completed!"
echo -e "  Results saved to: test_result/baseline_experiment_threshold_0.9/"

# Summary
echo -e "\n\n========================================================================"
echo "  All experiments completed!"
echo "========================================================================"
echo ""
echo "Results locations:"
echo "  • accuracy_threshold=0.8  → test_result/baseline_experiment_threshold_0.8/"
echo "  • accuracy_threshold=0.9  → test_result/baseline_experiment_threshold_0.9/"
echo ""
echo "Files in each directory:"
echo "  • results.csv            - Detailed results table"
echo "  • experiment_log.txt     - Full experiment log"
echo "  • {L3AB,L4,L6,L7,mnist}/ - Per-language results and artifacts"
echo ""
