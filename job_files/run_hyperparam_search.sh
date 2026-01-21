#!/usr/bin/env bash
# ==============================================================================
# Hyperparameter Search Script for Laptop
# ==============================================================================
# 
# This script runs hyperparameter optimization on a laptop/workstation.
#
# Usage:
#   ./run_hyperparam_search.sh benchmark    # Quick sanity check
#   ./run_hyperparam_search.sh grid         # Full grid search
#   ./run_hyperparam_search.sh binary       # Binary search for single param
#   ./run_hyperparam_search.sh help         # Show this help
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ==============================================================================
# Configuration
# ==============================================================================

# Hardware profile
HARDWARE="laptop"

# Default number of theorems per trial
NUM_THEOREMS=5

# Default parameter for binary search
BINARY_PARAM="num_workers"

# Whether to use WandB
USE_WANDB=false

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# Environment Setup
# ==============================================================================

# Try to activate conda environment
if command -v mamba &> /dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate lean-reinforcement 2>/dev/null || true
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate lean-reinforcement 2>/dev/null || true
fi

# Load environment variables
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Ensure output directories exist
mkdir -p logs hyperparam_results

# ==============================================================================
# Functions
# ==============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo " HYPERPARAMETER SEARCH - ${HARDWARE^^}"
    echo "============================================================"
    echo " Hardware: ${HARDWARE}"
    echo " Theorems per trial: ${NUM_THEOREMS}"
    echo " GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
    echo " WandB: ${USE_WANDB}"
    echo "============================================================"
    echo ""
}

run_benchmark() {
    print_header
    echo "Running quick benchmark..."
    
    python3 -m lean_reinforcement.training.hyperparam_search \
        --mode benchmark \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        ${USE_WANDB:+--use-wandb}
}

run_grid_search() {
    print_header
    echo "Running grid search..."
    echo "This may take several hours. Results will be saved to hyperparam_results/"
    
    python3 -m lean_reinforcement.training.hyperparam_search \
        --mode grid \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        ${USE_WANDB:+--use-wandb}
}

run_binary_search() {
    print_header
    echo "Running binary search for parameter: ${BINARY_PARAM}"
    
    python3 -m lean_reinforcement.training.hyperparam_search \
        --mode binary \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        --param "${BINARY_PARAM}" \
        ${USE_WANDB:+--use-wandb}
}

run_full_training() {
    print_header
    echo "Running full training with laptop-optimized settings..."
    
    python3 -m lean_reinforcement.training.train \
        --data-type novel_premises \
        --model-name "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
        --num-epochs 5 \
        --num-theorems 50 \
        --num-workers 10 \
        --mcts-type guided_rollout \
        --num-iterations 100 \
        --batch-size 16 \
        --num-tactics-to-expand 12 \
        --max-steps 40 \
        --max-time 300.0 \
        --env-timeout 180 \
        --proof-timeout 1200 \
        --train-epochs 1 \
        --save-checkpoints \
        ${USE_WANDB:+--use-wandb}
}

show_help() {
    echo "Hyperparameter Search for Theorem Proving"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  benchmark    Run quick benchmark with default settings (3-5 theorems)"
    echo "  grid         Run grid search over key parameters"
    echo "  binary       Run binary search for single parameter"
    echo "  train        Run full training with optimized settings"
    echo "  help         Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  NUM_THEOREMS     Number of theorems per trial (default: 5)"
    echo "  BINARY_PARAM     Parameter for binary search (default: num_workers)"
    echo "  USE_WANDB        Set to 'true' to enable WandB logging"
    echo "  CUDA_VISIBLE_DEVICES  GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 benchmark"
    echo "  NUM_THEOREMS=10 $0 grid"
    echo "  BINARY_PARAM=batch_size $0 binary"
    echo "  USE_WANDB=true $0 train"
}

# ==============================================================================
# Main
# ==============================================================================

# Parse command line
COMMAND="${1:-help}"

# Override defaults from environment
NUM_THEOREMS="${NUM_THEOREMS:-5}"
BINARY_PARAM="${BINARY_PARAM:-num_workers}"
USE_WANDB="${USE_WANDB:-false}"

case "${COMMAND}" in
    benchmark)
        run_benchmark
        ;;
    grid)
        run_grid_search
        ;;
    binary)
        run_binary_search
        ;;
    train)
        run_full_training
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: ${COMMAND}"
        show_help
        exit 1
        ;;
esac
