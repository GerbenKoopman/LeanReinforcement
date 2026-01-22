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
NUM_THEOREMS=50

# Default parameter for binary search
BINARY_PARAM="num_workers"

# Whether to use WandB
USE_WANDB=false

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# Environment Setup
# ==============================================================================


# Load environment variables if present
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Ensure output directories exist
mkdir -p logs hyperparam_results

# Determine how to run Python
# If CONDA_DEFAULT_ENV is set, we're already in a conda env
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    # Use Python from current environment
    PYTHON_CMD="python"
else
    # Try to use mamba/conda run
    if command -v mamba &> /dev/null; then
        PYTHON_CMD="mamba run -n lean-reinforcement python"
    elif command -v conda &> /dev/null; then
        PYTHON_CMD="conda run -n lean-reinforcement python"
    else
        PYTHON_CMD="python3"
    fi
fi

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
    echo " Python: $(python3 --version 2>&1)"
    echo "============================================================"
    echo ""
}

run_benchmark() {
    print_header
    echo "Running quick benchmark..."
    echo "(This may take 30-60 seconds to load models...)"
    echo ""
    
    ${PYTHON_CMD} -m lean_reinforcement.training.hyperparam_search \
        --mode benchmark \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        ${USE_WANDB:+--use-wandb} || {
        EXIT_CODE=$?
        echo ""
        echo "ERROR: Benchmark failed with exit code ${EXIT_CODE}"
        echo ""
        echo "Troubleshooting steps:"
        echo "  1. Verify the module is installed: python3 -m lean_reinforcement.training.hyperparam_search --help"
        echo "  2. Check dependencies: pip list | grep -E 'lean|loguru'"
        echo "  3. Check Python: python3 --version"
        exit ${EXIT_CODE}
    }
}

run_grid_search() {
    print_header
    echo "Running grid search..."
    echo "This may take several hours. Results will be saved to hyperparam_results/"
    
    ${PYTHON_CMD} -m lean_reinforcement.training.hyperparam_search \
        --mode grid \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        ${USE_WANDB:+--use-wandb}
}

run_binary_search() {
    print_header
    echo "Running binary search for parameter: ${BINARY_PARAM}"
    
    ${PYTHON_CMD} -m lean_reinforcement.training.hyperparam_search \
        --mode binary \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        --param "${BINARY_PARAM}" \
        ${USE_WANDB:+--use-wandb}
}

run_coordinate_descent() {
    print_header
    echo "Running coordinate descent search (binary search per dimension)..."
    echo "This is the most efficient search method for finding optimal hyperparameters."
    echo "Metric: proofs per hour"
    echo "Results will be saved to hyperparam_results/"
    echo ""
    echo "Parameters to optimize (in dependency order):"
    echo "  1. num_workers, batch_size (resource parameters)"
    echo "  2. num_tactics_to_expand, num_iterations (search behavior)"
    echo "  3. env_timeout, max_time, proof_timeout (timeouts)"
    echo "  4. max_steps, max_rollout_depth (search depth)"
    echo ""
    
    ${PYTHON_CMD} -m lean_reinforcement.training.hyperparam_search \
        --mode coordinate \
        --hardware "${HARDWARE}" \
        --num-theorems "${NUM_THEOREMS}" \
        --num-rounds "${NUM_ROUNDS}" \
        ${USE_WANDB:+--use-wandb}
}

run_full_training() {
    print_header
    echo "Running full training with laptop-optimized settings..."
    
    ${PYTHON_CMD} -m lean_reinforcement.training.train \
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
    echo "  grid         Run grid search over key parameters (slow, exhaustive)"
    echo "  binary       Run binary search for single parameter"
    echo "  coordinate   Run coordinate descent (RECOMMENDED - efficient, tests all params)"
    echo "  train        Run full training with optimized settings"
    echo "  help         Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  NUM_THEOREMS     Number of theorems per trial (default: 25)"
    echo "  NUM_ROUNDS       Number of rounds for coordinate descent (default: 2)"
    echo "  BINARY_PARAM     Parameter for binary search (default: num_workers)"
    echo "  USE_WANDB        Set to 'true' to enable WandB logging"
    echo "  CUDA_VISIBLE_DEVICES  GPU to use (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 benchmark"
    echo "  $0 coordinate                    # Recommended: efficient search"
    echo "  NUM_THEOREMS=10 $0 coordinate    # Quick coordinate descent"
    echo "  NUM_THEOREMS=10 $0 grid          # Grid search (slower)"
    echo "  BINARY_PARAM=batch_size $0 binary"
    echo "  USE_WANDB=true $0 train"
}

# ==============================================================================
# Main
# ==============================================================================

# Parse command line
COMMAND="${1:-help}"

# Override defaults from environment
NUM_THEOREMS="${NUM_THEOREMS:-25}"
NUM_ROUNDS="${NUM_ROUNDS:-2}"
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
    coordinate)
        run_coordinate_descent
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
