#!/bin/bash

# 1. Submit the environment setup job and capture its Job ID
# --parsable makes sbatch return ONLY the job ID (e.g., 12345)
# JOB_ID_1=$(sbatch --parsable environment.job)
# echo "Submitted Environment Job: $JOB_ID_1"

# 2. Submit the training jobs with a dependency on Job 1

JOB_ID_2=$(sbatch --parsable --dependency=afterok:20900517 train_mcts_euclidean.job)
echo "Submitted Train Job: $JOB_ID_2 (Dependency: 20900517)"

JOB_ID_3=$(sbatch --parsable --dependency=afterok:20900517 train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_3 (Dependency: 20900517)"

JOB_ID_4=$(sbatch --parsable --dependency=afterok:20900517 train_ppo_euclidean.job)
echo "Submitted Train Job: $JOB_ID_4 (Dependency: 20900517)"

JOB_ID_5=$(sbatch --parsable --dependency=afterok:20900517 train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_5 (Dependency: 20900517)"
