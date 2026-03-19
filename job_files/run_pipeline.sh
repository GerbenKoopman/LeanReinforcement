#!/bin/bash

JOB_ID_1=$(sbatch --parsable cython.job)
echo "Submitted Cython Job: $JOB_ID_1"

# 2. Submit the training jobs with a dependency on Job 1

JOB_ID_2=$(sbatch --parsable --dependency=afterok:JOB_ID_1 train_mcts_euclidean.job)
echo "Submitted Train Job: $JOB_ID_2 (Dependency: $JOB_ID_1)"

JOB_ID_3=$(sbatch --parsable --dependency=afterok:JOB_ID_1 train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_3 (Dependency: $JOB_ID_1)"

JOB_ID_4=$(sbatch --parsable --dependency=afterok:JOB_ID_1 train_ppo_euclidean.job)
echo "Submitted Train Job: $JOB_ID_4 (Dependency: $JOB_ID_1)"

JOB_ID_5=$(sbatch --parsable --dependency=afterok:JOB_ID_1 train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_5 (Dependency: $JOB_ID_1)"
