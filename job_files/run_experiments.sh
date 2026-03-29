#!/bin/bash

JOB_ID_1=$(sbatch --parsable train_mcts_euclidean.job)
echo "Submitted Train Job: $JOB_ID_1"

JOB_ID_2=$(sbatch --parsable train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_2"

JOB_ID_3=$(sbatch --parsable train_ppo_euclidean.job)
echo "Submitted Train Job: $JOB_ID_3"

JOB_ID_4=$(sbatch --parsable train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_4"
