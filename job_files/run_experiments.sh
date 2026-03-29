#!/bin/bash

JOB_ID_1=$(sbatch --parsable train_mcts_euclidean.job)
echo "Submitted Train Job: $JOB_ID_1"

JOB_ID_2=$(sbatch --parsable train_mcts_hyperbolic.job --curvature 0.1)
echo "Submitted Train Job: $JOB_ID_2"

JOB_ID_3=$(sbatch --parsable train_mcts_hyperbolic.job --curvature 0.5)
echo "Submitted Train Job: $JOB_ID_3"

JOB_ID_4=$(sbatch --parsable train_mcts_hyperbolic.job --curvature 1.0)
echo "Submitted Train Job: $JOB_ID_4"

JOB_ID_5=$(sbatch --parsable train_ppo_euclidean.job)
echo "Submitted Train Job: $JOB_ID_5"

JOB_ID_6=$(sbatch --parsable train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_6"

JOB_ID_7=$(sbatch --parsable train_ppo_hyperbolic.job --curvature 0.1)
echo "Submitted Train Job: $JOB_ID_7"

JOB_ID_8=$(sbatch --parsable train_ppo_hyperbolic.job --curvature 0.5)
echo "Submitted Train Job: $JOB_ID_8"

JOB_ID_9=$(sbatch --parsable train_ppo_hyperbolic.job --curvature 1.0)
echo "Submitted Train Job: $JOB_ID_9"