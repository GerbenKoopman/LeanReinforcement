#!/bin/bash

JOB_ID_1=$(sbatch --parsable train_mcts_euclidean.job)
echo "Submitted Train Job: $JOB_ID_1"

JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=0.1 train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_2, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_3=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=0.5 train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_3, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_4=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=1.0 train_mcts_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_4, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_5=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 train_ppo_euclidean.job)
echo "Submitted Train Job: $JOB_ID_5, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_6=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=0.1 train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_6, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_7=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=0.5 train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_7, Starts if $JOB_ID_1 finishes succesfully"

JOB_ID_8=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --export=curvature=1.0 train_ppo_hyperbolic.job)
echo "Submitted Train Job: $JOB_ID_8, Starts if $JOB_ID_1 finishes succesfully"
