#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
JOB_DIR="$SCRIPT_DIR"

cd "$ROOT_DIR"

set -a
if [ -f "$ROOT_DIR/.env" ]; then
	source "$ROOT_DIR/.env"
fi
set +a

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "ERROR: python not found. Set PYTHON_BIN to the interpreter with lean_reinforcement on PYTHONPATH." >&2
	exit 1
fi

CONFIG_DUMPER=("$PYTHON_BIN" -m lean_reinforcement.utilities.dump_config)
MCTS_TYPE="alpha_zero"
LAST_JOB_ID=""

has_value_head_weights() {
	local run_dir="$1"
	local mcts_type="$2"
	[ -f "$run_dir/value_head_${mcts_type}_latest.pth" ]
}

has_ppo_weights() {
	local run_dir="$1"
	shopt -s nullglob
	local actor_files=("$run_dir"/ppo_actor_epoch_*.pth)
	local critic_files=("$run_dir"/ppo_critic_epoch_*.pth)
	shopt -u nullglob
	[ "${#actor_files[@]}" -gt 0 ] && [ "${#critic_files[@]}" -gt 0 ]
}

find_matching_run() {
	local base_dir="$1"
	local training_mode="$2"
	local mcts_type="$3"
	shift 3
	local train_args=("$@")

	local tmp_config
	tmp_config=$(mktemp)
	"${CONFIG_DUMPER[@]}" "${train_args[@]}" > "$tmp_config"

	local best_dir=""
	local best_mtime=0

	if [ -d "$base_dir" ]; then
		for dir in "$base_dir"/*; do
			[ -d "$dir" ] || continue
			local config_file="$dir/training_config.json"
			[ -f "$config_file" ] || continue
			if ! cmp -s "$config_file" "$tmp_config"; then
				continue
			fi

			if [ "$training_mode" = "ppo" ]; then
				has_ppo_weights "$dir" || continue
			else
				has_value_head_weights "$dir" "$mcts_type" || continue
			fi

			local mtime
			mtime=$(stat -c %Y "$dir" 2>/dev/null || echo 0)
			if [ "$mtime" -gt "$best_mtime" ]; then
				best_mtime="$mtime"
				best_dir="$dir"
			fi
		done
	fi

	rm -f "$tmp_config"
	echo "$best_dir"
}

submit_eval() {
	local run_dir="$1"
	local use_hyperbolic="$2"
	local curvature="${3:-}"

	local export_vars="ALL,RUN_DIR=$run_dir"
	if [ "$use_hyperbolic" = "1" ]; then
		export_vars+=",USE_HYPERBOLIC=1"
	else
		export_vars+=",USE_HYPERBOLIC=0"
	fi
	if [ -n "$curvature" ]; then
		export_vars+=",CURVATURE=$curvature"
	fi

	sbatch --parsable --export="$export_vars" "$JOB_DIR/evaluate.job"
}

submit_train() {
	local job_file="$1"
	local dependency="${2:-}"
	local curvature="${3:-}"

	local sbatch_args=()
	if [ -n "$dependency" ]; then
		sbatch_args+=(--dependency="afterok:$dependency")
	fi
	if [ -n "$curvature" ]; then
		sbatch_args+=(--export="curvature=$curvature")
	fi

	sbatch --parsable "${sbatch_args[@]}" "$job_file"
}

run_experiment() {
	local label="$1"
	local job_file="$2"
	local base_dir_rel="$3"
	local training_mode="$4"
	local use_hyperbolic="$5"
	local curvature="${6:-}"
	local dependency="${7:-}"

	local base_dir_abs="$ROOT_DIR/$base_dir_rel"
	local corpus_base="${CORPUS_DIR:-}"
	local indexed_corpus_path="${corpus_base%/}/indexed_corpus.pkl"

	local train_args=(
		--no-use-wandb
		--training-mode "$training_mode"
		--mcts-type "$MCTS_TYPE"
		--checkpoint-dir "$base_dir_rel"
		--indexed-corpus-path "$indexed_corpus_path"
	)

	if [ "$use_hyperbolic" = "1" ]; then
		train_args+=(--use-hyperbolic)
	else
		train_args+=(--no-use-hyperbolic)
	fi

	if [ -n "$curvature" ]; then
		train_args+=(--curvature "$curvature")
	fi

	local match
	match=$(find_matching_run "$base_dir_abs" "$training_mode" "$MCTS_TYPE" "${train_args[@]}")
	if [ -n "$match" ]; then
		echo "Found matching checkpoint for $label: $match"
		local eval_job_id
		eval_job_id=$(submit_eval "$match" "$use_hyperbolic" "$curvature")
		echo "Submitted Eval Job: $eval_job_id (uses existing checkpoint)"
		LAST_JOB_ID=""
		return
	fi

	echo "No matching checkpoint found for $label; submitting training."
	local train_job_id
	if [ "$use_hyperbolic" = "1" ] && [ -n "$curvature" ]; then
		train_job_id=$(submit_train "$job_file" "$dependency" "$curvature")
	else
		train_job_id=$(submit_train "$job_file" "$dependency")
	fi
	echo "Submitted Train Job: $train_job_id"
	LAST_JOB_ID="$train_job_id"
}

PRIMARY_JOB_ID=""

run_experiment \
	"mcts_euclidean" \
	"$JOB_DIR/train_mcts_euclidean.job" \
	"checkpoints/mcts_euclidean" \
	"value_head" \
	"0" \
	""

if [ -n "$LAST_JOB_ID" ]; then
	PRIMARY_JOB_ID="$LAST_JOB_ID"
fi

# run_experiment \
#     "mcts_hyperbolic_curvature_0.1" \
#     "$JOB_DIR/train_mcts_hyperbolic.job" \
#     "checkpoints/mcts_hyperbolic" \
#     "value_head" \
#     "1" \
#     "0.1" \
#     "$PRIMARY_JOB_ID"

# run_experiment \
#     "mcts_hyperbolic_curvature_0.5" \
#     "$JOB_DIR/train_mcts_hyperbolic.job" \
#     "checkpoints/mcts_hyperbolic" \
#     "value_head" \
#     "1" \
#     "0.5" \
#     "$PRIMARY_JOB_ID"

run_experiment \
	"mcts_hyperbolic_curvature_1.0" \
	"$JOB_DIR/train_mcts_hyperbolic.job" \
	"checkpoints/mcts_hyperbolic" \
	"value_head" \
	"1" \
	"1.0" \
	"$PRIMARY_JOB_ID"

run_experiment \
	"ppo_euclidean" \
	"$JOB_DIR/train_ppo_euclidean.job" \
	"checkpoints/ppo_euclidean" \
	"ppo" \
	"0" \
	"" \
	"$PRIMARY_JOB_ID"

# run_experiment \
#     "ppo_hyperbolic_curvature_0.1" \
#     "$JOB_DIR/train_ppo_hyperbolic.job" \
#     "checkpoints/ppo_hyperbolic" \
#     "ppo" \
#     "1" \
#     "0.1" \
#     "$PRIMARY_JOB_ID"

# run_experiment \
#     "ppo_hyperbolic_curvature_0.5" \
#     "$JOB_DIR/train_ppo_hyperbolic.job" \
#     "checkpoints/ppo_hyperbolic" \
#     "ppo" \
#     "1" \
#     "0.5" \
#     "$PRIMARY_JOB_ID"

run_experiment \
	"ppo_hyperbolic_curvature_1.0" \
	"$JOB_DIR/train_ppo_hyperbolic.job" \
	"checkpoints/ppo_hyperbolic" \
	"ppo" \
	"1" \
	"1.0" \
	"$PRIMARY_JOB_ID"
