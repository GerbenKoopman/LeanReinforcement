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

# Single-parameter sweep defaults (edit as needed).
BASE_VALUE_HEAD_DIM=1024
BASE_VALUE_HEAD_HIDDEN_LAYERS=1
BASE_EXPLORATION_WEIGHT=1.41421356237
BASE_HYPERBOLIC_CURVATURE=1.0

SWEEP_VALUE_HEAD_DIMS_ENABLED=1
SWEEP_VALUE_HEAD_HIDDEN_LAYERS_ENABLED=1
SWEEP_EXPLORATION_WEIGHTS_ENABLED=1
SWEEP_CURVATURES_ENABLED=1

SWEEP_VALUE_HEAD_DIMS=(64 128 256)
SWEEP_VALUE_HEAD_HIDDEN_LAYERS=(1 2 4)
SWEEP_EXPLORATION_WEIGHTS=(1.0 0.5 0.25)
SWEEP_CURVATURES=(1.0 0.5 0.1)

has_value_head_weights() {
	local run_dir="$1"
	local mcts_type="$2"
	[ -f "$run_dir/value_head_${mcts_type}_latest.pth" ]
}

find_matching_run() {
	local base_dir="$1"
	local mcts_type="$2"
	shift 2
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

			has_value_head_weights "$dir" "$mcts_type" || continue

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
	local extra_args_str="${4:-}"

	local export_vars=("ALL")
	if [ -n "$curvature" ]; then
		export_vars+=("curvature=$curvature")
	fi
	if [ -n "$extra_args_str" ]; then
		export_vars+=("EXTRA_TRAIN_ARGS=$extra_args_str")
	fi

	local export_joined
	export_joined=$(IFS=,; echo "${export_vars[*]}")

	local sbatch_args=()
	if [ -n "$dependency" ]; then
		sbatch_args+=(--dependency="afterok:$dependency")
	fi
	sbatch_args+=(--export="$export_joined")

	sbatch --parsable "${sbatch_args[@]}" "$job_file"
}

run_experiment() {
	local label="$1"
	local job_file="$2"
	local base_dir_rel="$3"
	local use_hyperbolic="$4"
	local curvature="${5:-}"
	local dependency="${6:-}"
	local extra_args_str="${7:-}"

	local base_dir_abs="$ROOT_DIR/$base_dir_rel"
	local corpus_base="${CORPUS_DIR:-}"
	local indexed_corpus_path="${corpus_base%/}/indexed_corpus.pkl"

	local train_args=(
		--no-use-wandb
		--training-mode value_head
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

	local extra_args=()
	if [ -n "$extra_args_str" ]; then
		# shellcheck disable=SC2206
		extra_args=($extra_args_str)
		train_args+=("${extra_args[@]}")
	fi

	local match
	match=$(find_matching_run "$base_dir_abs" "$MCTS_TYPE" "${train_args[@]}")
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
		train_job_id=$(submit_train "$job_file" "$dependency" "$curvature" "$extra_args_str")
	else
		train_job_id=$(submit_train "$job_file" "$dependency" "" "$extra_args_str")
	fi
	echo "Submitted Train Job: $train_job_id"
	LAST_JOB_ID="$train_job_id"
}

queue_single_sweep() {
	local label_prefix="$1"
	local job_file="$2"
	local base_dir_rel="$3"
	local use_hyperbolic="$4"
	local sweep_mode="$5"
	shift 5
	local sweep_values=("$@")

	if [ "${#sweep_values[@]}" -eq 0 ]; then
		return
	fi

	for sweep_value in "${sweep_values[@]}"; do
		local dim="$BASE_VALUE_HEAD_DIM"
		local layers="$BASE_VALUE_HEAD_HIDDEN_LAYERS"
		local puct="$BASE_EXPLORATION_WEIGHT"
		local curvature="$BASE_HYPERBOLIC_CURVATURE"

		case "$sweep_mode" in
			value_head_hidden_layers)
				layers="$sweep_value"
				;;
			value_head_latent_dim)
				dim="$sweep_value"
				;;
			exploration_weight)
				puct="$sweep_value"
				;;
			curvature)
				curvature="$sweep_value"
				;;
		esac

		local label_suffix="baseline"
		case "$sweep_mode" in
			value_head_hidden_layers)
				label_suffix="layers${layers}"
				;;
			value_head_latent_dim)
				label_suffix="dim${dim}"
				;;
			exploration_weight)
				label_suffix="puct${puct}"
				;;
			curvature)
				label_suffix="curv${curvature//./p}"
				;;
		esac

		local label="${label_prefix}_${label_suffix}"
		local extra_args="--value-head-latent-dim $dim --value-head-hidden-layers $layers --exploration-weight $puct"
		local dependency="$PRIMARY_JOB_ID"

		run_experiment \
			"$label" \
			"$job_file" \
			"$base_dir_rel" \
			"$use_hyperbolic" \
			"$curvature" \
			"$dependency" \
			"$extra_args"

		if [ -z "$PRIMARY_JOB_ID" ] && [ -n "$LAST_JOB_ID" ]; then
			PRIMARY_JOB_ID="$LAST_JOB_ID"
		fi
	done
}

queue_value_head_ablation() {
	local label_prefix="$1"
	local job_file="$2"
	local base_dir_rel="$3"
	local use_hyperbolic="$4"

	if [ "$SWEEP_VALUE_HEAD_DIMS_ENABLED" = "1" ]; then
		queue_single_sweep \
			"$label_prefix" \
			"$job_file" \
			"$base_dir_rel" \
			"$use_hyperbolic" \
			"value_head_latent_dim" \
			"${SWEEP_VALUE_HEAD_DIMS[@]}"
	fi

	if [ "$SWEEP_VALUE_HEAD_HIDDEN_LAYERS_ENABLED" = "1" ]; then
		queue_single_sweep \
			"$label_prefix" \
			"$job_file" \
			"$base_dir_rel" \
			"$use_hyperbolic" \
			"value_head_hidden_layers" \
			"${SWEEP_VALUE_HEAD_HIDDEN_LAYERS[@]}"
	fi

	if [ "$SWEEP_EXPLORATION_WEIGHTS_ENABLED" = "1" ]; then
		queue_single_sweep \
			"$label_prefix" \
			"$job_file" \
			"$base_dir_rel" \
			"$use_hyperbolic" \
			"exploration_weight" \
			"${SWEEP_EXPLORATION_WEIGHTS[@]}"
	fi

	if [ "$SWEEP_CURVATURES_ENABLED" = "1" ]; then
		queue_single_sweep \
			"$label_prefix" \
			"$job_file" \
			"$base_dir_rel" \
			"$use_hyperbolic" \
			"curvature" \
			"${SWEEP_CURVATURES[@]}"
	fi
}

PRIMARY_JOB_ID=""

queue_value_head_ablation \
	"mcts_hyperbolic" \
	"$JOB_DIR/train_mcts_hyperbolic.job" \
	"checkpoints/mcts_hyperbolic" \
	"1"
