set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on Citation Prediction data.
# follow the instructions in examples/citation_prediction/README.md for setting up the dataset
# and for starting the local search server
# export WANDB_API_KEY=<your_key_here>
# bash examples/citation_prediction/run_citation_prediction.sh

# path for dataset (.parquet files) containing the prompts and metadata for each question
DATA_DIR="$HOME/data/citation_prediction"
PROMPT_STYLE=${PROMPT_STYLE:-"short"}   # "short" or "extended"

RUN_NAME="skyrl-citation-prediction_qwen3-4b_${PROMPT_STYLE}_4turns_maxgeneratelen_500-multiturn-sync-TIS_2.0"

TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/${PROMPT_STYLE}/train.parquet']" \
  data.val_data="['${DATA_DIR}/${PROMPT_STYLE}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path="Qwen/Qwen3-4B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.use_liger_kernel=true \
  trainer.policy.sequence_parallel_size=2 \
  trainer.ref.use_liger_kernel=true \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.5 \
  trainer.epochs=3 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=500 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>"]' \
  environment.env_class="citation_prediction" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.citation_prediction.log_requests=false \
  environment.skyrl_gym.citation_prediction.search_url="http://127.0.0.1:8000/retrieve" \
  environment.skyrl_gym.citation_prediction.topk=3 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-citation-prediction" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/${RUN_NAME}" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>"]' \
  trainer.export_path="$HOME/${RUN_NAME}/exports" \
  trainer.eval_interval=50 \
  $@
