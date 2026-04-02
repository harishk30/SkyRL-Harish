# AI Lab Partition (Della) — Constraints & Best Practices

## Hardware
- H200 PCIe GPUs, 141 GB each
- 8 GPUs per node, 64 CPU cores, 1.5 TB system RAM
- 18 nodes total

## Constraints
- < 10 GPUs per job
- < 150 GB memory per GPU (141 GB hardware limit, so no issue)
- 8 CPU cores per GPU max
- Partition: `--partition=ailab` (must be in `ailab` Unix group, submit from `della-gpu`)
- No `--qos` needed (it's just for accounting)
- No `--constraint` needed (only H200s on this partition)
- Use `--gres=gpu:N` not `--gres=gpu:h200:N`

## Best Practices
- Target 2 GPUs per job for 4B model training (sufficient with 141 GB/GPU)
- Keep jobs < 18 hours to share resources fairly
- Use SLURM job arrays (`--array=0-N%1`) for sweep configs — they share scheduling priority
- Run retriever on the same node, sharing GPU 0 with training (set CUDA_VISIBLE_DEVICES=0 for retriever)
- Set `gpu_memory_utilization` to ~0.8 (retriever shares GPU 0 but only needs ~55 GB for embedding model + FAISS)
- Save checkpoints regularly and use `trainer.resume_mode=latest` for continuation

## Access
- Must SSH to `della-gpu.princeton.edu` (not `della9`)
- Same filesystem as other Della login nodes
- Added via AI Lab RSE team (ailab-rse@princeton.edu)
