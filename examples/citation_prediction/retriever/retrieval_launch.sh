EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_06b"}
CORPUS_PATH="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl"

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction/qwen3_4b_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="/home/hk4638/scratch/shared/data/qwen3_06_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
fi

python examples/citation_prediction/retriever/retrieval_server.py \
  --index_path $INDEX_FILE \
  --corpus_path $CORPUS_PATH \
  --topk 3 \
  --retriever_name $RETRIEVER_NAME \
  --retriever_model $MODEL_PATH \
  --faiss_gpu
