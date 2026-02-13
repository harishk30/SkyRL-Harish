EMBEDDING_MODEL=${EMBEDDING_MODEL:-"qwen3_06b"}

if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_DIR=/home/hk4638/scratch/data/citation_prediction
    INDEX_FILE=${INDEX_DIR}/qwen3_Flat.index
    CORPUS_PATH=/home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    MODEL_PATH="/scratch/gpfs/ZHUANGL/hk4638/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_DIR=/home/hk4638/scratch/data/citation_prediction/qwen3_06b_embed
    INDEX_FILE=${INDEX_DIR}/qwen3_Flat.index
    CORPUS_PATH=/home/hk4638/scratch/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl
    RETRIEVER_NAME=qwen3
fi

python examples/citation_prediction/retriever/retrieval_server.py \
  --index_path $INDEX_FILE \
  --corpus_path $CORPUS_PATH \
  --topk 3 \
  --retriever_name $RETRIEVER_NAME \
  --retriever_model $MODEL_PATH \
  --faiss_gpu
