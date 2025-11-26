#!/bin/bash
# Exemplo de script para executar fine-tuning

# Configurações básicas
PRETRAINED_CHECKPOINT="weights/mobilenetv3_large_MCP_best.ckpt"
DATASET_ROOT="/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/"
NETWORK="mobilenetv3_large"
CLASSIFIER="MCP"
EPOCHS=20
BATCH_SIZE=64

# Escolha a estratégia (FULL_FINETUNE, HEAD_ONLY, ou PROGRESSIVE)
STRATEGY="FULL_FINETUNE"

# Diretório para salvar modelos fine-tuned
SAVE_PATH="weights/finetuned_${STRATEGY}_$(date +%Y%m%d_%H%M%S)"

# Dataset de validação
VAL_DATASET="lfw"
VAL_ROOT="data/lfw/val"

echo "=========================================="
echo "Fine-tuning com estratégia: ${STRATEGY}"
echo "=========================================="
echo "Checkpoint pré-treinado: ${PRETRAINED_CHECKPOINT}"
echo "Dataset: ${DATASET_ROOT}"
echo "Salvar em: ${SAVE_PATH}"
echo "=========================================="

# Executar fine-tuning
python finetune.py \
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
    --root "${DATASET_ROOT}" \
    --network "${NETWORK}" \
    --strategy "${STRATEGY}" \
    --classifier "${CLASSIFIER}" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr-scheduler MultiStepLR \
    --milestones 10 15 \
    --gamma 0.1 \
    --save-path "${SAVE_PATH}" \
    --val-dataset "${VAL_DATASET}" \
    --val-root "${VAL_ROOT}" \
    --print-freq 50

echo ""
echo "Fine-tuning concluído!"
echo "Modelo salvo em: ${SAVE_PATH}"
echo "Melhor modelo: ${SAVE_PATH}/*_best.ckpt"
