# Face Recognition Training Framework

## Índice

- [Características](#características)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Arquiteturas Suportadas](#arquiteturas-suportadas)
- [Datasets](#datasets)
- [Argumentos de Treinamento](#argumentos-de-treinamento)
- [Validação com RetinaFace](#validação-com-retinaface)
- [Estrutura de Outputs](#estrutura-de-outputs)
- [ONNX](#onnx)
- [Avaliação](#avaliação)

## Características

- **Múltiplas Arquiteturas**: SphereFace (20/36/64), MobileNet (v1/v2/v3)
- **Loss Functions**: CosFace (MCP), SphereFace (AL), ArcFace, Linear
- **Validação Automática**: Métricas completas a cada época (LFW/CelebA)
- **Face Validation**: Detecção de faces com RetinaFace durante validação (opcional)
- **Visualizações**: ROC Curve, Confusion Matrix, Training Curves
- **Early Stopping**: Patience configurável
- **Multi-GPU**: Suporte a treinamento distribuído
- **Exportação ONNX**: Deploy em produção com formato otimizado

## Instalação
```bash
# Dependências básicas
pip install -r requirements.txt
```

## Uso Rápido

### Treinamento Básico
```bash
python train.py \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network mobilenetv3_large \
    --classifier MCP \
    --val-dataset lfw \
    --val-root data/lfw/val \
    --epochs 30 \
    --batch-size 64
```

### Treinamento com Validação de Faces
```bash
python train.py \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network mobilenetv3_large \
    --classifier MCP \
    --val-dataset lfw \
    --val-root data/lfw/val \
    --use-retinaface-validation \
    --no-face-policy exclude \
    --epochs 30
```

## Arquiteturas Suportadas

### Backbones

- **SphereFace**: `sphere20`, `sphere36`, `sphere64`
- **MobileNet**: `mobilenetv1`, `mobilenetv2`, `mobilenetv3_small`, `mobilenetv3_large`

Todos os modelos geram embeddings de 512 dimensões.

### Loss Functions

- **MCP** (Margin Cosine Product): Implementação CosFace
- **AL** (Angle Linear): Implementação SphereFace  
- **ARC**: Implementação ArcFace
- **L** (Linear): Classificador linear padrão

## Datasets

### Treinamento

- **WebFace**: 10,572 identidades
- **VggFace2**: 8,631 identidades
- **MS1M**: 85,742 identidades
- **VggFaceHQ**: 9,131 identidades

### Validação

- **LFW** (Labeled Faces in the Wild): Benchmark padrão
- **CelebA**: Dataset de celebridades

**Estrutura esperada:**
```
data/
├── train/
│   └── <dataset_name>/
│       ├── identity_1/
│       │   ├── img1.jpg
│       │   └── img2.jpg
│       └── identity_2/
│           └── ...
└── lfw/val/
    ├── lfw_ann.txt
    └── <person_name>/
        ├── <person_name>_0001.jpg
        └── ...
```

## Argumentos de Treinamento

### Dataset e Paths

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--root` | str | `data/train/webface_112x112/` | Diretório de imagens de treino |
| `--database` | str | `WebFace` | Dataset: WebFace, VggFace2, MS1M, VggFaceHQ |

### Modelo

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--network` | str | `sphere20` | Arquitetura: sphere20/36/64, mobilenetv1/v2/v3_small/v3_large |
| `--classifier` | str | `MCP` | Loss function: MCP, AL, ARC, L |

### Treinamento

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--batch-size` | int | 512 | Tamanho do batch |
| `--epochs` | int | 30 | Número de épocas |
| `--lr` | float | 0.1 | Learning rate inicial |
| `--momentum` | float | 0.9 | Momentum do SGD |
| `--weight-decay` | float | 5e-4 | Weight decay |
| `--num-workers` | int | 8 | Workers do DataLoader |

### Learning Rate Scheduler

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--lr-scheduler` | str | `MultiStepLR` | Tipo: MultiStepLR, StepLR |
| `--milestones` | int[] | `[10, 20, 25]` | Épocas para reduzir LR (MultiStepLR) |
| `--step-size` | int | 10 | Período de decay (StepLR) |
| `--gamma` | float | 0.1 | Fator multiplicativo de decay |

### Validação

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--val-dataset` | str | `lfw` | Dataset de validação: lfw, celeba |
| `--val-root` | str | `data/lfw/val` | Diretório do dataset de validação |
| `--val-threshold` | float | 0.35 | Threshold de similaridade |

### Checkpoints

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--save-path` | str | `weights` | Diretório para salvar checkpoints |
| `--checkpoint` | str | None | Checkpoint para continuar treino |

### Multi-GPU

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--world-size` | int | 1 | Número de processos distribuídos |
| `--local_rank` | int | 0 | Rank local para treinamento distribuído |

## Validação com RetinaFace

Sistema opcional de validação de faces usando RetinaFace da UniFace durante a avaliação no LFW/CelebA.

### Argumentos

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--use-retinaface-validation` | flag | False | Habilita validação com RetinaFace |
| `--no-face-policy` | str | `exclude` | Política para imagens sem face: exclude, include |
| `--retinaface-conf-threshold` | float | 0.5 | Threshold de confiança do detector |
| `--face-validation-cache-dir` | str | `face_validation_cache` | Diretório de cache |

### Funcionamento

1. **Primeira época**: Valida todas as imagens do dataset de validação uma única vez
2. **Cache**: Salva resultados em `face_validation_cache/<dataset>_validation.json`
3. **Épocas seguintes**: Carrega cache instantaneamente (sem reprocessamento)
4. **Filtragem**: Aplica política configurada (exclude/include)
5. **Relatório final**: Gera `face_validation_report.json` com estatísticas

### Políticas

- **exclude**: Remove pares onde alguma imagem não tem face detectada
- **include**: Mantém todos os pares, relatório disponível para auditoria

### Outputs Gerados
```
face_validation_cache/
└── lfw_validation.json           # Cache de detecções

weights/metrics/final_evaluation/
└── face_validation_report.json   # Relatório detalhado com:
                                   # - Estatísticas de detecção
                                   # - Lista de imagens sem faces
                                   # - Imagens com múltiplas faces
                                   # - Taxa de exclusão
```

## Estrutura de Outputs

### Durante o Treinamento
```
weights/
├── <model>_<classifier>_best.ckpt    # Melhor modelo
├── <model>_<classifier>_last.ckpt    # Último checkpoint
├── <model>_<classifier>_best.onnx    # Melhor modelo (ONNX)
├── <model>_<classifier>_last.onnx    # Último checkpoint (ONNX)
│
├── metrics/
│   ├── epoch_001/
│   │   ├── lfw_roc_curve.png
│   │   └── lfw_confusion_matrix.png
│   ├── epoch_002/
│   │   └── ...
│   └── final_evaluation/
│       ├── lfw_roc_curve.png
│       ├── lfw_confusion_matrix.png
│       └── face_validation_report.json  # Se RetinaFace habilitado
│
└── final_report/
    ├── training_curves.png               # Loss, Accuracy, F1, AUC
    ├── confusion_matrix_evolution.png    # Evolução (início/meio/fim)
    ├── learning_rate_schedule.png        # Schedule do LR
    ├── all_metrics_overview.png          # Todas métricas lado-a-lado
    ├── face_validation_stats.png         # Se RetinaFace habilitado
    ├── training_history.json             # Histórico completo
    └── training_summary.txt              # Resumo estatístico
```

### Métricas Rastreadas

**Treinamento:**
- Training Loss
- Training Accuracy
- Validation Accuracy (split interno)

**Validação Externa (LFW/CelebA):**
- Mean Similarity
- Accuracy, Precision, Recall, F1-Score
- AUC, EER (Equal Error Rate)
- FAR, FRR
- Confusion Matrix
- ROC Curve

**Face Validation (se habilitado):**
- Total de pares
- Pares válidos
- Pares excluídos
- Taxa de exclusão

## ONNX

### Exportação

Exporte checkpoints PyTorch para formato ONNX otimizado para deployment em produção.

#### Uso Básico
```bash
python scripts/onnx_export.py \
    --weights weights/mobilenetv3_large_MCP_best.ckpt \
    --network mobilenetv3_large
```

#### Com Batch Dinâmico (Recomendado)
```bash
python scripts/onnx_export.py \
    --weights weights/mobilenetv3_large_MCP_best.ckpt \
    --network mobilenetv3_large \
    --dynamic
```

### Argumentos

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `-w, --weights` | str | `./weights/mobilenetv2_mcp.pth` | Caminho do checkpoint (.ckpt ou .pth) |
| `-n, --network` | str | `mobilenetv2` | Arquitetura: sphere20/36/64, mobilenetv1/v2/v3_small/v3_large |
| `--dynamic` | flag | False | Habilita batch size dinâmico |

### Output Gerado

O arquivo ONNX é salvo no mesmo diretório do checkpoint:
```
weights/
├── mobilenetv3_large_MCP_best.ckpt
└── mobilenetv3_large_MCP_best.onnx    # ← Arquivo exportado
```

**Especificações:**
- Input shape: `(batch, 3, 112, 112)`
- Output shape: `(batch, 512)`
- Opset version: 16
- Batch dinâmico: Suporta qualquer tamanho de batch (se `--dynamic` habilitado)

### Inferência

Execute inferência usando modelos ONNX exportados:
```bash
python onnx_inference.py
```

Edite o script para configurar:
- Caminho do modelo ONNX
- Paths das imagens a comparar
- Threshold de similaridade

O script realiza:
1. Carregamento do modelo ONNX
2. Detecção de faces com RetinaFace
3. Extração de embeddings
4. Comparação de similaridade entre faces

## Avaliação

### Standalone
```bash
python evaluate.py
```

### Via Notebook

Use `1.Notebooks/Eval.ipynb` para análises detalhadas e visualizações customizadas.

### Retomar Treinamento
```bash
python train.py \
    --checkpoint weights/mobilenetv3_large_MCP_last.ckpt \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network mobilenetv3_large \
    --classifier MCP
```

O histórico de métricas é preservado automaticamente.

### Multi-GPU
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py \
    --world-size 2 \
    --root data/train/vggface2_aligned \
    --database VggFace2 \
    --network mobilenetv3_large \
    --classifier MCP
```

### Preprocessamento

As imagens devem ser:
- **Tamanho**: 112x112 pixels
- **Formato**: RGB
- **Normalização**: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

O framework aplica resize e normalização automática durante treinamento.

## Licença

Este projeto é fornecido para fins de pesquisa.