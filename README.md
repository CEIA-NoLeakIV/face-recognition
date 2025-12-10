# Face Recognition

Framework PyTorch para treinamento e deploy de modelos de reconhecimento facial.

## Funcionalidades

- Múltiplas arquiteturas: SphereFace (20/36/64), MobileNet (V1/V2/V3)
- Funções de loss: CosFace, SphereFace, ArcFace, Linear
- Validação automática no LFW/CelebA
- Validação de faces com RetinaFace (opcional)
- Exportação ONNX para produção
- Visualização de métricas de treinamento

## Instalação

```bash
pip install -r requirements.txt
```

## Início Rápido

### Treinamento

```bash
python train.py \
    --root data/train/vggface2_112x112 \
    --database VggFace2 \
    --network mobilenetv3_large \
    --classifier MCP \
    --epochs 30 \
    --batch-size 64
```

### Inferência

```python
from inference import load_model, compare_faces

model = load_model("mobilenetv3_large", "weights/model.ckpt", device)
similarity, is_same = compare_faces(model, device, "img1.jpg", "img2.jpg")
```

### Avaliação

```bash
python evaluate.py
```

### Exportação ONNX

```bash
python scripts/onnx_export.py \
    --weights weights/model.ckpt \
    --network mobilenetv3_large \
    --dynamic
```

## Estrutura do Projeto

```
├── train.py              # Treinamento de modelos
├── evaluate.py           # Avaliação LFW/CelebA
├── inference.py          # Extração de features e comparação
├── finetune.py           # Fine-tuning de modelos pré-treinados
├── onnx_inference.py     # Inferência com ONNX runtime
├── models/               # Arquiteturas de rede
├── utils/                # Dataset, métricas, helpers
├── scripts/              # Utilitários de exportação ONNX
├── weights/              # Checkpoints dos modelos
└── metrics/              # Logs e gráficos de treinamento
    ├── logs/             # Logs JSON por época
    └── final_evaluation/ # Gráficos e métricas finais
```

## Arquiteturas

| Modelo | Parâmetros | Embedding |
|--------|------------|-----------|
| sphere20 | 24.5M | 512-d |
| sphere36 | 34.5M | 512-d |
| sphere64 | 54.5M | 512-d |
| mobilenetv1 | 3.2M | 512-d |
| mobilenetv2 | 2.2M | 512-d |
| mobilenetv3_small | 1.5M | 512-d |
| mobilenetv3_large | 4.2M | 512-d |

## Argumentos de Treinamento

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--root` | - | Caminho do dataset de treino |
| `--database` | WebFace | Dataset: WebFace, VggFace2, MS1M |
| `--network` | sphere20 | Arquitetura do modelo |
| `--classifier` | MCP | Loss: MCP, AL, ARC, L |
| `--batch-size` | 512 | Tamanho do batch |
| `--epochs` | 30 | Épocas de treinamento |
| `--lr` | 0.1 | Learning rate |
| `--val-dataset` | lfw | Dataset de validação |

### Validação com RetinaFace

Habilita detecção de faces durante a validação, excluindo pares sem faces detectadas:

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--use-retinaface-validation` | False | Habilita validação com RetinaFace |
| `--no-face-policy` | exclude | Política para imagens sem face: exclude, include |
| `--retinaface-conf-threshold` | 0.5 | Threshold de confiança do detector |

```bash
python train.py \
    --root data/train/vggface2_112x112 \
    --database VggFace2 \
    --network mobilenetv3_large \
    --use-retinaface-validation \
    --no-face-policy exclude
```

## Saídas

Após o treinamento, as métricas são salvas em `metrics/`:

```
metrics/
├── logs/                       # Logs JSON por época
│   ├── epoch_001.json
│   ├── epoch_002.json
│   └── ...
├── epoch_1/                    # Métricas por época
│   ├── lfw_roc_curve.png
│   └── lfw_confusion_matrix.png
└── final_evaluation/
    ├── accuracy_loss_curves.png
    ├── training_curves.png
    ├── lfw_roc_curve.png
    ├── lfw_confusion_matrix.png
    ├── training_history.json
    └── training_summary.txt
```

## Fine-tuning

Três estratégias disponíveis:

```bash
# Fine-tuning completo
python finetune.py --strategy FULL_FINETUNE ...

# Apenas head (backbone congelado)
python finetune.py --strategy HEAD_ONLY ...

# Descongelamento progressivo
python finetune.py --strategy PROGRESSIVE ...
```

## Licença

MIT License