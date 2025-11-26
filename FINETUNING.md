# Guia de Fine-Tuning para Face Recognition

Este guia apresenta um passo a passo completo para realizar fine-tuning de modelos pré-treinados de face recognition em novos datasets usando o script `finetune.py`.

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estratégias de Fine-Tuning](#estratégias-de-fine-tuning)
3. [Pré-requisitos](#pré-requisitos)
4. [Preparação do Dataset](#preparação-do-dataset)
5. [Executando o Fine-Tuning](#executando-o-fine-tuning)
6. [Monitoramento e Métricas](#monitoramento-e-métricas)
7. [Salvando e Carregando Modelos](#salvando-e-carregando-modelos)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

O fine-tuning permite adaptar modelos pré-treinados em VGGFace2 (validados no LFW) para novos datasets específicos. O script oferece 3 estratégias diferentes de fine-tuning, cada uma com suas vantagens e casos de uso.

### Estrutura do Processo

```
Modelo Pré-treinado (VGGFace2)
    ↓
Fine-tuning no Novo Dataset
    ↓
Modelo Fine-tuned Salvo
```

---

## 🎨 Estratégias de Fine-Tuning

### 1. FULL_FINETUNE (Fine-tuning Completo)

**Descrição:** Todos os parâmetros do modelo (backbone + classification head) são atualizados durante o treinamento, porém com learning rates reduzidos para preservar os conhecimentos pré-treinados.

**Vantagens:**
- Máxima flexibilidade para adaptação ao novo dataset
- Melhor para datasets grandes e diversos
- Permite ajustes finos em todas as camadas

**Desvantagens:**
- Requer mais recursos computacionais
- Risco de overfitting em datasets pequenos
- Necessita mais épocas de treinamento

**Learning Rates Padrão:**
- Backbone: `0.01` (10x menor que treinamento do zero)
- Head: `0.1`

**Recomendado para:**
- Datasets com mais de 10.000 imagens
- Quando o novo dataset é similar ao dataset de pré-treinamento
- Quando há recursos computacionais suficientes

---

### 2. HEAD_ONLY (Apenas Classification Head)

**Descrição:** O backbone do modelo é completamente congelado (parâmetros fixos), e apenas o classification head é treinado para as novas classes.

**Vantagens:**
- Muito rápido de treinar
- Baixo risco de overfitting
- Ideal para datasets pequenos
- Preserva completamente os features aprendidos no pré-treinamento

**Desvantagens:**
- Limita a capacidade de adaptação ao novo dataset
- Não aproveita ajustes específicos do backbone

**Learning Rates Padrão:**
- Backbone: `0.0` (congelado)
- Head: `0.1`

**Recomendado para:**
- Datasets com menos de 5.000 imagens
- Quando o dataset é muito similar ao de pré-treinamento
- Quando há limitações de tempo/computação
- Transfer learning rápido

---

### 3. PROGRESSIVE (Descongelamento Progressivo)

**Descrição:** Treinamento em 3 fases progressivas:
1. **Fase 1 (0-33% épocas):** Apenas classification head treinado
2. **Fase 2 (33-66% épocas):** Descongela layers finais do backbone
3. **Fase 3 (66-100% épocas):** Descongela todo o backbone

**Vantagens:**
- Equilíbrio entre adaptação e preservação de conhecimento
- Aprendizado gradual e estável
- Bom para datasets médios
- Reduz risco de overfitting inicial

**Desvantagens:**
- Mais complexo de configurar
- Pode ser mais lento que HEAD_ONLY

**Learning Rates Padrão:**
- Backbone: `0.005` (quando descongelado)
- Head: `0.1`

**Recomendado para:**
- Datasets com 5.000 - 10.000 imagens
- Quando há tempo suficiente para treinamento
- Quando precisa de melhor adaptação que HEAD_ONLY mas com menos risco que FULL_FINETUNE

---

## 📦 Pré-requisitos

### 1. Ambiente Python

Certifique-se de ter as dependências instaladas:

```bash
pip install -r requirements.txt
```

### 2. Modelo Pré-treinado

Você precisa de um checkpoint do modelo pré-treinado. O modelo deve ter sido treinado em VGGFace2 e validado no LFW.

**Estrutura esperada do checkpoint:**
```python
{
    'model': state_dict,           # Pesos do backbone
    'epoch': int,                  # Época final
    'optimizer': state_dict,       # Estado do otimizador (opcional)
    'lr_scheduler': state_dict,    # Estado do scheduler (opcional)
    'args': namespace              # Argumentos do treinamento (opcional)
}
```

### 3. Dataset Preparado

O dataset deve estar organizado da seguinte forma:

```
/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
├── person_001/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── person_002/
│   ├── image001.jpg
│   └── ...
└── ...
```

**Requisitos:**
- Cada pasta representa uma identidade/classe diferente
- Imagens devem estar alinhadas e redimensionadas (recomendado 112x112)
- Formatos suportados: `.jpg`, `.jpeg`, `.png`

---

## 📁 Preparação do Dataset

### 1. Verificar Estrutura

```bash
# Verificar número de classes
ls -d /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/* | wc -l

# Verificar número total de imagens
find /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l
```

### 2. Estatísticas do Dataset

O script contará automaticamente:
- Número de classes (identidades)
- Número de imagens por classe
- Total de imagens

---

## 🚀 Executando o Fine-Tuning

### Comando Básico

```bash
python finetune.py \
    --pretrained-checkpoint weights/mobilenetv3_large_MCP_best.ckpt \
    --root /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --network mobilenetv3_large \
    --strategy FULL_FINETUNE \
    --classifier MCP \
    --epochs 20 \
    --batch-size 64
```

### Exemplos por Estratégia

#### Exemplo 1: FULL_FINETUNE (Fine-tuning Completo)

```bash
python finetune.py \
    --pretrained-checkpoint weights/mobilenetv3_large_MCP_best.ckpt \
    --root /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --network mobilenetv3_large \
    --strategy FULL_FINETUNE \
    --classifier MCP \
    --epochs 25 \
    --batch-size 64 \
    --lr-backbone 0.01 \
    --lr-head 0.1 \
    --lr-scheduler MultiStepLR \
    --milestones 15 20 \
    --gamma 0.1 \
    --save-path weights/finetuned_full \
    --val-dataset lfw \
    --val-root data/lfw/val
```

#### Exemplo 2: HEAD_ONLY (Apenas Head)

```bash
python finetune.py \
    --pretrained-checkpoint weights/mobilenetv3_large_MCP_best.ckpt \
    --root /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --network mobilenetv3_large \
    --strategy HEAD_ONLY \
    --classifier MCP \
    --epochs 10 \
    --batch-size 128 \
    --lr-head 0.1 \
    --save-path weights/finetuned_head_only \
    --val-dataset lfw \
    --val-root data/lfw/val
```

#### Exemplo 3: PROGRESSIVE (Progressivo)

```bash
python finetune.py \
    --pretrained-checkpoint weights/mobilenetv3_large_MCP_best.ckpt \
    --root /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --network mobilenetv3_large \
    --strategy PROGRESSIVE \
    --classifier MCP \
    --epochs 30 \
    --batch-size 64 \
    --lr-backbone 0.005 \
    --lr-head 0.1 \
    --lr-scheduler MultiStepLR \
    --milestones 10 20 25 \
    --gamma 0.1 \
    --save-path weights/finetuned_progressive \
    --val-dataset lfw \
    --val-root data/lfw/val
```

### Parâmetros Principais

| Parâmetro | Descrição | Padrão | Obrigatório |
|-----------|-----------|--------|-------------|
| `--pretrained-checkpoint` | Caminho do checkpoint pré-treinado | - | ✅ Sim |
| `--root` | Diretório do novo dataset | - | ✅ Sim |
| `--network` | Arquitetura do modelo | - | ✅ Sim |
| `--strategy` | Estratégia de fine-tuning | `FULL_FINETUNE` | Não |
| `--classifier` | Tipo de classificador | `MCP` | Não |
| `--epochs` | Número de épocas | `20` | Não |
| `--batch-size` | Tamanho do batch | `64` | Não |
| `--lr-backbone` | Learning rate do backbone | Auto | Não |
| `--lr-head` | Learning rate do head | `0.1` | Não |
| `--save-path` | Diretório para salvar modelos | `weights/finetuned` | Não |
| `--val-dataset` | Dataset de validação | `lfw` | Não |
| `--val-root` | Diretório do dataset de validação | `data/lfw/val` | Não |

---

## 📊 Monitoramento e Métricas

### Durante o Treinamento

O script exibe informações a cada época:

```
Epoch: [1/20][00100/00500] Loss: 2.345, Accuracy: 45.23%, LR: 0.01000 Time: 0.123s
```

### Métricas de Validação

A cada época, o modelo é avaliado no conjunto de validação (LFW ou CelebA):

```
==================================================
EPOCH 1 VALIDATION METRICS
==================================================

Validation Metrics (Threshold=0.35):
  Precision: 0.8234
  Recall:    0.7891
  F1-Score:  0.8059
  Accuracy:  0.8123

ROC Metrics:
  AUC: 0.9123
  EER: 0.0821

Internal validation accuracy: 0.8456
==================================================
```

### Arquivos de Saída

Os resultados são salvos em:

```
weights/finetuned/
├── mobilenetv3_large_MCP_finetuned_full_last.ckpt      # Último checkpoint
├── mobilenetv3_large_MCP_finetuned_full_best.ckpt      # Melhor checkpoint
└── metrics/
    ├── epoch_1/
    │   ├── lfw_roc_curve.png
    │   └── lfw_confusion_matrix.png
    ├── epoch_2/
    │   └── ...
    └── final_evaluation/
        ├── lfw_roc_curve.png
        └── lfw_confusion_matrix.png
```

### Early Stopping

O script inclui early stopping automático que para o treinamento se não houver melhoria por 10 épocas consecutivas.

---

## 💾 Salvando e Carregando Modelos

### Estrutura do Checkpoint Salvo

```python
{
    'epoch': int,                          # Época atual
    'model': state_dict,                   # Pesos do backbone
    'classification_head': state_dict,     # Pesos do classification head
    'optimizer': state_dict,               # Estado do otimizador
    'lr_scheduler': state_dict,            # Estado do scheduler
    'num_classes': int,                    # Número de classes do novo dataset
    'strategy': str,                       # Estratégia usada
    'pretrained_checkpoint': str,          # Caminho do checkpoint original
    'args': namespace                      # Argumentos do fine-tuning
}
```

### Carregando um Modelo Fine-tuned

```python
import torch
from inference import get_network, load_model

# Caminho do checkpoint fine-tuned
checkpoint_path = "weights/finetuned/mobilenetv3_large_MCP_finetuned_full_best.ckpt"

# Carregar checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Obter número de classes
num_classes = checkpoint['num_classes']
strategy = checkpoint['strategy']

print(f"Modelo fine-tuned com estratégia: {strategy}")
print(f"Número de classes: {num_classes}")

# Carregar modelo usando a função existente (pode precisar ajuste)
model = get_network('mobilenetv3_large')(embedding_dim=512)
model.load_state_dict(checkpoint['model'])

# Carregar classification head
from utils.metrics import MarginCosineProduct
classification_head = MarginCosineProduct(512, num_classes)
classification_head.load_state_dict(checkpoint['classification_head'])
```

### Usando o Modelo para Inferência

```python
import torch
from inference import extract_features

# Carregar modelo e head
model.eval()
classification_head.eval()

# Extrair features
with torch.no_grad():
    embeddings = model(images)  # (batch_size, 512)
    outputs, _ = classification_head(embeddings, labels)  # Para treinamento
    # ou apenas embeddings para comparação de similaridade
```

---

## 🔧 Configurações Avançadas

### Learning Rates Customizados

Você pode especificar learning rates personalizados:

```bash
python finetune.py \
    --pretrained-checkpoint weights/model.ckpt \
    --root /path/to/dataset/ \
    --network mobilenetv3_large \
    --strategy FULL_FINETUNE \
    --lr-backbone 0.005 \      # LR menor para backbone
    --lr-head 0.05             # LR menor para head
```

### Schedulers de Learning Rate

#### MultiStepLR (Recomendado)

```bash
--lr-scheduler MultiStepLR \
--milestones 10 15 20 \
--gamma 0.1
```

Reduz o LR em 10x nas épocas 10, 15 e 20.

#### StepLR

```bash
--lr-scheduler StepLR \
--step-size 5 \
--gamma 0.5
```

Reduz o LR pela metade a cada 5 épocas.

### Validação com RetinaFace

Para validação mais rigorosa, use RetinaFace:

```bash
--use-retinaface-validation \
--no-face-policy exclude \
--retinaface-conf-threshold 0.5
```

### Treinamento Distribuído (Multi-GPU)

```bash
torchrun --nproc_per_node=4 finetune.py \
    --pretrained-checkpoint weights/model.ckpt \
    --root /path/to/dataset/ \
    --network mobilenetv3_large \
    --strategy FULL_FINETUNE \
    --batch-size 64
```

---

## 🐛 Troubleshooting

### Problema: "Checkpoint not found"

**Solução:** Verifique o caminho do checkpoint:

```bash
ls -lh weights/mobilenetv3_large_MCP_best.ckpt
```

### Problema: "Dataset directory does not exist"

**Solução:** Verifique se o caminho do dataset está correto:

```bash
ls /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/
```

### Problema: "Out of memory" (OOM)

**Soluções:**
1. Reduza o batch size:
   ```bash
   --batch-size 32  # ou menor
   ```
2. Use estratégia HEAD_ONLY (menos parâmetros treináveis)
3. Reduza o número de workers:
   ```bash
   --num-workers 4
   ```

### Problema: Accuracy não melhora

**Soluções:**
1. Verifique se o learning rate está adequado:
   - Muito alto: reduce o LR
   - Muito baixo: aumente o LR
2. Tente outra estratégia (ex: FULL_FINETUNE se estava usando HEAD_ONLY)
3. Aumente o número de épocas
4. Verifique a qualidade do dataset

### Problema: Overfitting

**Soluções:**
1. Use estratégia HEAD_ONLY
2. Aumente weight decay:
   ```bash
   --weight-decay 1e-3
   ```
3. Use data augmentation (já incluído: random horizontal flip)
4. Reduza o número de épocas ou use early stopping mais agressivo

### Problema: Modelo não está aprendendo novas classes

**Soluções:**
1. Verifique se o classification head foi criado corretamente (número de classes)
2. Use learning rate maior para o head:
   ```bash
   --lr-head 0.2
   ```
3. Verifique se o dataset está bem balanceado

---

## 📈 Boas Práticas

### 1. Escolha da Estratégia

- **Dataset pequeno (< 5K imagens):** HEAD_ONLY
- **Dataset médio (5K - 10K):** PROGRESSIVE
- **Dataset grande (> 10K):** FULL_FINETUNE

### 2. Learning Rates

- Comece com os valores padrão
- Se não houver melhoria após 5 épocas, reduza o LR
- Se houver instabilidade (loss muito alto), reduza o LR

### 3. Monitoramento

- Acompanhe tanto a accuracy interna quanto a similaridade no LFW
- Se a similaridade no LFW diminuir muito, pode estar havendo overfitting
- Use os gráficos ROC salvos para análise detalhada

### 4. Checkpoints

- Sempre salve o melhor modelo (`_best.ckpt`)
- Mantenha também o último checkpoint para continuar treinamento
- Documente qual estratégia e hiperparâmetros foram usados

---

## 📚 Referências e Recursos

- [Documentação do PyTorch](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Artigo: "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

---

## 📝 Exemplo Completo

Aqui está um exemplo completo de fine-tuning end-to-end:

```bash
# 1. Verificar dataset
python -c "from utils.dataset import ImageFolder; ds = ImageFolder('/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/'); print(f'Classes: {len(set([l for _, l in ds.samples]))}, Images: {len(ds)}')"

# 2. Fine-tuning com FULL_FINETUNE
python finetune.py \
    --pretrained-checkpoint weights/mobilenetv3_large_MCP_best.ckpt \
    --root /dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/ \
    --network mobilenetv3_large \
    --strategy FULL_FINETUNE \
    --classifier MCP \
    --epochs 20 \
    --batch-size 64 \
    --lr-backbone 0.01 \
    --lr-head 0.1 \
    --lr-scheduler MultiStepLR \
    --milestones 10 15 \
    --gamma 0.1 \
    --save-path weights/finetuned_mydataset \
    --val-dataset lfw \
    --val-root data/lfw/val \
    --print-freq 50

# 3. Avaliar modelo fine-tuned
python evaluate.py \
    --model-path weights/finetuned_mydataset/mobilenetv3_large_MCP_finetuned_full_best.ckpt \
    --network mobilenetv3_large \
    --val-dataset lfw \
    --val-root data/lfw/val
```

---

**Última atualização:** 2024

**Autor:** Sistema de Fine-tuning para Face Recognition

---

## 🎓 Glossário

- **Backbone:** Parte do modelo responsável por extrair features (ex: MobileNet, SphereNet)
- **Classification Head:** Camada final que classifica os embeddings em classes
- **Checkpoint:** Arquivo contendo os pesos do modelo em um determinado momento do treinamento
- **Embedding:** Representação vetorial de uma face (normalmente 512 dimensões)
- **Fine-tuning:** Processo de ajustar um modelo pré-treinado para um novo dataset
- **Learning Rate:** Taxa de aprendizado, controla o tamanho dos ajustes nos pesos
- **Overfitting:** Quando o modelo memoriza o dataset de treino mas não generaliza bem
- **State Dict:** Dicionário contendo os pesos de um modelo PyTorch
