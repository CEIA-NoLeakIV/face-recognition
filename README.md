# Face Recognition Training Framework

Framework de treinamento para reconhecimento facial baseado em CNNs com suporte a múltiplas arquiteturas e loss functions.

## Visão Geral

Este projeto implementa um pipeline completo para treinamento e avaliação de modelos de reconhecimento facial. Suporta diferentes arquiteturas de redes neurais e métodos de loss baseados em margem angular para aprendizado de embeddings discriminativos.

## 🆕 Novas Funcionalidades (v2.0)

### Sistema de Tracking Automático de Treinamento

- **TrainingTracker**: Sistema completo que rastreia métricas automaticamente durante o treinamento
- **Plots Automáticos**: Gera visualizações profissionais a cada época sem necessidade de flags
- **Relatório Final**: Cria relatório completo com todas as métricas e gráficos ao final do treinamento
- **Histórico Preservado**: Salva histórico completo em checkpoints para análise posterior

### Métricas Expandidas

Além das métricas originais, agora calcula automaticamente:

- **F1 Score**: Harmonic mean de precision e recall
- **Precision**: Proporção de verdadeiros positivos entre predições positivas
- **Recall**: Proporção de verdadeiros positivos identificados corretamente
- **AUC Score**: Área sob a curva ROC
- **ROC Curve**: Curva ROC completa com visualização
- **Confusion Matrix**: Matriz de confusão com evolução ao longo do treinamento
- **TAR/FAR/FRR**: Métricas biométricas (True Accept Rate, False Accept Rate, False Reject Rate)

### Visualizações Geradas

O sistema agora gera automaticamente:

**Durante o treinamento (por época):**
- `weights/epoch_XXX/lfw_roc_curve.png` - ROC curve
- `weights/epoch_XXX/lfw_confusion_matrix.png` - Confusion matrix

**Ao final do treinamento:**
- `weights/final_report/training_curves.png` - Curvas de loss, accuracy, F1, AUC, similarity
- `weights/final_report/confusion_matrix_evolution.png` - Evolução da matriz (início/meio/fim)
- `weights/final_report/learning_rate_schedule.png` - Schedule do learning rate
- `weights/final_report/all_metrics_overview.png` - Overview de todas as métricas
- `weights/final_report/training_history.json` - Histórico completo em JSON
- `weights/final_report/training_summary.txt` - Resumo estatístico

## Arquiteturas Suportadas

### Backbones Disponíveis

- **SphereFace Networks**: sphere20, sphere36, sphere64
- **MobileNet Family**: mobilenetv1, mobilenetv2, mobilenetv3_small, mobilenetv3_large

### Loss Functions

- **MCP (Margin Cosine Product)**: Implementação do CosFace
- **AL (Angle Linear)**: Implementação do SphereFace
- **ARC**: Implementação do ArcFace
- **L (Linear)**: Classificador linear padrão

## Datasets Suportados

### Datasets de Treinamento

O framework suporta os seguintes datasets para treinamento:

- **WebFace**: 10,572 identidades
- **VggFace2**: 8,631 identidades
- **MS1M**: 85,742 identidades
- **VggFaceHQ**: 9,131 identidades (imagens de alta qualidade com tamanhos variados)

### Datasets de Validação

O framework suporta os seguintes datasets para validação:

- **LFW (Labeled Faces in the Wild)**: Benchmark padrão para reconhecimento facial
- **CelebA**: Dataset de celebridades com múltiplas imagens por identidade

## Estrutura do Projeto

```
├── models/                  # Arquiteturas das redes
├── utils/                   # Utilitários e métricas
│   ├── dataset.py          # Carregamento de dados
│   ├── metrics.py          # Loss functions
│   ├── general.py          # Funções auxiliares
│   ├── validation_split.py # Split de validação
│   └── training_tracker.py # 🆕 Sistema de tracking
├── train.py                # Script de treinamento (atualizado)
├── evaluate.py             # Avaliação em LFW/CelebA (atualizado)
├── inference.py            # Inferência e comparação
└── requirements.txt        # Dependências do projeto (atualizado)
```

## Instalação

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Dependências Atualizadas

```txt
numpy==2.1.3
opencv-python==4.10.0.84
pillow==11.0.0
tqdm==4.67.1
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
uniface
onnxruntime
scikit-learn==1.5.2      # 🆕 Para métricas
matplotlib==3.9.2         # 🆕 Para visualizações
seaborn==0.13.2          # 🆕 Para plots profissionais
pandas==2.2.3            # 🆕 Para análise de dados
```

## Uso

### Treinamento

Comando básico para treinamento (sem mudanças no comando):

```bash
python train.py \
    --root <caminho_dataset> \
    --database <nome_database> \
    --network <arquitetura> \
    --classifier <tipo_loss> \
    --val-dataset <dataset_validacao> \
    --val-root <caminho_validacao> \
    --batch-size <tamanho_batch> \
    --epochs <num_epocas> \
    --lr <taxa_aprendizado>
```

#### Parâmetros Principais

**Dataset de Treinamento:**
- `--root`: Caminho para o diretório das imagens de treinamento
- `--database`: Nome do dataset (WebFace, VggFace2, MS1M, VggFaceHQ)

**Dataset de Validação:**
- `--val-dataset`: Dataset de validação (lfw, celeba) - Padrão: lfw
- `--val-root`: Caminho para o diretório do dataset de validação - Padrão: data/lfw/val

**Modelo:**
- `--network`: Arquitetura da rede (sphere20, mobilenetv3_large, etc.)
- `--classifier`: Tipo de loss function (MCP, AL, ARC, L)

**Hiperparâmetros:**
- `--batch-size`: Tamanho do batch (padrão: 512)
- `--epochs`: Número de épocas (padrão: 30)
- `--lr`: Taxa de aprendizado inicial (padrão: 0.1)
- `--lr-scheduler`: Tipo de scheduler (StepLR ou MultiStepLR)
- `--milestones`: Épocas para redução da taxa de aprendizado (padrão: [10, 20, 25])
- `--gamma`: Fator de redução do learning rate (padrão: 0.1)
- `--momentum`: Momentum do SGD (padrão: 0.9)
- `--weight-decay`: Weight decay (padrão: 5e-4)

**Outros:**
- `--save-path`: Diretório para salvar checkpoints (padrão: weights)
- `--checkpoint`: Caminho para checkpoint para retomar treinamento
- `--num-workers`: Número de workers para DataLoader (padrão: 8)
- `--print-freq`: Frequência de impressão de logs (padrão: 100)

### 🆕 O Que Acontece Automaticamente Durante o Treinamento

O sistema agora:

1. **Rastreia todas as métricas** automaticamente (loss, accuracy, F1, precision, recall, AUC, etc.)
2. **Salva plots a cada época** em `weights/epoch_XXX/`
3. **Exibe métricas completas** nos logs a cada época
4. **Gera relatório final** completo em `weights/final_report/` ao terminar
5. **Preserva histórico** em checkpoints (pode retomar com histórico intacto)

### 🆕 Exemplo de Logs Durante Treinamento

```
==================================================
External Validation - Epoch 1
==================================================
LFW - Simplified Evaluation (Positive Pairs Only):
Mean Similarity: 0.6256 | Standard Deviation: 0.1339

Additional Metrics (Threshold: 0.3847):
  Accuracy:  0.9650
  F1 Score:  0.9823
  Precision: 0.9651
  Recall:    1.0000
  AUC Score: 0.9956

Confusion Matrix:
  TN:     0  FP:   105
  FN:     0  TP:  2895

Internal Validation (VggFace2 subset): 0.8523

External Validation Metrics (LFW):
  Mean Similarity: 0.6256
  Best Threshold:  0.3847
  Accuracy:        0.9650
  F1 Score:        0.9823
  Precision:       0.9651
  Recall:          1.0000
  AUC Score:       0.9956
==================================================

✅ ROC curve saved to: weights/epoch_001/lfw_roc_curve.png
✅ Confusion matrix saved to: weights/epoch_001/lfw_confusion_matrix.png
```

### Retomar Treinamento

Para continuar um treinamento anterior:

```bash
python train.py \
    --checkpoint weights/sphere20_MCP_last.ckpt \
    --root data/train/webface/ \
    --database WebFace \
    --network sphere20 \
    --classifier MCP
```

**🆕 O histórico de métricas é preservado automaticamente!**

### Avaliação

Avaliação standalone em LFW ou CelebA:

```bash
python evaluate.py
```

O script avalia os modelos treinados e calcula **todas as métricas** incluindo as novas (F1, Precision, Recall, AUC, ROC, Confusion Matrix).

### 🆕 Avaliação com Novas Métricas no Notebook

O notebook `1.Notebooks/Eval.ipynb` foi atualizado com novas células para:

- Calcular todas as 12+ métricas automaticamente
- Visualizar ROC Curve
- Visualizar Confusion Matrix
- Analisar sensibilidade ao threshold
- Exportar resultados completos em JSON e CSV

### Inferência

#### Comparação entre Duas Imagens

```bash
python inference.py
```

O script de inferência permite:
- Comparar duas imagens faciais
- Extrair embeddings de múltiplas imagens
- Calcular similaridade entre faces

## Detalhes de Implementação

### Pré-processamento

As imagens são processadas da seguinte forma:
- **Resize obrigatório para 112x112 pixels** (aplicado automaticamente)
- Normalização: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Formato: RGB

### Data Augmentation

**Durante o treinamento:**
- Resize para 112x112 (obrigatório)
- Random horizontal flip
- Normalização

**Na avaliação:**
- Resize para 112x112 (obrigatório)
- Test-time augmentation com flip horizontal
- Concatenação de features da imagem original e flipped

### Otimização

- **Otimizador**: SGD com momentum 0.9
- **Weight Decay**: 5e-4
- **Learning Rate Scheduler**: MultiStepLR com redução por fator de 0.1 nos milestones

### Validação e Checkpoints

O treinamento inclui:

1. **Split de Validação Interno**: 10% do dataset de treino separado para validação de classificação
2. **Avaliação Externa (LFW/CelebA)**: Executada a cada época para avaliar qualidade dos embeddings
3. **Early Stopping**: Patience de 10 épocas sem melhoria
4. **Salvamento de Modelos**:
   - `*_last.ckpt`: Último checkpoint (salvo a cada época)
   - `*_best.ckpt`: Melhor modelo baseado nas métricas de validação

### 🆕 Conteúdo dos Checkpoints (Atualizado)

Os checkpoints agora salvam:
- Estado do modelo (pesos)
- Estado do otimizador
- Estado do scheduler
- Época atual
- Argumentos de treinamento
- **Histórico completo de treinamento** (todas as métricas de todas as épocas)
- **Melhores métricas alcançadas** (similarity, AUC, F1)

## Estrutura de Dados Esperada

### Dataset de Treinamento

```
data/train/
└── <dataset_name>/
    ├── identity_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── identity_2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

### Dataset LFW para Validação

```
data/lfw/val/
├── lfw_ann.txt
└── <pessoa_nome>/
    ├── <pessoa_nome>_0001.jpg
    ├── <pessoa_nome>_0002.jpg
    └── ...
```

### Dataset CelebA para Validação

```
data/celeba/
├── celeba_pairs.txt
└── img_align_celeba/
    └── img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

## 🆕 Estrutura de Arquivos Gerados

Após o treinamento, a seguinte estrutura é criada automaticamente:

```
weights/
├── <model>_<classifier>_best.ckpt    # Melhor modelo
├── <model>_<classifier>_last.ckpt    # Último checkpoint
│
├── epoch_001/                         # Plots de cada época
│   ├── lfw_roc_curve.png
│   └── lfw_confusion_matrix.png
├── epoch_002/
│   ├── lfw_roc_curve.png
│   └── lfw_confusion_matrix.png
├── ...
├── epoch_030/
│   ├── lfw_roc_curve.png
│   └── lfw_confusion_matrix.png
│
└── final_report/                      # Relatório final completo
    ├── training_curves.png            # Curvas de Loss, Accuracy, F1, AUC, Similarity
    ├── confusion_matrix_evolution.png # Evolução da matriz (início, meio, fim)
    ├── learning_rate_schedule.png     # Schedule do learning rate
    ├── all_metrics_overview.png       # Overview de todas as métricas
    ├── training_history.json          # Histórico completo em JSON
    └── training_summary.txt           # Resumo estatístico em texto
```

## Métricas

### Durante o Treinamento

- **Loss**: CrossEntropyLoss
- **Training Accuracy**: Acurácia de classificação no batch atual
- **Internal Validation Accuracy**: Acurácia no subset de validação interna (10% do dataset de treino)
- **External Validation Metrics**: Métricas completas no dataset de validação externo (LFW ou CelebA)

### 🆕 Métricas de Validação Externa (Completas)

**Similaridade:**
- Mean Similarity
- Standard Deviation
- Min/Max/Median

**Classificação:**
- Accuracy (com threshold automático)
- F1 Score
- Precision
- Recall
- AUC Score

**Biométricas:**
- TAR (True Acceptance Rate)
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)

**Matrizes:**
- Confusion Matrix (TN, FP, FN, TP)
- ROC Curve completa

### Logs

O treinamento imprime logs a cada `--print-freq` batches com todas as métricas.

Ao final de cada época:
- Acurácia de validação interna
- **Todas as métricas de validação externa** (12+ métricas)
- Salvamento automático de plots

### 🆕 Critério de Best Model

O melhor modelo é selecionado com base em **múltiplos critérios**:
- Mean Similarity (principal)
- AUC Score
- F1 Score

Qualquer melhoria em qualquer uma dessas métricas salva o modelo como `_best.ckpt`.

## Características Técnicas

### Embedding Dimension

Todos os modelos geram embeddings de 512 dimensões por padrão.

### Suporte a GPU

O framework detecta automaticamente GPUs disponíveis e move os modelos para CUDA quando possível.

### Treinamento Distribuído

Suporte para treinamento multi-GPU com DistributedDataParallel:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<num_gpus> \
    train.py --world-size <num_gpus> [outros argumentos]
```

## 🆕 Análise dos Resultados

### Interpretando as Visualizações

**training_curves.png** (Principal):
- Top-Left: Training Loss com melhor época marcada
- Top-Right: Curvas de Accuracy (Train/Val/External)
- Bottom-Left: Métricas de classificação (F1, Precision, Recall)
- Bottom-Right: AUC e Similarity (dual Y-axis)

**confusion_matrix_evolution.png**:
- Mostra como o modelo aprende ao longo do tempo
- Três épocas: início, meio, fim
- Visualiza redução de erros (FP/FN)

**learning_rate_schedule.png**:
- Curva do learning rate ao longo das épocas
- Mostra milestones de decay

**all_metrics_overview.png**:
- Comparação lado-a-lado de todas as métricas
- 6 gráficos individuais

### Usando o Histórico JSON

```python
import json

# Carregar histórico
history = json.load(open('weights/final_report/training_history.json'))

# Acessar métricas
epochs = history['epochs']
train_loss = history['train_loss']
external_auc = history['external_auc']
f1_scores = history['external_f1']

# Plotar custom
import matplotlib.pyplot as plt
plt.plot(epochs, external_auc)
plt.show()
```

## Licença

Este projeto é fornecido para fins educacionais e de pesquisa.

## 🆕 Changelog

### v2.0.0 (2024-11-04)

**Adicionado:**
- Sistema TrainingTracker para rastreamento automático de métricas
- 12+ novas métricas de avaliação (F1, Precision, Recall, AUC, TAR, FAR, FRR)
- Visualizações automáticas (4 tipos de plots profissionais)
- Relatório final completo com todas as métricas
- Histórico preservado em checkpoints
- ROC Curves automáticas
- Confusion Matrix com evolução
- Learning Rate schedule visualization

**Modificado:**
- `train.py` - Integração com TrainingTracker
- `evaluate.py` - Métricas expandidas com visualizações
- `Eval.ipynb` - 7 novas células com análises completas
- `requirements.txt` - Novas dependências (scikit-learn, matplotlib, seaborn, pandas)

**Comportamento:**
- Plots agora são **sempre** salvos automaticamente (sem necessidade de flags)
- Todas as métricas são calculadas automaticamente a cada época
- Relatório final é gerado automaticamente ao término do treinamento
