# Face Recognition Training Framework

Framework de treinamento para reconhecimento facial baseado em CNNs com suporte a múltiplas arquiteturas e loss functions.

## Visão Geral

Este projeto implementa um pipeline completo para treinamento e avaliação de modelos de reconhecimento facial. Suporta diferentes arquiteturas de redes neurais e métodos de loss baseados em margem angular para aprendizado de embeddings discriminativos.

## Arquiteturas Suportadas

### Backbones Disponíveis

- **SphereFace Networks**: sphere20, sphere36, sphere64
- **MobileNet Family**: mobilenetv1, mobilenetv2, mobilenetv3_small, mobilenetv3_large

### Loss Functions

- **MCP (Margin Cosine Product)**: Implementação do CosFace
- **AL (Angle Linear)**: Implementação do SphereFace
- **ARC**: Implementação do ArcFace (*)
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
│   └── validation_split.py # Split de validação
├── train.py                # Script de treinamento
├── evaluate.py             # Avaliação em LFW/CelebA
├── inference.py            # Inferência e comparação
└── requirements.txt        # Dependências do projeto
```

## Instalação

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Resolução de Problemas com CUDA

Caso o PyTorch não reconheça a versão do CUDA instalada no sistema, instale manualmente uma versão compatível

## Uso

### Treinamento

Comando básico para treinamento:

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
- `--network`: Arquitetura da rede (sphere20, sphere36, sphere64, mobilenetv1, mobilenetv2, mobilenetv3_small, mobilenetv3_large)
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

### Avaliação

Avaliação standalone em LFW ou CelebA:

**Avaliar em LFW (padrão):**
```bash
python evaluate.py
```

**Avaliar em CelebA:**
```bash
# Edite evaluate.py e ajuste os parâmetros na chamada da função eval()
# Exemplo:
eval(model, model_path='weights/model.pth', val_dataset='celeba', val_root='data/celeba')
```

O script avalia os modelos treinados e calcula a similaridade média entre pares de faces positivos no dataset de validação.

### Inferência

#### Comparação entre Duas Imagens

```bash
python inference.py
```

O script de inferência permite:
- Comparar duas imagens faciais
- Extrair embeddings de múltiplas imagens
- Calcular similaridade entre faces

Edite as variáveis no final do arquivo `inference.py` para configurar:
- Nome do modelo
- Caminho dos pesos
- Caminhos das imagens
- Threshold de similaridade

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
2. **Avaliação Externa (LFW/CelebA)**: Executada a cada época no processo de rank 0 para avaliar qualidade dos embeddings
3. **Early Stopping**: Patience de 10 épocas sem melhoria na similaridade do dataset de validação externo
4. **Salvamento de Modelos**:
   - `*_last.ckpt`: Último checkpoint (salvo a cada época)
   - `*_best.ckpt`: Melhor modelo baseado na similaridade do dataset de validação externo (LFW ou CelebA)

### Conteúdo dos Checkpoints

Os checkpoints salvos contêm:
- Estado do modelo (pesos)
- Estado do otimizador
- Estado do scheduler
- Época atual
- Argumentos de treinamento

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

Cada subdiretório representa uma identidade diferente.

**Nota**: O dataset VggFaceHQ pode conter imagens de tamanhos variados, mas todas serão automaticamente redimensionadas para 112x112 durante o pré-processamento.

### Dataset LFW para Validação

```
data/lfw/val/
├── lfw_ann.txt
└── <pessoa_nome>/
    ├── <pessoa_nome>_0001.jpg
    ├── <pessoa_nome>_0002.jpg
    └── ...
```

#### Formato do arquivo lfw_ann.txt

```
num_pairs
pessoa1_nome 0001 0002
pessoa2_nome 0001 0003
...
```

Cada linha representa um par positivo (mesma pessoa) com o formato: `nome_pessoa numero_img1 numero_img2`

### Dataset CelebA para Validação

```
data/celeba/
├── celeba_pairs.txt
└── img_align_celeba/
    └── img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        ├── 000003.jpg
        └── ...
```

#### Formato do arquivo celeba_pairs.txt

```
header_line
000001.jpg 000045.jpg
000001.jpg 000123.jpg
000002.jpg 000089.jpg
...
```

Cada linha representa um par positivo (mesma pessoa) com o formato: `imagem1.jpg imagem2.jpg`

**Nota**: O arquivo deve conter apenas pares de imagens da mesma identidade. A primeira linha é um header e é ignorada.

## Características Técnicas

### Embedding Dimension

Todos os modelos geram embeddings de 512 dimensões por padrão.

### Suporte a GPU

O framework detecta automaticamente GPUs disponíveis e move os modelos para CUDA quando possível. Para forçar uso de CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
python train.py [argumentos]
```

### Treinamento Distribuído

Suporte para treinamento multi-GPU com DistributedDataParallel:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<num_gpus> \
    train.py --world-size <num_gpus> [outros argumentos]
```

### Treinamento Determinístico

Para resultados reprodutíveis:

```bash
python train.py --use-deterministic-algorithms [outros argumentos]
```

Nota: Pode reduzir a performance.

## Métricas

### Durante o Treinamento

- **Loss**: CrossEntropyLoss
- **Training Accuracy**: Acurácia de classificação no batch atual
- **Internal Validation Accuracy**: Acurácia no subset de validação interna (10% do dataset de treino)
- **External Validation Similarity**: Similaridade média entre pares positivos no dataset de validação externo (LFW ou CelebA)

### Logs

O treinamento imprime logs a cada `--print-freq` batches:
- Época atual
- Loss médio
- Acurácia média de treinamento
- Learning rate atual
- Tempo de processamento

Ao final de cada época:
- Acurácia de validação interna (classificação no subset do dataset de treino)
- Similaridade de validação externa (LFW ou CelebA)

### Critério de Best Model

O melhor modelo é selecionado com base na **similaridade média do dataset de validação externo** (LFW ou CelebA), não na acurácia de classificação interna. Isso garante que o modelo aprenda embeddings discriminativos que generalizam bem para identidades não vistas.

## Licença

Este projeto é fornecido para fins educacionais e de pesquisa.
