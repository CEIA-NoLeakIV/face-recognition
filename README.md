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
- **ARC**: Implementação do ArcFace
- **L (Linear)**: Classificador linear padrão

## Datasets Suportados

O framework suporta os seguintes datasets:

- **WebFace**: 10,572 identidades
- **VggFace2**: 8,631 identidades
- **MS1M**: 85,742 identidades
- **VggFaceHQ**: 9,131 identidades

## Estrutura do Projeto

```
├── models/                  # Arquiteturas das redes
├── utils/                   # Utilitários e métricas
│   ├── dataset.py          # Carregamento de dados
│   ├── metrics.py          # Loss functions
│   ├── general.py          # Funções auxiliares
│   └── validation_split.py # Split de validação
├── train.py                # Script de treinamento
├── evaluate.py             # Avaliação em LFW
├── inference.py            # Inferência e comparação
└── requirements.txt        # Dependências do projeto
```

## Instalação

### Instalar Dependências

```bash
pip install -r requirements.txt
```

### Resolução de Problemas com CUDA

Caso o PyTorch não reconheça a versão do CUDA instalada no sistema, instale manualmente uma versão compatível:

**Para CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Para CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Para CPU apenas:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verifique a versão do CUDA disponível com:
```bash
nvidia-smi
```

## Uso

### Treinamento

Comando básico para treinamento:

```bash
python train.py \
    --root <caminho_dataset> \
    --database <nome_database> \
    --network <arquitetura> \
    --classifier <tipo_loss> \
    --batch-size <tamanho_batch> \
    --epochs <num_epocas> \
    --lr <taxa_aprendizado>
```

#### Parâmetros Principais

**Dataset:**
- `--root`: Caminho para o diretório das imagens de treinamento
- `--database`: Nome do dataset (WebFace, VggFace2, MS1M, VggFaceHQ)

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

#### Exemplo de Treinamento

```bash
python train.py \
    --root data/train/webface/ \
    --database WebFace \
    --network sphere20 \
    --classifier MCP \
    --batch-size 256 \
    --epochs 30 \
    --lr 0.1 \
    --milestones 10 20 25
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

### Avaliação

Avaliação em LFW (Labeled Faces in the Wild):

```bash
python evaluate.py
```

O script avalia os modelos treinados e calcula a similaridade média entre pares de faces positivos no dataset LFW.

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
- Redimensionamento para 112x112 pixels
- Normalização: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Formato: RGB

### Data

Durante o treinamento:
- Random horizontal flip
- Resize para 112x112

Na avaliação:
- Test-time augmentation com flip horizontal
- Concatenação de features da imagem original e flipped

### Otimização

- **Otimizador**: SGD com momentum 0.9
- **Weight Decay**: 5e-4
- **Learning Rate Scheduler**: MultiStepLR com redução por fator de 0.1 nos milestones

### Validação e Checkpoints

O treinamento inclui:

1. **Split de Validação**: 10% do dataset de treino separado para validação
2. **Avaliação LFW**: Executada a cada época no processo de rank 0
3. **Early Stopping**: Patience de 10 épocas sem melhoria
4. **Salvamento de Modelos**:
   - `*_last.ckpt`: Último checkpoint (salvo a cada época)
   - `*_best.ckpt`: Melhor modelo baseado na similaridade LFW

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

### Dataset LFW para Validação

```
data/val/
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

Cada linha representa um par positivo (mesma pessoa).

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
- **Validation Accuracy**: Acurácia no subset de validação interna
- **LFW Similarity**: Similaridade média entre pares positivos no LFW

### Logs

O treinamento imprime logs a cada `--print-freq` batches:
- Época atual
- Loss médio
- Acurácia média
- Learning rate atual
- Tempo de processamento

## Referências

Este projeto implementa conceitos dos seguintes trabalhos:

- **SphereFace**: Deep Hypersphere Embedding for Face Recognition (Liu et al., 2017)
- **CosFace**: Large Margin Cosine Loss for Deep Face Recognition (Wang et al., 2018)
- **ArcFace**: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., 2019)

## Licença

Este projeto é fornecido para fins educacionais e de pesquisa.
