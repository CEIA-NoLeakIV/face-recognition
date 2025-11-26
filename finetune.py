"""
Script de Fine-tuning para Face Recognition

Este script permite realizar fine-tuning de modelos pré-treinados em novos datasets
usando 3 estratégias diferentes:

1. FULL_FINETUNE: Fine-tuning completo do backbone e classification head com learning rate reduzido
2. HEAD_ONLY: Apenas o classification head é treinado (backbone congelado)
3. PROGRESSIVE: Treinamento progressivo (primeiro head, depois descongela gradualmente o backbone)
"""

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import evaluate
from utils.dataset import ImageFolder
from utils.metrics import MarginCosineProduct, AngleLinear
from utils.general import (
    setup_seed,
    reduce_tensor,
    save_on_master,
    calculate_accuracy,
    init_distributed_mode,
    AverageMeter,
    EarlyStopping,
    LOGGER,
)
from utils.face_validation import FaceValidator, print_validation_summary
from utils.validation_split import create_validation_split

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)


def count_classes_in_dataset(root_dir):
    """Conta o número de classes (identidades) no dataset."""
    if not os.path.exists(root_dir):
        raise ValueError(f"Dataset directory does not exist: {root_dir}")
    
    class_names = sorted([entry.name for entry in os.scandir(root_dir) if entry.is_dir()])
    num_classes = len(class_names)
    
    LOGGER.info(f"Found {num_classes} classes in dataset: {root_dir}")
    return num_classes


def get_classification_head(classifier, embedding_dim, num_classes):
    """Cria o classification head apropriado."""
    classifiers = {
        'MCP': MarginCosineProduct(embedding_dim, num_classes),
        'AL': AngleLinear(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False)
    }

    if classifier not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier}")

    return classifiers[classifier]


def freeze_model_backbone(model):
    """Congela todos os parâmetros do backbone do modelo."""
    for param in model.parameters():
        param.requires_grad = False
    LOGGER.info("Backbone model frozen.")


def unfreeze_model_backbone(model):
    """Descongela todos os parâmetros do backbone do modelo."""
    for param in model.parameters():
        param.requires_grad = True
    LOGGER.info("Backbone model unfrozen.")


def get_trainable_params_count(model):
    """Conta o número de parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def progressive_unfreeze(model, epoch, total_epochs):
    """
    Estratégia de descongelamento progressivo para modelos MobileNet.
    Descongela layers gradualmente do final para o início.
    """
    # Para modelos MobileNet, temos features (backbone) e output_layer
    if hasattr(model, 'features'):
        # MobileNetV2/V3
        features = model.features
        num_features = len(list(features))
        
        # Dividir em 3 etapas: 1/3, 2/3, todos
        if epoch < total_epochs // 3:
            # Primeira etapa: congelar tudo exceto última parte
            for i, module in enumerate(features):
                if i < num_features * 2 // 3:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
        elif epoch < total_epochs * 2 // 3:
            # Segunda etapa: descongelar até metade
            for i, module in enumerate(features):
                if i < num_features // 3:
                    for param in module.parameters():
                        param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            # Terceira etapa: descongelar tudo
            for param in model.parameters():
                param.requires_grad = True
    
    elif hasattr(model, 'layer4'):
        # SphereNet (sphere20/36/64)
        # Descongelar do layer4 para layer1
        if epoch < total_epochs // 3:
            # Apenas layer4
            for param in model.layer4.parameters():
                param.requires_grad = True
            for layer in [model.layer1, model.layer2, model.layer3]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif epoch < total_epochs * 2 // 3:
            # layer3 e layer4
            for layer in [model.layer3, model.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True
            for layer in [model.layer1, model.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            # Todos os layers
            for param in model.parameters():
                param.requires_grad = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for face recognition models"
    )

    parser.add_argument(
        '--pretrained-checkpoint',
        type=str,
        required=True,
        help='Path to the pretrained model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        default='/dados/datasets/aligned_112x112/vggface2_dataset_all_splits_merged/',
        help='Path to the root directory of fine-tuning dataset'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='FULL_FINETUNE',
        choices=['FULL_FINETUNE', 'HEAD_ONLY', 'PROGRESSIVE'],
        help='Fine-tuning strategy: FULL_FINETUNE (full fine-tuning with reduced LR), '
             'HEAD_ONLY (freeze backbone, train only head), '
             'PROGRESSIVE (progressive unfreezing)'
    )

    parser.add_argument(
        '--network',
        type=str,
        required=True,
        choices=[
            'sphere20', 'sphere36', 'sphere64', 'mobilenetv1',
            'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large'
        ],
        help='Network architecture (must match pretrained model)'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='MCP',
        choices=['MCP', 'AL', 'L'],
        help='Type of classifier to use: MCP (MarginCosineProduct/CosFace), AL (SphereFace), L (Linear)'
    )

    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for fine-tuning')
    
    # Learning rates - diferentes para cada estratégia
    parser.add_argument(
        '--lr-backbone',
        type=float,
        default=None,
        help='Learning rate for backbone (if None, will use strategy defaults)'
    )
    parser.add_argument(
        '--lr-head',
        type=float,
        default=None,
        help='Learning rate for classification head (if None, will use strategy defaults)'
    )
    
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay for optimizer'
    )

    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='MultiStepLR',
        choices=['StepLR', 'MultiStepLR'],
        help='Learning rate scheduler type'
    )
    parser.add_argument('--step-size', type=int, default=10, help='Step size for StepLR')
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 15],
        help='Milestones for MultiStepLR'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Gamma for LR scheduler'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default='weights/finetuned',
        help='Path to save fine-tuned model checkpoints'
    )
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Frequency for printing training progress'
    )

    parser.add_argument(
        '--val-dataset',
        type=str,
        default='lfw',
        choices=['lfw', 'celeba'],
        help='Validation dataset to use'
    )
    parser.add_argument(
        '--val-root',
        type=str,
        default='data/lfw/val',
        help='Path to validation dataset root directory'
    )
    parser.add_argument(
        '--val-threshold',
        type=float,
        default=0.35,
        help='Similarity threshold for validation metrics'
    )

    parser.add_argument(
        '--use-retinaface-validation',
        action='store_true',
        help='Enable face validation using RetinaFace during evaluation'
    )
    parser.add_argument(
        '--no-face-policy',
        type=str,
        default='exclude',
        choices=['exclude', 'include'],
        help='Policy for images without detected faces'
    )
    parser.add_argument(
        '--retinaface-conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for RetinaFace'
    )
    parser.add_argument(
        '--face-validation-cache-dir',
        type=str,
        default='face_validation_cache',
        help='Directory to cache face validation results'
    )

    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only"
    )

    return parser.parse_args()


def train_one_epoch(
    model,
    classification_head,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    params
):
    """Treina uma época do modelo."""
    model.train()
    classification_head.train()
    
    losses = AverageMeter("Avg Loss", ":6.3f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    last_batch_idx = len(data_loader) - 1

    start_time = time.time()
    for batch_idx, (images, target) in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx

        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        embeddings = model(images)
        
        if isinstance(classification_head, torch.nn.Linear):
            output = classification_head(embeddings)
            cosine_output = output
        else:
            output_with_margin, cosine_output = classification_head(embeddings, target)
            output = output_with_margin

        loss = criterion(output, target)
        accuracy = calculate_accuracy(cosine_output, target)

        if hasattr(params, 'distributed') and params.distributed:
            reduced_loss = reduce_tensor(loss, params.world_size)
            accuracy = reduce_tensor(accuracy, params.world_size)
        else:
            reduced_loss = loss

        loss.backward()
        optimizer.step()

        losses.update(reduced_loss.item(), images.size(0))
        accuracy_meter.update(accuracy.item(), images.size(0))
        batch_time.update(time.time() - start_time)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        if batch_idx % params.print_freq == 0 or last_batch:
            lr = optimizer.param_groups[0]['lr']
            log = (
                f'Epoch: [{epoch+1}/{params.epochs}][{batch_idx:05d}/{len(data_loader):05d}] '
                f'Loss: {losses.avg:6.3f}, '
                f'Accuracy: {accuracy_meter.avg:4.2f}%, '
                f'LR: {lr:.5f} '
                f'Time: {batch_time.avg:4.3f}s'
            )
            LOGGER.info(log)

    log = (
        f'Epoch [{epoch}/{params.epochs}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )
    LOGGER.info(log)


def validate_model(model, classification_head, val_loader, device):
    """Valida o modelo no conjunto de validação interno."""
    model.eval()
    classification_head.eval()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            embeddings = model(images)
            if isinstance(classification_head, torch.nn.Linear):
                outputs = classification_head(embeddings)
            else:
                outputs, _ = classification_head(embeddings, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    model.train()
    classification_head.train()
    
    return accuracy


def main(params):
    init_distributed_mode(params)

    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Carregar checkpoint pré-treinado
    LOGGER.info(f'Loading pretrained checkpoint from: {params.pretrained_checkpoint}')
    if not os.path.exists(params.pretrained_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {params.pretrained_checkpoint}")
    
    checkpoint = torch.load(params.pretrained_checkpoint, map_location="cpu")
    
    # Identificar arquitetura e inicializar modelo
    networks = {
        'sphere20': sphere20,
        'sphere36': sphere36,
        'sphere64': sphere64,
        'mobilenetv1': MobileNetV1,
        'mobilenetv2': MobileNetV2,
        'mobilenetv3_small': mobilenet_v3_small,
        'mobilenetv3_large': mobilenet_v3_large
    }

    if params.network not in networks:
        raise ValueError(f"Unsupported network: {params.network}")

    model = networks[params.network](embedding_dim=512)
    
    # Carregar pesos do backbone do checkpoint
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        LOGGER.info(f"Loaded pretrained model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        LOGGER.info("Loaded pretrained model weights")
    
    model.to(device)

    # Contar classes no novo dataset
    num_classes = count_classes_in_dataset(params.root)
    
    # Criar novo classification head para o novo número de classes
    classification_head = get_classification_head(params.classifier, 512, num_classes)
    classification_head.to(device)

    # Aplicar estratégia de fine-tuning
    if params.strategy == 'HEAD_ONLY':
        freeze_model_backbone(model)
        LOGGER.info("Strategy: HEAD_ONLY - Only classification head will be trained")
    elif params.strategy == 'FULL_FINETUNE':
        unfreeze_model_backbone(model)
        LOGGER.info("Strategy: FULL_FINETUNE - Full model will be fine-tuned")
    elif params.strategy == 'PROGRESSIVE':
        freeze_model_backbone(model)
        LOGGER.info("Strategy: PROGRESSIVE - Progressive unfreezing will be applied")

    # Configurar learning rates baseado na estratégia
    if params.lr_backbone is None:
        if params.strategy == 'FULL_FINETUNE':
            lr_backbone = 0.01  # LR reduzido para fine-tuning completo
        elif params.strategy == 'HEAD_ONLY':
            lr_backbone = 0.0  # Não usado, mas necessário
        else:  # PROGRESSIVE
            lr_backbone = 0.005  # LR ainda menor para descongelamento progressivo
    else:
        lr_backbone = params.lr_backbone

    if params.lr_head is None:
        lr_head = 0.1  # LR maior para o novo head
    else:
        lr_head = params.lr_head

    LOGGER.info(f"Learning rates - Backbone: {lr_backbone}, Head: {lr_head}")

    # Contar parâmetros treináveis
    trainable_backbone = get_trainable_params_count(model)
    trainable_head = get_trainable_params_count(classification_head)
    LOGGER.info(f"Trainable parameters - Backbone: {trainable_backbone:,}, Head: {trainable_head:,}")

    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[params.local_rank], output_device=params.local_rank
        )
        classification_head = torch.nn.parallel.DistributedDataParallel(
            classification_head, device_ids=[params.local_rank], output_device=params.local_rank
        )
        model_without_ddp = model.module
        classification_head_without_ddp = classification_head.module
    else:
        model_without_ddp = model
        classification_head_without_ddp = classification_head

    os.makedirs(params.save_path, exist_ok=True)
    metrics_save_path = os.path.join(params.save_path, 'metrics')
    os.makedirs(metrics_save_path, exist_ok=True)

    face_validator = None
    if params.use_retinaface_validation:
        try:
            LOGGER.info("Initializing RetinaFace face validator...")
            face_validator = FaceValidator(
                conf_threshold=params.retinaface_conf_threshold,
                cache_dir=params.face_validation_cache_dir
            )
            LOGGER.info("✅ Face validator initialized successfully")
        except Exception as e:
            LOGGER.warning(f"⚠️  Could not initialize face validator: {e}")
            face_validator = None

    LOGGER.info('Loading fine-tuning dataset.')
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    full_dataset = ImageFolder(root=params.root, transform=train_transform)
    train_dataset, val_dataset = create_validation_split(full_dataset, val_split=0.1)

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    val_dataset.dataset.transform = val_transform

    LOGGER.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers,
        pin_memory=True
    )

    LOGGER.info(f'Dataset size: {len(train_loader.dataset)}, Number of classes: {num_classes}')

    # Configurar otimizador com diferentes LRs
    if params.strategy == 'HEAD_ONLY':
        optimizer = torch.optim.SGD(
            [{'params': classification_head_without_ddp.parameters()}],
            lr=lr_head,
            momentum=params.momentum,
            weight_decay=params.weight_decay
        )
    else:
        optimizer = torch.optim.SGD([
            {'params': model_without_ddp.parameters(), 'lr': lr_backbone},
            {'params': classification_head_without_ddp.parameters(), 'lr': lr_head}
        ],
            momentum=params.momentum,
            weight_decay=params.weight_decay
        )

    criterion = torch.nn.CrossEntropyLoss()

    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=params.milestones, gamma=params.gamma
        )
    elif params.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=params.step_size, gamma=params.gamma
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {params.lr_scheduler}")

    best_accuracy = 0.0
    curr_accuracy = 0.0
    early_stopping = EarlyStopping(patience=10)

    LOGGER.info(f'Fine-tuning started - Strategy: {params.strategy}, Network: {params.network}')
    
    for epoch in range(params.epochs):
        if params.distributed:
            train_sampler.set_epoch(epoch)
        
        # Aplicar descongelamento progressivo se necessário
        if params.strategy == 'PROGRESSIVE':
            progressive_unfreeze(model_without_ddp, epoch, params.epochs)
            # Recriar otimizador se necessário para incluir novos parâmetros
            if epoch > 0 and epoch == params.epochs // 3:
                # Recriar otimizador quando começamos a descongelar
                optimizer = torch.optim.SGD([
                    {'params': [p for p in model_without_ddp.parameters() if p.requires_grad], 'lr': lr_backbone},
                    {'params': classification_head_without_ddp.parameters(), 'lr': lr_head}
                ],
                    momentum=params.momentum,
                    weight_decay=params.weight_decay
                )
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=params.milestones, gamma=params.gamma
                )
                LOGGER.info(f"Optimizer recreated at epoch {epoch+1} for progressive unfreezing")
        
        train_one_epoch(
            model,
            classification_head,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            params
        )
        lr_scheduler.step()

        base_filename = f'{params.network}_{params.classifier}_finetuned_{params.strategy.lower()}'

        last_save_path = os.path.join(params.save_path, f'{base_filename}_last.ckpt')

        checkpoint_save = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'classification_head': classification_head_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'num_classes': num_classes,
            'strategy': params.strategy,
            'pretrained_checkpoint': params.pretrained_checkpoint,
            'args': params
        }

        save_on_master(checkpoint_save, last_save_path)

        if params.local_rank == 0:
            LOGGER.info(f'\n{"="*70}')
            LOGGER.info(f'EPOCH {epoch+1} VALIDATION METRICS')
            LOGGER.info(f'{"="*70}')
            
            epoch_metrics_path = os.path.join(metrics_save_path, f'epoch_{epoch+1}')
            os.makedirs(epoch_metrics_path, exist_ok=True)
            
            curr_accuracy, _, metrics = evaluate.eval(
                model_without_ddp, 
                device=device,
                val_dataset=params.val_dataset,
                val_root=params.val_root,
                compute_full_metrics=True,
                save_metrics_path=epoch_metrics_path,
                threshold=params.val_threshold,
                face_validator=face_validator,
                no_face_policy=params.no_face_policy
            )
            
            LOGGER.info(f'\nValidation Metrics (Threshold={params.val_threshold}):')
            
            if not metrics or len(metrics) <= 2:
                LOGGER.warning(f'  ⚠️  No valid pairs for evaluation!')
            else:
                if 'precision' in metrics:
                    LOGGER.info(f'  Precision: {metrics["precision"]:.4f}')
                if 'recall' in metrics:
                    LOGGER.info(f'  Recall:    {metrics["recall"]:.4f}')
                if 'f1' in metrics:
                    LOGGER.info(f'  F1-Score:  {metrics["f1"]:.4f}')
                if 'accuracy' in metrics:
                    LOGGER.info(f'  Accuracy:  {metrics["accuracy"]:.4f}')
            
            if 'auc' in metrics:
                LOGGER.info(f'\nROC Metrics:')
                LOGGER.info(f'  AUC: {metrics["auc"]:.4f}')
                if 'eer' in metrics:
                    LOGGER.info(f'  EER: {metrics["eer"]:.4f}')
            
            LOGGER.info(f'{"="*70}\n')
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True
            )
            
            val_accuracy = validate_model(model_without_ddp, classification_head_without_ddp, val_loader, device)
            LOGGER.info(f'Internal validation accuracy: {val_accuracy:.4f}\n')

        if early_stopping(epoch, curr_accuracy):
            break

        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            save_on_master(
                checkpoint_save,
                os.path.join(params.save_path, f'{base_filename}_best.ckpt')
            )
            LOGGER.info(
                f"New best {params.val_dataset.upper()} similarity: {best_accuracy:.4f}. "
                f"Model saved to {params.save_path} with `_best` postfix.\n"
            )

    if params.local_rank == 0:
        LOGGER.info(f'\n{"="*70}')
        LOGGER.info('FINAL COMPREHENSIVE EVALUATION')
        LOGGER.info(f'{"="*70}')
        
        final_metrics_path = os.path.join(metrics_save_path, 'final_evaluation')
        os.makedirs(final_metrics_path, exist_ok=True)
        
        _, _, final_metrics = evaluate.eval(
            model_without_ddp,
            device=device,
            val_dataset=params.val_dataset,
            val_root=params.val_root,
            compute_full_metrics=True,
            save_metrics_path=final_metrics_path,
            threshold=params.val_threshold,
            face_validator=face_validator,
            no_face_policy=params.no_face_policy
        )
        
        LOGGER.info(f'\nFinal Validation Metrics:')
        if final_metrics and len(final_metrics) > 2:
            if 'mean_similarity' in final_metrics:
                LOGGER.info(f'  Mean Similarity: {final_metrics["mean_similarity"]:.4f}')
            if 'accuracy' in final_metrics:
                LOGGER.info(f'  Accuracy:        {final_metrics["accuracy"]:.4f}')
            if 'auc' in final_metrics:
                LOGGER.info(f'  AUC:             {final_metrics["auc"]:.4f}')

    LOGGER.info('Fine-tuning completed.')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
