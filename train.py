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
from utils.training_tracker import TrainingTracker

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)

from utils.validation_split import create_validation_split

def parse_arguments():
    parser = argparse.ArgumentParser(description=("Command-line arguments for training a face recognition model"))

    parser.add_argument(
        '--root',
        type=str,
        default='data/train/webface_112x112/',
        help='Path to the root directory of training images.'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='WebFace',
        choices=['WebFace', 'VggFace2', "MS1M", 'VggFaceHQ'],
        help='Database to use for training. Options: WebFace, VggFace2, MS1M, VggFaceHQ.'
    )

    parser.add_argument(
        '--network',
        type=str,
        default='sphere20',
        choices=[
            'sphere20', 'sphere36', 'sphere64', 'mobilenetv1',
            'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large'
        ],
        help='Network architecture to use. Options: sphere20, sphere36, sphere64, mobile.'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='MCP',
        choices=['ARC', 'MCP', 'AL', 'L'],
        help='Type of classifier to use. Options: ARC (ArcFace), MCP (MarginCosineProduct), AL (SphereFace), L (Linear).'
    )

    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training. Default: 512.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training. Default: 30.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. Default: 0.1.')
    parser.add_argument(
        '--lr-scheduler',
        type=str,
        default='MultiStepLR',
        choices=['StepLR', 'MultiStepLR'],
        help='Learning rate scheduler type.'
    )
    parser.add_argument('--step-size', type=int, default=10, help='Period of learning rate decay for StepLR.')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='Multiplicative factor of learning rate decay for StepLR and ExponentialLR.'
    )
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='+',
        default=[10, 20, 25],
        help='List of epoch indices to reduce learning rate for MultiStepLR (ignored if StepLR is used).'
    )
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD optimizer. Default: 0.9.')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay for SGD optimizer. Default: 5e-4.'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights',
        help='Path to save model checkpoints. Default: `weights`.'
    )
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loader workers. Default: 8.')
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to continue training.")
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Frequency (in batches) for printing training progress. Default: 100.'
    )

    parser.add_argument(
        '--val-dataset',
        type=str,
        default='lfw',
        choices=['lfw', 'celeba'],
        help='Validation dataset to use. Options: lfw, celeba. Default: lfw.'
    )
    parser.add_argument(
        '--val-root',
        type=str,
        default='data/lfw/val',
        help='Path to validation dataset root directory. Default: data/lfw/val.'
    )
    parser.add_argument(
        '--val-threshold',
        type=float,
        default=0.35,
        help='Similarity threshold for validation metrics. Default: 0.35.'
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
        help='Policy for images without detected faces: exclude or include. Default: exclude.'
    )
    parser.add_argument(
        '--retinaface-conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for RetinaFace face detection. Default: 0.5.'
    )
    parser.add_argument(
        '--face-validation-cache-dir',
        type=str,
        default='face_validation_cache',
        help='Directory to cache face validation results. Default: face_validation_cache.'
    )

    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only."
    )

    return parser.parse_args()


def validate_model(model, classification_head, val_loader, criterion, device):
    """Valida o modelo e retorna accuracy e loss"""
    model.eval()
    classification_head.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            embeddings = model(images)
            if isinstance(classification_head, torch.nn.Linear):
                outputs = classification_head(embeddings)
            else:
                outputs, _ = classification_head(embeddings, targets)
            
            # Calcular loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    model.train()
    classification_head.train()
    
    return accuracy, avg_loss


def get_classification_head(classifier, embedding_dim, num_classes):
    classifiers = {
        'MCP': MarginCosineProduct(embedding_dim, num_classes),
        'AL': AngleLinear(embedding_dim, num_classes),
        'L': torch.nn.Linear(embedding_dim, num_classes, bias=False)
    }

    if classifier not in classifiers:
        raise ValueError(f"Unsupported classifier type: {classifier}")

    return classifiers[classifier]


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
    """Treina uma época e retorna loss e accuracy médios"""
    model.train()
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

        if params.distributed:
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
        f'Epoch [{epoch+1}/{params.epochs}] Summary: '
        f'Loss: {losses.avg:6.3f}, '
        f'Accuracy: {accuracy_meter.avg:4.2f}%, '
        f'Total Time: {batch_time.sum:4.3f}s'
    )
    LOGGER.info(log)
    
    # Retornar métricas da época
    return losses.avg, accuracy_meter.avg, batch_time.sum


def main(params):
    init_distributed_mode(params)

    setup_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    db_config = {
        'WebFace': {'num_classes': 10572},
        'VggFace2': {'num_classes': 8631},
        'MS1M': {'num_classes': 85742},
        'VggFaceHQ': {'num_classes': 9131}
    }
    
    if params.database not in db_config:
        raise ValueError("Unsupported database!")

    num_classes = db_config[params.database]['num_classes']

    LOGGER.info(f'Training on database: {params.database}')

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
    model.to(device)

    classification_head = get_classification_head(params.classifier, 512, num_classes)
    classification_head.to(device)

    if params.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[params.local_rank], output_device=params.local_rank
        )
        classification_head = torch.nn.parallel.DistributedDataParallel(
            classification_head, device_ids=[params.local_rank], output_device=params.local_rank
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    os.makedirs(params.save_path, exist_ok=True)
    metrics_save_path = 'metrics'
    os.makedirs(metrics_save_path, exist_ok=True)

    # Inicializar TrainingTracker
    experiment_name = f"{params.network}_{params.classifier}_{params.database}"
    tracker = TrainingTracker(
        save_dir=metrics_save_path,
        experiment_name=experiment_name
    )
    
    if params.use_retinaface_validation:
        tracker.set_face_validation_policy(params.no_face_policy)

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
            LOGGER.warning("Continuing without face validation")
            face_validator = None

    LOGGER.info('Loading training data.')
    
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
    
    # DataLoader para validação interna
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True
    )

    LOGGER.info(f'Length of training dataset: {len(train_loader.dataset)}, Number of Identities: {num_classes}')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': classification_head.parameters()}
    ],
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )

    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {params.lr_scheduler}")

    start_epoch = 0
    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location="cpu")

        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt['epoch']
        LOGGER.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch}')
        
        # Tentar carregar histórico do tracker
        tracker.load_history_from_checkpoint(params.checkpoint)

    best_accuracy = 0.0
    curr_accuracy = 0.0
    early_stopping = EarlyStopping(patience=10)

    LOGGER.info(f'Training started for {params.network}, Classifier: {params.classifier}')
    for epoch in range(start_epoch, params.epochs):
        if params.distributed:
            train_sampler.set_epoch(epoch)
        
        # Treinar época e obter métricas
        train_loss, train_accuracy, epoch_time = train_one_epoch(
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

        base_filename = f'{params.network}_{params.classifier}'

        last_save_path = os.path.join(params.save_path, f'{base_filename}_last.ckpt')

        checkpoint = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': params
        }

        save_on_master(checkpoint, last_save_path)

        if params.local_rank == 0:
            # Validação interna
            val_accuracy, val_loss = validate_model(
                model_without_ddp, 
                classification_head, 
                val_loader, 
                criterion, 
                device
            )
            
            LOGGER.info(f'\n{"="*70}')
            LOGGER.info(f'EPOCH {epoch+1} VALIDATION METRICS')
            LOGGER.info(f'{"="*70}')
            
            epoch_metrics_path = os.path.join(metrics_save_path, f'epoch_{epoch+1}')
            os.makedirs(epoch_metrics_path, exist_ok=True)
            
            # Avaliação externa (LFW/CelebA)
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
            
            # Registrar métricas no tracker
            current_lr = optimizer.param_groups[0]['lr']
            tracker.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,  # já está em %
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                external_metrics=metrics,
                learning_rate=current_lr,
                epoch_time=epoch_time
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
                LOGGER.info(f'  EER: {metrics["eer"]:.4f}')
            
            if 'far' in metrics and 'frr' in metrics:
                LOGGER.info(f'\nError Rates:')
                LOGGER.info(f'  FAR (False Accept Rate):  {metrics["far"]:.4f}')
                LOGGER.info(f'  FRR (False Reject Rate):  {metrics["frr"]:.4f}')
            
            LOGGER.info(f'{"="*70}\n')
            
            LOGGER.info(f'Internal validation - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}\n')

        if early_stopping(epoch, curr_accuracy):
            break

        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            save_on_master(checkpoint, os.path.join(params.save_path, f'{base_filename}_best.ckpt'))
            LOGGER.info(
                f"New best {params.val_dataset.upper()} similarity: {best_accuracy:.4f}. "
                f"Model saved to {params.save_path} with `_best` postfix.\n"
            )

    # Gerar relatório final com gráficos
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
        
        if face_validator is not None:
            report_path = os.path.join(final_metrics_path, 'face_validation_report.json')
            face_validator.save_validation_report(
                output_path=report_path,
                dataset_name=params.val_dataset
            )
            print_validation_summary(face_validator)
        
        LOGGER.info(f'\nFinal Validation Metrics:')
        
        if not final_metrics or len(final_metrics) <= 2:
            LOGGER.warning(f'  ⚠️  No valid pairs for final evaluation!')
        else:
            if 'mean_similarity' in final_metrics:
                LOGGER.info(f'  Mean Similarity: {final_metrics["mean_similarity"]:.4f} ± {final_metrics.get("std_similarity", 0.0):.4f}')
            
            if 'precision' in final_metrics:
                LOGGER.info(f'  Precision:       {final_metrics["precision"]:.4f}')
            if 'recall' in final_metrics:
                LOGGER.info(f'  Recall:          {final_metrics["recall"]:.4f}')
            if 'f1' in final_metrics:
                LOGGER.info(f'  F1-Score:        {final_metrics["f1"]:.4f}')
            if 'accuracy' in final_metrics:
                LOGGER.info(f'  Accuracy:        {final_metrics["accuracy"]:.4f}')
            
            if 'auc' in final_metrics:
                LOGGER.info(f'\nROC Analysis:')
                LOGGER.info(f'  AUC:             {final_metrics["auc"]:.4f}')
                if 'eer' in final_metrics:
                    LOGGER.info(f'  EER:             {final_metrics["eer"]:.4f} (threshold: {final_metrics.get("eer_threshold", 0.0):.4f})')
                
                for key in final_metrics:
                    if key.startswith('TAR@FAR'):
                        far_value = key.split('=')[1]
                        LOGGER.info(f'  TAR@FAR={far_value}: {final_metrics[key]:.4f}')
            
            if 'confusion_matrix' in final_metrics:
                LOGGER.info(f'\nConfusion Matrix:')
                LOGGER.info(f'  True Negatives:  {final_metrics.get("true_negatives", 0)}')
                LOGGER.info(f'  False Positives: {final_metrics.get("false_positives", 0)}')
                LOGGER.info(f'  False Negatives: {final_metrics.get("false_negatives", 0)}')
                LOGGER.info(f'  True Positives:  {final_metrics.get("true_positives", 0)}')
                
                if 'far' in final_metrics:
                    LOGGER.info(f'\n  FAR (False Accept Rate): {final_metrics["far"]:.4f}')
                if 'frr' in final_metrics:
                    LOGGER.info(f'  FRR (False Reject Rate): {final_metrics["frr"]:.4f}')
        
        # Gerar relatório final com gráficos (salva na mesma pasta de métricas)
        tracker.generate_final_report(metrics_save_path=final_metrics_path)
        
        LOGGER.info(f'\n{"="*70}')
        LOGGER.info(f'Metrics saved to: {final_metrics_path}')
        LOGGER.info(f'  - ROC Curve: {params.val_dataset}_roc_curve.png')
        LOGGER.info(f'  - Confusion Matrix: {params.val_dataset}_confusion_matrix.png')
        LOGGER.info(f'  - Accuracy/Loss Curves: accuracy_loss_curves.png')
        LOGGER.info(f'  - Training Curves: training_curves.png')
        LOGGER.info(f'  - Logs: {tracker.logs_dir}')
        LOGGER.info(f'{"="*70}\n')

    LOGGER.info('Training completed.')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)