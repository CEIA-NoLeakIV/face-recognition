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

# Import face validation (optional, will warn if not available)
try:
    from utils.face_validation import FaceValidator, print_validation_summary
    FACE_VALIDATION_AVAILABLE = True
except ImportError:
    FACE_VALIDATION_AVAILABLE = False
    LOGGER.warning("Face validation module not available. RetinaFace validation will be disabled.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Command-line arguments for training a face recognition model"))

    # Dataset and Paths
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

    # Model Settings
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

    # Training Hyperparameters
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training. Default: 512.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training. Default: 30.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate. Default: 0.1.')
    
    # lr_scheduler configuration
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

    # Validation Settings
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

    # 🆕 FACE VALIDATION SETTINGS (RetinaFace)
    parser.add_argument(
        '--use-retinaface-validation',
        action='store_true',
        help='Enable face validation using RetinaFace during validation. Images without detected faces will be handled according to --no-face-policy.'
    )
    parser.add_argument(
        '--no-face-policy',
        type=str,
        default='exclude',
        choices=['exclude', 'include'],
        help='Policy for handling images without detected faces. "exclude": skip them during validation (default). "include": use them anyway.'
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
        help='Directory to store face validation cache files. Default: face_validation_cache.'
    )

    parser.add_argument("--world-size", default=1, type=int, help="Number of distributed processes")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only."
    )

    return parser.parse_args()


def validate_model(model, classification_head, val_loader, device):
    """Validates the model on validation subset"""
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
                outputs = classification_head(embeddings, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
    
    accuracy = total_correct / total_samples
    
    # Return to training mode
    model.train()
    classification_head.train()
    
    return accuracy


# Define a function to select a classification head
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
    criterion, optimizer,
    data_loader,
    device,
    epoch,
    params
) -> None:
    model.train()
    losses = AverageMeter("Avg Loss", ":6.3f")
    batch_time = AverageMeter("Batch Time", ":4.3f")
    accuracy_meter = AverageMeter("Accuracy", ":4.2f")
    last_batch_idx = len(data_loader) - 1

    start_time = time.time()
    for batch_idx, (images, target) in enumerate(data_loader):
        last_batch = last_batch_idx == batch_idx

        # Move data to device
        images = images.to(device)
        target = target.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(images)
        if isinstance(classification_head, torch.nn.Linear):
            output = classification_head(embeddings)
        else:
            output = classification_head(embeddings, target)

        # Compute loss and accuracy
        loss = criterion(output, target)

        # calculate_accuracy is a function to compute classification accuracy.
        accuracy = calculate_accuracy(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update meters
        if params.distributed:
            reduced_loss = reduce_tensor(loss.data, params.world_size)
            losses.update(reduced_loss.item(), images.size(0))
        else:
            losses.update(loss.item(), images.size(0))

        accuracy_meter.update(accuracy.item(), images.size(0))

        # Update time
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        # Print progress
        if batch_idx % params.print_freq == 0 or last_batch:
            if params.local_rank == 0:
                LOGGER.info(
                    f"Epoch [{epoch+1}/{params.epochs}][{batch_idx}/{last_batch_idx}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc: {accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f}) "
                    f"Time: {batch_time.val:.3f}s"
                )


def main(args):
    params = args

    # Setup for distributed training
    init_distributed_mode(params)
    device = torch.device(params.device if hasattr(params, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Seed for reproducibility
    setup_seed(42)

    # Database configuration
    db_config = {
        'WebFace': {'num_classes': 10572},
        'VggFace2': {'num_classes': 8631},
        'MS1M': {'num_classes': 85742},
        'VggFaceHQ': {'num_classes': 9131}
    }

    LOGGER.info(f'Training on database: {params.database}')
    num_classes = db_config[params.database]['num_classes']

    # Model selection based on arguments
    if params.network == 'sphere20':
        model = sphere20(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere36':
        model = sphere36(embedding_dim=512, in_channels=3)
    elif params.network == 'sphere64':
        model = sphere64(embedding_dim=512, in_channels=3)
    elif params.network == "mobilenetv1":
        model = MobileNetV1(embedding_dim=512)
    elif params.network == "mobilenetv2":
        model = MobileNetV2(embedding_dim=512)
    elif params.network == "mobilenetv3_small":
        model = mobilenet_v3_small(embedding_dim=512)
    elif params.network == "mobilenetv3_large":
        model = mobilenet_v3_large(embedding_dim=512)
    else:
        raise ValueError("Unsupported network!")

    # No need for DataParallel, we are using a single GPU
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    # Create save path if it does not exist
    os.makedirs(params.save_path, exist_ok=True)
    
    # Create metrics save path
    metrics_save_path = os.path.join(params.save_path, 'metrics')
    os.makedirs(metrics_save_path, exist_ok=True)

    # Select classification head
    classification_head = get_classification_head(params.classifier, embedding_dim=512, num_classes=num_classes)
    classification_head = classification_head.to(device)

    # Transformations for images
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # DataLoader with validation split
    LOGGER.info('Loading training data.')
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

    LOGGER.info(f'Length of training dataset: {len(train_loader.dataset)}, Number of Identities: {num_classes}')

    # 🆕 INITIALIZE FACE VALIDATOR (if enabled)
    face_validator = None
    if params.use_retinaface_validation:
        if not FACE_VALIDATION_AVAILABLE:
            LOGGER.error("❌ RetinaFace validation requested but face_validation module is not available!")
            LOGGER.error("Please ensure utils/face_validation.py exists and uniface is installed.")
            raise ImportError("Face validation module required but not available")
        
        LOGGER.info("\n" + "="*70)
        LOGGER.info("🔍 RETINAFACE FACE VALIDATION ENABLED")
        LOGGER.info("="*70)
        LOGGER.info(f"Confidence threshold: {params.retinaface_conf_threshold}")
        LOGGER.info(f"No-face policy: {params.no_face_policy}")
        LOGGER.info(f"Cache directory: {params.face_validation_cache_dir}")
        
        try:
            face_validator = FaceValidator(
                gpu_id=0 if torch.cuda.is_available() else -1,
                conf_threshold=params.retinaface_conf_threshold,
                cache_dir=params.face_validation_cache_dir
            )
            LOGGER.info("✅ FaceValidator initialized successfully\n")
        except Exception as e:
            LOGGER.error(f"❌ Failed to initialize FaceValidator: {e}")
            raise

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': classification_head.parameters()}
    ],
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay
    )
    
    # Learning rate scheduler
    if params.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    elif params.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {params.lr_scheduler}")

    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Resume training if checkpoint is provided
    start_epoch = 0
    best_accuracy = 0.0

    if params.checkpoint and os.path.isfile(params.checkpoint):
        checkpoint = torch.load(params.checkpoint, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        LOGGER.info(f'Resuming training from epoch {start_epoch}')
    else:
        LOGGER.info('Starting training from scratch')

    # Training loop
    base_filename = f'{params.network}_{params.classifier}'
    
    for epoch in range(start_epoch, params.epochs):
        if params.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_one_epoch(
            model, classification_head, criterion, optimizer,
            train_loader, device, epoch, params
        )

        # Step the learning rate scheduler
        lr_scheduler.step()

        # Prepare checkpoint
        last_save_path = os.path.join(params.save_path, f'{base_filename}_last.ckpt')

        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': params
        }

        save_on_master(checkpoint, last_save_path)

        if params.local_rank == 0:
            # Validation Evaluation with metrics
            LOGGER.info(f'\n{"="*70}')
            LOGGER.info(f'EPOCH {epoch+1} VALIDATION METRICS')
            LOGGER.info(f'{"="*70}')
            
            # Salvar métricas por época
            epoch_metrics_path = os.path.join(metrics_save_path, f'epoch_{epoch+1:03d}')
            os.makedirs(epoch_metrics_path, exist_ok=True)
            
            # 🆕 Pass face_validator to evaluate.eval
            curr_accuracy, _, metrics = evaluate.eval(
                model_without_ddp, 
                device=device,
                val_dataset=params.val_dataset,
                val_root=params.val_root,
                compute_full_metrics=True,
                save_metrics_path=epoch_metrics_path,
                threshold=params.val_threshold,
                face_validator=face_validator,  # 🆕 NEW PARAMETER
                no_face_policy=params.no_face_policy  # 🆕 NEW PARAMETER
            )
            
            # Log das métricas principais
            LOGGER.info(f'\nValidation Metrics (Threshold={params.val_threshold}):')
            LOGGER.info(f'  Precision: {metrics["precision"]:.4f}')
            LOGGER.info(f'  Recall:    {metrics["recall"]:.4f}')
            LOGGER.info(f'  F1-Score:  {metrics["f1"]:.4f}')
            LOGGER.info(f'  Accuracy:  {metrics["accuracy"]:.4f}')
            
            if 'auc' in metrics:
                LOGGER.info(f'\nROC Metrics:')
                LOGGER.info(f'  AUC: {metrics["auc"]:.4f}')
                LOGGER.info(f'  EER: {metrics["eer"]:.4f}')
            
            if 'far' in metrics and 'frr' in metrics:
                LOGGER.info(f'\nError Rates:')
                LOGGER.info(f'  FAR (False Accept Rate):  {metrics["far"]:.4f}')
                LOGGER.info(f'  FRR (False Reject Rate):  {metrics["frr"]:.4f}')
            
            # 🆕 Log face validation statistics if available
            if 'face_validation_stats' in metrics:
                face_stats = metrics['face_validation_stats']
                LOGGER.info(f'\n{"="*70}')
                LOGGER.info('FACE DETECTION STATISTICS (RetinaFace)')
                LOGGER.info(f'{"="*70}')
                LOGGER.info(f'  Policy: {params.no_face_policy.upper()}')
                LOGGER.info(f'  Total pairs checked:     {face_stats.get("total_pairs", 0)}')
                LOGGER.info(f'  Valid pairs used:        {face_stats.get("valid_pairs", 0)}')
                LOGGER.info(f'  Excluded pairs:          {face_stats.get("excluded_pairs", 0)}')
                if face_stats.get('total_pairs', 0) > 0:
                    exclusion_rate = face_stats.get('excluded_pairs', 0) / face_stats.get('total_pairs', 1) * 100
                    LOGGER.info(f'  Exclusion rate:          {exclusion_rate:.2f}%')
                
                if face_stats.get('excluded_pairs', 0) > 0:
                    LOGGER.warning(f'  ⚠️  {face_stats["excluded_pairs"]} pairs excluded due to missing face detection')
            
            LOGGER.info(f'{"="*70}\n')
            
            # Internal validation (for monitoring only)
            val_loader = DataLoader(
                val_dataset,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True
            )
            
            val_accuracy = validate_model(model_without_ddp, classification_head, val_loader, device)
            LOGGER.info(f'Internal validation accuracy ({params.database} subset): {val_accuracy:.4f}\n')

        if early_stopping(epoch, curr_accuracy):
            break

        # Save the best model if validation similarity improves
        if curr_accuracy > best_accuracy:
            best_accuracy = curr_accuracy
            save_on_master(checkpoint, os.path.join(params.save_path, f'{base_filename}_best.ckpt'))
            LOGGER.info(
                f"New best {params.val_dataset.upper()} similarity: {best_accuracy:.4f}. "
                f"Model saved to {params.save_path} with `_best` postfix.\n"
            )

    # Final comprehensive evaluation after training
    if params.local_rank == 0:
        LOGGER.info(f'\n{"="*70}')
        LOGGER.info('FINAL COMPREHENSIVE EVALUATION')
        LOGGER.info(f'{"="*70}')
        
        # Criar diretório para métricas finais
        final_metrics_path = os.path.join(metrics_save_path, 'final_evaluation')
        os.makedirs(final_metrics_path, exist_ok=True)
        
        # 🆕 Pass face_validator to final evaluation
        _, _, final_metrics = evaluate.eval(
            model_without_ddp,
            device=device,
            val_dataset=params.val_dataset,
            val_root=params.val_root,
            compute_full_metrics=True,
            save_metrics_path=final_metrics_path,
            threshold=params.val_threshold,
            face_validator=face_validator,  # 🆕 NEW PARAMETER
            no_face_policy=params.no_face_policy  # 🆕 NEW PARAMETER
        )
        
        LOGGER.info(f'\nFinal Validation Metrics:')
        LOGGER.info(f'  Mean Similarity: {final_metrics["mean_similarity"]:.4f} ± {final_metrics["std_similarity"]:.4f}')
        LOGGER.info(f'  Precision:       {final_metrics["precision"]:.4f}')
        LOGGER.info(f'  Recall:          {final_metrics["recall"]:.4f}')
        LOGGER.info(f'  F1-Score:        {final_metrics["f1"]:.4f}')
        LOGGER.info(f'  Accuracy:        {final_metrics["accuracy"]:.4f}')
        
        if 'auc' in final_metrics:
            LOGGER.info(f'\nROC Analysis:')
            LOGGER.info(f'  AUC:             {final_metrics["auc"]:.4f}')
            LOGGER.info(f'  EER:             {final_metrics["eer"]:.4f} (threshold: {final_metrics["eer_threshold"]:.4f})')
            
            # TAR@FAR metrics
            for key in final_metrics:
                if key.startswith('TAR@FAR'):
                    far_value = key.split('=')[1]
                    LOGGER.info(f'  TAR@FAR={far_value}: {final_metrics[key]:.4f}')
        
        if 'confusion_matrix' in final_metrics:
            LOGGER.info(f'\nConfusion Matrix:')
            LOGGER.info(f'  True Negatives:  {final_metrics["true_negatives"]}')
            LOGGER.info(f'  False Positives: {final_metrics["false_positives"]}')
            LOGGER.info(f'  False Negatives: {final_metrics["false_negatives"]}')
            LOGGER.info(f'  True Positives:  {final_metrics["true_positives"]}')
            LOGGER.info(f'\n  FAR (False Accept Rate): {final_metrics["far"]:.4f}')
            LOGGER.info(f'  FRR (False Reject Rate): {final_metrics["frr"]:.4f}')
        
        # 🆕 Save face validation report to final_report if validator was used
        if face_validator is not None and 'face_validation_stats' in final_metrics:
            face_report_path = os.path.join(final_metrics_path, 'face_validation_report.json')
            face_validator.save_validation_report(
                output_path=face_report_path,
                dataset_name=f"{params.val_dataset.upper()}_final"
            )
            LOGGER.info(f'\n📊 Face validation report saved to: {face_report_path}')
            
            # Print final summary
            print_validation_summary(face_validator)
        
        LOGGER.info(f'\n{"="*70}')
        LOGGER.info(f'Metrics saved to: {final_metrics_path}')
        LOGGER.info(f'  - ROC Curve: {params.val_dataset}_roc_curve.png')
        LOGGER.info(f'  - Confusion Matrix: {params.val_dataset}_confusion_matrix.png')
        if face_validator is not None:
            LOGGER.info(f'  - Face Validation Report: face_validation_report.json')
        LOGGER.info(f'{"="*70}\n')

    LOGGER.info('Training completed.')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)