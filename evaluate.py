import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)

# Import face validation (optional)
try:
    from utils.face_validation import FaceValidator, validate_lfw_pairs
    FACE_VALIDATION_AVAILABLE = True
except ImportError:
    FACE_VALIDATION_AVAILABLE = False


def extract_deep_features(model, image, device):
    """
    Extracts deep features for an image using the model, including both the original and flipped versions.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model used for feature extraction.
        image (PIL.Image): The input image to extract features from.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: Combined feature vector of original and flipped images.
    """

    # Define transforms
    original_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    flipped_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Apply transforms
    original_image_tensor = original_transform(image).unsqueeze(0).to(device)
    flipped_image_tensor = flipped_transform(image).unsqueeze(0).to(device)

    # Extract features
    original_features = model(original_image_tensor)
    flipped_features = model(flipped_image_tensor)

    # Combine and return features
    combined_features = torch.cat([original_features, flipped_features], dim=1).squeeze()
    return combined_features


def compute_metrics_from_predictions(predictions, threshold=0.35):
    """
    Calcula métricas de classificação a partir das predições.
    
    Args:
        predictions: Array com formato [path1, path2, similarity, ground_truth]
        threshold: Limiar de similaridade para classificação
        
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    if len(predictions) == 0:
        return {}
    
    # Extrair ground truth e similaridades
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    # Predições binárias baseadas no threshold
    y_pred = (similarities > threshold).astype(int)
    
    # Calcular métricas básicas
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calcular taxas de erro
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'far': far,
        'frr': frr
    }
    
    # Calcular ROC e AUC se houver ambas as classes
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Calcular EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_threshold_idx]
        eer_threshold = thresholds[eer_threshold_idx]
        
        # Calcular TAR@FAR
        tar_at_far = {}
        for far_value in [0.001, 0.01, 0.1]:
            idx = np.where(fpr <= far_value)[0]
            if len(idx) > 0:
                tar_at_far[f'TAR@FAR={far_value}'] = tpr[idx[-1]]
        
        metrics.update({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            **tar_at_far
        })
    
    return metrics


def plot_roc_curve(predictions, save_path=None):
    """
    Plota e salva a curva ROC.
    
    Args:
        predictions: Array com formato [path1, path2, similarity, ground_truth]
        save_path: Caminho para salvar o gráfico
        
    Returns:
        tuple: (fpr, tpr, auc_score)
    """
    if len(predictions) == 0:
        return None, None, None
    
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    # Verificar se há ambas as classes
    if len(np.unique(y_true)) < 2:
        print("Warning: Cannot compute ROC curve - only one class present in ground truth")
        return None, None, None
    
    fpr, tpr, _ = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fpr, tpr, roc_auc


def plot_confusion_matrix(predictions, threshold, save_path=None):
    """
    Plota e salva a matriz de confusão.
    
    Args:
        predictions: Array com formato [path1, path2, similarity, ground_truth]
        threshold: Limiar de similaridade
        save_path: Caminho para salvar o gráfico (opcional)
        
    Returns:
        np.ndarray: Matriz de confusão
    """
    if len(predictions) == 0:
        return None
    
    # Extrair ground truth e predições
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    y_pred = (similarities > threshold).astype(int)
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Gerar visualização se save_path fornecido
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        classes = ['Different', 'Same']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        
        # Adicionar valores na matriz
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return cm


def eval(model, model_path=None, device=None, val_dataset='lfw', val_root='data/lfw/val', 
         compute_full_metrics=False, save_metrics_path=None, threshold=0.35,
         face_validator=None, no_face_policy='exclude'):
    """
    Evaluate the model on validation dataset (LFW or CelebA)
    
    Args:
        model: The model to evaluate
        model_path: Path to model weights (optional)
        device: Device to run evaluation on
        val_dataset: Dataset to use for validation ('lfw' or 'celeba')
        val_root: Root directory of validation data
        compute_full_metrics: Se True, calcula métricas completas (ROC, confusion matrix, etc)
        save_metrics_path: Diretório para salvar gráficos de métricas
        threshold: Limiar de similaridade para métricas de classificação
        face_validator: 🆕 FaceValidator instance for RetinaFace validation (optional)
        no_face_policy: 🆕 Policy for images without faces ('exclude' or 'include')
        
    Returns:
        tuple: (mean_similarity, predictions, metrics_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()

    root = val_root
    
    # Select annotation file and image path logic based on dataset
    if val_dataset == 'lfw':
        ann_file = os.path.join(root, 'lfw_ann.txt')
        try:
            with open(ann_file) as f:
                lines = f.readlines()
                # Primeira linha contém o número de pares, pular ela
                pair_lines = lines[1:]
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'lfw_ann.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
    elif val_dataset == 'celeba':
        ann_file = os.path.join(root, 'celeba_pairs.txt')
        try:
            with open(ann_file) as f:
                pair_lines = f.readlines()[1:]  # Skip header if exists
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'celeba_pairs.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
    else:
        raise ValueError(f"Unsupported validation dataset: {val_dataset}. Choose 'lfw' or 'celeba'.")

    # 🆕 FACE VALIDATION: Validate and filter pairs if face_validator is provided
    valid_pairs_list = None
    face_validation_stats = {}
    
    if face_validator is not None and FACE_VALIDATION_AVAILABLE:
        print(f"\n{'='*70}")
        print("🔍 VALIDATING FACES WITH RETINAFACE")
        print(f"{'='*70}")
        
        if val_dataset == 'lfw':
            valid_pairs_tuples, excluded_pairs, pair_stats = validate_lfw_pairs(
                face_validator,
                lfw_root=root,
                ann_file=ann_file,
                policy=no_face_policy
            )
            
            # Convert tuples to pair_lines format
            valid_pairs_list = []
            for path1, path2, is_same in valid_pairs_tuples:
                valid_pairs_list.append((path1, path2, is_same))
            
            # Store statistics
            face_validation_stats = pair_stats
            
            print(f"\nFace Validation Results:")
            print(f"  Total pairs:        {pair_stats['total_pairs']}")
            print(f"  Valid pairs:        {pair_stats['valid_pairs']}")
            print(f"  Excluded pairs:     {pair_stats['excluded_pairs']} ({pair_stats['exclusion_rate']:.2f}%)")
            print(f"  Policy:             {no_face_policy.upper()}")
            
            if pair_stats['excluded_pairs'] > 0:
                print(f"\n⚠️  WARNING: {pair_stats['excluded_pairs']} pairs were excluded due to missing face detection")
            
            print(f"{'='*70}\n")
        
        elif val_dataset == 'celeba':
            # Similar implementation for CelebA (not fully implemented here)
            print("Warning: Face validation for CelebA not fully implemented yet")
            valid_pairs_list = None

    # Process pairs
    predicts = []
    pairs_to_process = []
    
    # 🆕 If face validation was performed and pairs were filtered
    if valid_pairs_list is not None:
        # Use filtered pairs
        for path1, path2, is_same in valid_pairs_list:
            pairs_to_process.append((path1, path2, is_same))
    else:
        # Use all pairs from annotation file (original behavior)
        for line in pair_lines:
            parts = line.strip().split()

            if val_dataset == 'lfw':
                if len(parts) == 3:
                    person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                    
                    # Format filename as: "Person_Name_0001.jpg"
                    filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                    filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                    
                    # Build full path
                    path1 = os.path.join(root, person_name, filename1)
                    path2 = os.path.join(root, person_name, filename2)
                    is_same = '1'
                elif len(parts) == 4:
                    # Pares negativos: pessoa1 img1 pessoa2 img2
                    person1, img_num1, person2, img_num2 = parts
                    
                    filename1 = f'{person1}_{int(img_num1):04d}.jpg'
                    filename2 = f'{person2}_{int(img_num2):04d}.jpg'
                    
                    path1 = os.path.join(root, person1, filename1)
                    path2 = os.path.join(root, person2, filename2)
                    is_same = '0'
                else:
                    # Skip lines that don't have 3 or 4 columns
                    continue
                    
            elif val_dataset == 'celeba':
                if len(parts) == 2:
                    # Format: img1.jpg img2.jpg (both same identity - positive pairs only)
                    filename1, filename2 = parts[0], parts[1]
                    
                    # CelebA images are in img_align_celeba/img_align_celeba folder
                    path1 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', filename1)
                    path2 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', filename2)
                    is_same = '1'
                else:
                    continue
            
            pairs_to_process.append((path1, path2, is_same))
    
    # Extract features for all pairs
    with torch.no_grad():
        for path1, path2, is_same in pairs_to_process:
            try:
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: Image not found, skipping pair: {path1} or {path2}")
                continue

            f1 = extract_deep_features(model, img1, device)
            f2 = extract_deep_features(model, img2, device)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, distance.item(), is_same])
    
    if len(predicts) == 0:
        print("Warning: No valid pairs were processed in the evaluation.")
        return 0.0, np.array([]), {}

    # Convert to numpy array
    predicts = np.array(predicts)
    
    # Calculate mean similarity
    similarities = predicts[:, 2].astype(float)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    # Initialize metrics dictionary
    metrics = {
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'median_similarity': np.median(similarities),
    }
    
    # 🆕 Add face validation statistics to metrics if available
    if face_validation_stats:
        metrics['face_validation_stats'] = face_validation_stats
    
    # Compute full metrics if requested
    if compute_full_metrics:
        classification_metrics = compute_metrics_from_predictions(predicts, threshold)
        metrics.update(classification_metrics)
        
        # Save visualizations if path provided
        if save_metrics_path:
            os.makedirs(save_metrics_path, exist_ok=True)
            
            # ROC Curve
            roc_path = os.path.join(save_metrics_path, f'{val_dataset}_roc_curve.png')
            plot_roc_curve(predicts, save_path=roc_path)
            
            # Confusion Matrix
            cm_path = os.path.join(save_metrics_path, f'{val_dataset}_confusion_matrix.png')
            plot_confusion_matrix(predicts, threshold, save_path=cm_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"{val_dataset.upper()} - Simplified Evaluation (Positive Pairs Only):")
    print(f"Mean Similarity: {mean_similarity:.4f} | Standard Deviation: {std_similarity:.4f}")
    
    if compute_full_metrics and 'accuracy' in metrics:
        print(f"\nAdditional Metrics (Threshold: {threshold:.4f}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        if 'auc' in metrics:
            print(f"  AUC Score: {metrics['auc']:.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"  TN: {metrics['true_negatives']:5d}  FP: {metrics['false_positives']:5d}")
            print(f"  FN: {metrics['false_negatives']:5d}  TP: {metrics['true_positives']:5d}")
    
    print(f"{'='*50}\n")
    
    return mean_similarity, predicts, metrics


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example: Load model and evaluate
    model = mobilenet_v3_large(embedding_dim=512)
    model_path = "weights/mobilenetv3_large_MCP_best.ckpt"
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
    
    # Evaluate on LFW
    mean_sim, preds, metrics = eval(
        model,
        device=device,
        val_dataset='lfw',
        val_root='data/lfw/val',
        compute_full_metrics=True,
        save_metrics_path='evaluation_metrics',
        threshold=0.35
    )
    
    print(f"Mean Similarity: {mean_sim:.4f}")
    print(f"Total pairs evaluated: {len(preds)}")