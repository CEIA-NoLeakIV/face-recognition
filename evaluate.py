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
        dict: Dicionário com precision, recall, f1, accuracy
    """
    if len(predictions) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
    
    # Extrair ground truth e predições
    y_true = predictions[:, 3].astype(int)  # Ground truth (0 ou 1)
    similarities = predictions[:, 2].astype(float)
    y_pred = (similarities > threshold).astype(int)  # Predições baseadas no threshold
    
    # Calcular métricas
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def compute_roc_metrics(predictions, save_path=None):
    """
    Calcula métricas ROC, TAR@FAR, e gera gráficos.
    
    Args:
        predictions: Array com formato [path1, path2, similarity, ground_truth]
        save_path: Caminho para salvar os gráficos (opcional)
        
    Returns:
        dict: Dicionário com métricas ROC completas
    """
    if len(predictions) == 0:
        return None
    
    # Extrair ground truth e similaridades
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    # Calcular ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Calcular EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    # Calcular TAR@FAR específicos
    far_targets = [0.001, 0.01, 0.1]  # 0.1%, 1%, 10%
    tar_at_far = {}
    
    for far_target in far_targets:
        idx = np.argmin(np.abs(fpr - far_target))
        tar_at_far[f'TAR@FAR={far_target}'] = tpr[idx]
    
    # Gerar gráfico ROC se save_path fornecido
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.scatter([fpr[eer_threshold_idx]], [tpr[eer_threshold_idx]], 
                   color='red', s=100, label=f'EER = {eer:.4f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (TAR)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        **tar_at_far
    }


def compute_confusion_matrix(predictions, threshold=0.35, save_path=None):
    """
    Calcula e visualiza matriz de confusão.
    
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
         compute_full_metrics=False, save_metrics_path=None, threshold=0.35):
    """
    Evaluate the model on validation dataset (LFW or CelebA).
    
    Args:
        model: The model to evaluate
        model_path: Path to model weights (optional)
        device: Device to run evaluation on
        val_dataset: Dataset to use for validation ('lfw' or 'celeba')
        val_root: Root directory of validation data
        compute_full_metrics: Se True, calcula métricas completas (ROC, confusion matrix, etc)
        save_metrics_path: Diretório para salvar gráficos de métricas
        threshold: Limiar de similaridade para métricas de classificação
        
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

    predicts = []
    with torch.no_grad():
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

    predicts = np.array(predicts)
    similarities = predicts[:, 2].astype(float)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    # Calcular métricas básicas
    basic_metrics = compute_metrics_from_predictions(predicts, threshold=threshold)
    
    dataset_name = val_dataset.upper()
    print(f'{dataset_name} Evaluation Results:')
    print(f'Mean Similarity: {mean_similarity:.4f} | Std: {std_similarity:.4f}')
    print(f'Precision: {basic_metrics["precision"]:.4f} | Recall: {basic_metrics["recall"]:.4f}')
    print(f'F1-Score: {basic_metrics["f1"]:.4f} | Accuracy: {basic_metrics["accuracy"]:.4f}')
    
    metrics_dict = {
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        **basic_metrics
    }
    
    # Calcular métricas completas se solicitado
    if compute_full_metrics and save_metrics_path:
        # ROC e métricas relacionadas
        roc_path = os.path.join(save_metrics_path, f'{val_dataset}_roc_curve.png')
        roc_metrics = compute_roc_metrics(predicts, save_path=roc_path)
        
        if roc_metrics:
            print(f'\nROC Metrics:')
            print(f'AUC: {roc_metrics["auc"]:.4f}')
            print(f'EER: {roc_metrics["eer"]:.4f} (threshold: {roc_metrics["eer_threshold"]:.4f})')
            for key, value in roc_metrics.items():
                if key.startswith('TAR@FAR'):
                    print(f'{key}: {value:.4f}')
            
            metrics_dict.update(roc_metrics)
        
        # Matriz de confusão
        cm_path = os.path.join(save_metrics_path, f'{val_dataset}_confusion_matrix.png')
        cm = compute_confusion_matrix(predicts, threshold=threshold, save_path=cm_path)
        
        if cm is not None:
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f'\nConfusion Matrix:')
            print(f'True Negatives: {tn}, False Positives: {fp}')
            print(f'False Negatives: {fn}, True Positives: {tp}')
            print(f'FAR (False Accept Rate): {far:.4f}')
            print(f'FRR (False Reject Rate): {frr:.4f}')
            
            metrics_dict.update({
                'confusion_matrix': cm,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'far': far,
                'frr': frr
            })
    
    return mean_similarity, predicts, metrics_dict


if __name__ == '__main__':
    _, result, _ = eval(sphere20(512).to('cuda'), model_path='weights/sphere20_mcp.pth')
    _, result, _ = eval(sphere36(512).to('cuda'), model_path='weights/sphere36_mcp.pth')
    _, result, _ = eval(MobileNetV1(512).to('cuda'), model_path='weights/mobilenetv1_mcp.pth')
    _, result, _ = eval(MobileNetV2(512).to('cuda'), model_path='weights/mobilenetv2_mcp.pth')
    _, result, _ = eval(mobilenet_v3_small(512).to('cuda'), model_path='weights/mobilenetv3_small_mcp.pth')
    _, result, _ = eval(mobilenet_v3_large(512).to('cuda'), model_path='weights/mobilenetv3_large_mcp.pth')