import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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


def k_fold_split(n=6000, n_folds=10):
    """
    Creates k-fold splits for cross-validation.
    
    Args:
        n (int): Total number of samples
        n_folds (int): Number of folds
        
    Returns:
        list: List of [train_indices, test_indices] for each fold
    """
    folds = []
    base = list(range(n))
    fold_size = n // n_folds

    for idx in range(n_folds):
        test = base[idx * fold_size:(idx + 1) * fold_size]
        train = base[:idx * fold_size] + base[(idx + 1) * fold_size:]
        folds.append([train, test])

    return folds


def eval_accuracy(predictions, threshold):
    """
    Calculates accuracy for binary classification.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        threshold (float): Threshold for classification
        
    Returns:
        float: Accuracy score
    """
    y_true = []
    y_pred = []

    for _, _, distance, gt in predictions:
        y_true.append(int(gt))
        pred = 1 if float(distance) > threshold else 0
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    return accuracy


def find_best_threshold(predictions, thresholds):
    """
    Finds the best threshold that maximizes accuracy.
    
    Args:
        predictions (np.ndarray): Predictions array
        thresholds (list): List of thresholds to test
        
    Returns:
        float: Best threshold
    """
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        accuracy = eval_accuracy(predictions, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def compute_roc_curve(predictions, save_path=None):
    """
    Computes ROC curve and AUC score.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        save_path (str, optional): Path to save ROC curve plot
        
    Returns:
        dict: Dictionary containing fpr, tpr, auc_score, and roc_thresholds
    """
    # Extract ground truth and similarity scores
    y_true = predictions[:, 3].astype(int)
    y_scores = predictions[:, 2].astype(float)
    
    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    # Plot ROC curve if save_path is provided
    if save_path:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to: {save_path}")
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc_score': auc_score,
        'roc_thresholds': roc_thresholds
    }


def compute_f1_score(predictions, threshold):
    """
    Computes F1 score.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        threshold (float): Classification threshold
        
    Returns:
        float: F1 score
    """
    y_true = predictions[:, 3].astype(int)
    y_scores = predictions[:, 2].astype(float)
    y_pred = (y_scores > threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    return f1


def compute_precision(predictions, threshold):
    """
    Computes precision score.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        threshold (float): Classification threshold
        
    Returns:
        float: Precision score
    """
    y_true = predictions[:, 3].astype(int)
    y_scores = predictions[:, 2].astype(float)
    y_pred = (y_scores > threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    return precision


def compute_recall(predictions, threshold):
    """
    Computes recall score.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        threshold (float): Classification threshold
        
    Returns:
        float: Recall score
    """
    y_true = predictions[:, 3].astype(int)
    y_scores = predictions[:, 2].astype(float)
    y_pred = (y_scores > threshold).astype(int)
    
    recall = recall_score(y_true, y_pred, zero_division=0)
    return recall


def compute_confusion_matrix(predictions, threshold, save_path=None):
    """
    Computes and plots confusion matrix.
    
    Args:
        predictions (np.ndarray): Array with columns [path1, path2, distance, gt]
        threshold (float): Classification threshold
        save_path (str, optional): Path to save confusion matrix plot
        
    Returns:
        np.ndarray: Confusion matrix
    """
    y_true = predictions[:, 3].astype(int)
    y_scores = predictions[:, 2].astype(float)
    y_pred = (y_scores > threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix if save_path is provided
    if save_path:
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Different', 'Same']
        )
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    return cm


def eval(model, model_path=None, device=None, val_dataset='lfw', val_root='data/lfw/val', 
         compute_metrics=True, save_plots=False, plots_dir='weights'):
    """
    Evaluate the model on validation dataset (LFW or CelebA) with comprehensive metrics.
    
    Args:
        model: The model to evaluate
        model_path: Path to model weights (optional)
        device: Device to run evaluation on
        val_dataset: Dataset to use for validation ('lfw' or 'celeba')
        val_root: Root directory of validation data
        compute_metrics: Whether to compute additional metrics (ROC, F1, etc.)
        save_plots: Whether to save plots (ROC curve, confusion matrix)
        plots_dir: Directory to save plots
        
    Returns:
        tuple: (accuracy_proxy, predictions, metrics_dict)
            - accuracy_proxy: Mean similarity score
            - predictions: Array of predictions
            - metrics_dict: Dictionary with all computed metrics
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
                pair_lines = f.readlines()[1:]
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
                else:
                    # Skip lines that don't have 3 columns
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

    dataset_name = val_dataset.upper()
    print(f'{dataset_name} - Simplified Evaluation (Positive Pairs Only):')
    print(f'Mean Similarity: {mean_similarity:.4f} | Standard Deviation: {std_similarity:.4f}')

    # Initialize metrics dictionary
    metrics_dict = {
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity
    }
    
    # Compute additional metrics if requested
    if compute_metrics and len(predicts) > 0:
        # Find best threshold
        thresholds = np.linspace(0.2, 0.8, 100)
        best_threshold = find_best_threshold(predicts, thresholds)
        
        # Compute all metrics
        accuracy = eval_accuracy(predicts, best_threshold)
        f1 = compute_f1_score(predicts, best_threshold)
        precision = compute_precision(predicts, best_threshold)
        recall = compute_recall(predicts, best_threshold)
        
        # Add to metrics dictionary
        metrics_dict.update({
            'best_threshold': best_threshold,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })
        
        # Compute ROC curve
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            roc_save_path = os.path.join(plots_dir, f'{val_dataset}_roc_curve.png')
            cm_save_path = os.path.join(plots_dir, f'{val_dataset}_confusion_matrix.png')
        else:
            roc_save_path = None
            cm_save_path = None
            
        roc_results = compute_roc_curve(predicts, save_path=roc_save_path)
        cm = compute_confusion_matrix(predicts, best_threshold, save_path=cm_save_path)
        
        # Add ROC results to metrics
        metrics_dict.update({
            'auc_score': roc_results['auc_score'],
            'confusion_matrix': cm
        })
        
        # Print metrics
        print(f'\nAdditional Metrics (Threshold: {best_threshold:.4f}):')
        print(f'  Accuracy:  {accuracy:.4f}')
        print(f'  F1 Score:  {f1:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall:    {recall:.4f}')
        print(f'  AUC Score: {roc_results["auc_score"]:.4f}')
        print(f'\nConfusion Matrix:')
        print(f'  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}')
        print(f'  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}')

    accuracy_proxy = mean_similarity 
    
    return accuracy_proxy, predicts, metrics_dict


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = [
        (sphere20(512), 'weights/sphere20_mcp.pth'),
        (sphere36(512), 'weights/sphere36_mcp.pth'),
        (MobileNetV1(512), 'weights/mobilenetv1_mcp.pth'),
        (MobileNetV2(512), 'weights/mobilenetv2_mcp.pth'),
        (mobilenet_v3_small(512), 'weights/mobilenetv3_small_mcp.pth'),
        (mobilenet_v3_large(512), 'weights/mobilenetv3_large_mcp.pth')
    ]
    
    for model, model_path in models_to_test:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*70}")
        _, _, metrics = eval(
            model.to(device), 
            model_path=model_path,
            compute_metrics=True,
            save_plots=True,
            plots_dir='weights'
        )