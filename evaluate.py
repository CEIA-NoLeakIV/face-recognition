import os
import csv
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from collections import defaultdict

def extract_deep_features(model, img, device):
    """Extract deep features from an image"""
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze(0).cpu()


def compute_metrics_from_predictions(predictions, threshold=0.35):
    """
    Compute classification metrics from predictions array
    
    Args:
        predictions: Array with format [path1, path2, similarity, ground_truth]
        threshold: Similarity threshold
        
    Returns:
        dict: Dictionary with metrics
    """
    if len(predictions) == 0:
        return {}
    
    # Extract ground truth and similarities
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    # Compute predictions based on threshold
    y_pred = (similarities > threshold).astype(int)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Statistics
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    median_similarity = np.median(similarities)
    
    # Find best threshold (maximum accuracy)
    thresholds = np.linspace(similarities.min(), similarities.max(), 100)
    accuracies = []
    for t in thresholds:
        y_pred_t = (similarities > t).astype(int)
        acc_t = accuracy_score(y_true, y_pred_t)
        accuracies.append(acc_t)
    
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'mean_similarity': mean_similarity,
        'std_similarity': std_similarity,
        'min_similarity': min_similarity,
        'max_similarity': max_similarity,
        'median_similarity': median_similarity,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0
    }


def compute_roc_metrics(predictions, save_path=None):
    """
    Compute ROC curve and related metrics
    
    Args:
        predictions: Array with format [path1, path2, similarity, ground_truth]
        save_path: Path to save ROC curve plot (optional)
        
    Returns:
        dict: Dictionary with ROC metrics
    """
    if len(predictions) == 0:
        return {}
    
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Compute EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    # Compute TAR@FAR metrics
    tar_at_far_001 = tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0
    tar_at_far_01 = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0
    tar_at_far_1 = tpr[np.where(fpr <= 0.1)[0][-1]] if np.any(fpr <= 0.1) else 0
    
    # Generate plot if save_path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.plot(fpr[eer_threshold_idx], tpr[eer_threshold_idx], 'ro', markersize=10, 
                 label=f'EER = {eer:.4f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (TAR)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'TAR@FAR=0.001': tar_at_far_001,
        'TAR@FAR=0.01': tar_at_far_01,
        'TAR@FAR=0.1': tar_at_far_1,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'far': fpr.mean(),
        'frr': (1 - tpr).mean()
    }


def compute_confusion_matrix(predictions, threshold, save_path=None):
    """
    Compute and optionally plot confusion matrix
    
    Args:
        predictions: Array with format [path1, path2, similarity, ground_truth]
        threshold: Similarity threshold
        save_path: Path to save plot (optional)
        
    Returns:
        np.ndarray: Confusion matrix
    """
    if len(predictions) == 0:
        return None
    
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    y_pred = (similarities > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
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


def load_audit_log_pairs(audit_log_path, val_root):
    """
    Load image pairs from audit_log.csv file.
    
    Args:
        audit_log_path: Path to audit_log.csv file
        val_root: Root directory where images are stored (can contain train/ and val/ subdirectories)
        
    Returns:
        list: List of tuples (path1, path2, is_same) where is_same is '1' or '0'
    """
    pairs = []
    
    if not os.path.exists(audit_log_path):
        raise FileNotFoundError(f"Audit log file not found: {audit_log_path}")
    
    def find_image_path(filename, cpf=None):
        """Helper function to find image path in various possible locations."""
        # List of possible locations to search
        search_paths = [
            os.path.join(val_root, filename),  # Direct in root
            os.path.join(val_root, 'train', filename),  # In train/ subdirectory
            os.path.join(val_root, 'val', filename),  # In val/ subdirectory
        ]
        
        # If CPF provided, also try organizing by CPF
        if cpf:
            search_paths.append(os.path.join(val_root, cpf, filename))
        
        # Try each path
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    with open(audit_log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            query_filename = row.get('query_filename', '').strip()
            match_filename = row.get('match_filename', '').strip()
            query_cpf = row.get('query_cpf', '').strip()
            match_cpf = row.get('match_cpf', '').strip()
            
            if not query_filename or not match_filename:
                continue
            
            # Determine ground truth: same person if CPFs match
            is_same = '1' if query_cpf == match_cpf else '0'
            
            # Find both image paths
            path1 = find_image_path(query_filename, query_cpf)
            path2 = find_image_path(match_filename, match_cpf)
            
            # Only add pair if both images exist
            if path1 and path2:
                pairs.append((path1, path2, is_same))
    
    return pairs


def load_mapping_val_pairs(mapping_val_path, max_pairs_per_person=None, negative_ratio=1.0, seed=42):
    """
    Load image pairs from mapping_val.csv file by generating positive and negative pairs.
    
    Args:
        mapping_val_path: Path to mapping_val.csv file
        max_pairs_per_person: Maximum number of positive pairs to generate per person (None = all)
        negative_ratio: Ratio of negative pairs to positive pairs (1.0 = equal numbers)
        seed: Random seed for reproducible negative pair generation
        
    Returns:
        list: List of tuples (path1, path2, is_same) where is_same is '1' or '0'
    """
    pairs = []
    
    if not os.path.exists(mapping_val_path):
        raise FileNotFoundError(f"Mapping val file not found: {mapping_val_path}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Group images by CPF
    images_by_cpf = defaultdict(list)
    
    with open(mapping_val_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cpf = row.get('cpf', '').strip()
            caminho_imagem = row.get('caminho_imagem', '').strip()
            
            if not cpf or not caminho_imagem:
                continue
            
            # Check if image path exists (caminho_imagem is already an absolute path)
            if os.path.exists(caminho_imagem):
                images_by_cpf[cpf].append(caminho_imagem)
    
    # Generate positive pairs (same CPF)
    positive_pairs = []
    for cpf, images in images_by_cpf.items():
        if len(images) < 2:
            continue  # Need at least 2 images for a pair
        
        # Generate all pairs for this person
        person_pairs = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                person_pairs.append((images[i], images[j], '1'))
        
        # Limit pairs per person if specified
        if max_pairs_per_person and len(person_pairs) > max_pairs_per_person:
            person_pairs = random.sample(person_pairs, max_pairs_per_person)
        
        positive_pairs.extend(person_pairs)
    
    pairs.extend(positive_pairs)
    
    # Generate negative pairs (different CPFs)
    num_negative = int(len(positive_pairs) * negative_ratio)
    
    cpfs_list = list(images_by_cpf.keys())
    negative_pairs = []
    negative_pairs_set = set()  # Use set for efficient duplicate checking
    
    attempts = 0
    max_attempts = num_negative * 20  # Prevent infinite loop
    
    while len(negative_pairs) < num_negative and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two different CPFs
        cpf1, cpf2 = random.sample(cpfs_list, 2)
        
        # Get random images from each CPF
        img1 = random.choice(images_by_cpf[cpf1])
        img2 = random.choice(images_by_cpf[cpf2])
        
        # Create pair tuple (ensure consistent ordering for set)
        if img1 < img2:
            pair_tuple = (img1, img2)
        else:
            pair_tuple = (img2, img1)
        
        # Avoid duplicate pairs
        if pair_tuple not in negative_pairs_set:
            negative_pairs_set.add(pair_tuple)
            negative_pairs.append((img1, img2, '0'))
    
    pairs.extend(negative_pairs)
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    return pairs


def eval(
    model, 
    model_path=None, 
    device=None, 
    val_dataset='lfw', 
    val_root='data/lfw/val', 
    compute_full_metrics=False, 
    save_metrics_path=None, 
    threshold=0.35,
    face_validator=None,
    no_face_policy='exclude'
):
    """
    Evaluate the model on validation dataset (LFW, CelebA, audit_log, or mapping_val).
    
    Args:
        model: The model to evaluate
        model_path: Path to model weights (optional)
        device: Device to run evaluation on
        val_dataset: Dataset to use for validation ('lfw', 'celeba', 'audit_log', or 'mapping_val')
        val_root: Root directory of validation data (for audit_log/mapping_val, should contain the CSV file)
        compute_full_metrics: If True, compute complete metrics (ROC, confusion matrix, etc)
        save_metrics_path: Directory to save metric plots
        threshold: Similarity threshold for classification metrics
        face_validator: FaceValidator instance (optional)
        no_face_policy: Policy for images without faces ('exclude' or 'include')
        
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
    
    # Select annotation file based on dataset
    if val_dataset == 'lfw':
        ann_file = os.path.join(root, 'lfw_ann.txt')
        try:
            with open(ann_file) as f:
                lines = f.readlines()
                pair_lines = lines[1:]  # Skip header
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'lfw_ann.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
    elif val_dataset == 'celeba':
        ann_file = os.path.join(root, 'celeba_pairs.txt')
        try:
            with open(ann_file) as f:
                pair_lines = f.readlines()[1:]  # Skip header
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'celeba_pairs.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
    elif val_dataset == 'audit_log':
        ann_file = os.path.join(root, 'audit_log.csv')
        try:
            pairs = load_audit_log_pairs(ann_file, root)
            # Convert pairs to a format compatible with existing processing
            pair_lines = pairs  # This will be handled differently in processing loop
        except FileNotFoundError:
            print(f"ERROR: Audit log file 'audit_log.csv' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
        except Exception as e:
            print(f"ERROR: Failed to load audit log: {e}")
            return 0.0, np.array([]), {}
    elif val_dataset == 'mapping_val':
        ann_file = os.path.join(root, 'mapping_val.csv')
        try:
            pairs = load_mapping_val_pairs(ann_file)
            # Convert pairs to a format compatible with existing processing
            pair_lines = pairs  # This will be handled differently in processing loop
        except FileNotFoundError:
            print(f"ERROR: Mapping val file 'mapping_val.csv' not found in '{root}'. Check the path.")
            return 0.0, np.array([]), {}
        except Exception as e:
            print(f"ERROR: Failed to load mapping_val: {e}")
            return 0.0, np.array([]), {}
    else:
        raise ValueError(f"Unsupported validation dataset: {val_dataset}. Choose 'lfw', 'celeba', 'audit_log', or 'mapping_val'.")

    # Face validation with RetinaFace (if enabled)
    use_validated_pairs = False
    validated_pairs_list = []
    
    if face_validator is not None:
        print(f"\n{'='*70}")
        print("FACE VALIDATION WITH RETINAFACE")
        print(f"{'='*70}")
        
        from utils.face_validation import validate_lfw_pairs, validate_audit_log_pairs, print_validation_summary
        
        if val_dataset in ['audit_log', 'mapping_val']:
            # For audit_log and mapping_val, pairs are already in (path1, path2, is_same) format
            validated_pairs_list, excluded_pairs, face_stats = validate_audit_log_pairs(
                validator=face_validator,
                audit_log_pairs=pair_lines,
                policy=no_face_policy
            )
            use_validated_pairs = True
        else:
            # For lfw and celeba, use existing validation
            valid_pairs, excluded_pairs, face_stats = validate_lfw_pairs(
                validator=face_validator,
                lfw_root=root,
                ann_file=ann_file,
                policy=no_face_policy
            )
            
            # Convert back to pair_lines format for lfw/celeba
            pair_lines_filtered = []
            for path1, path2, is_same in valid_pairs:
                if val_dataset == 'lfw':
                    # Extract person name and image number from path
                    parts1 = path1.replace('\\', '/').split('/')
                    parts2 = path2.replace('\\', '/').split('/')
                    
                    person1 = parts1[-2]
                    person2 = parts2[-2]
                    
                    filename1 = parts1[-1]
                    filename2 = parts2[-1]
                    
                    img_num1 = filename1.split('_')[-1].split('.')[0]
                    img_num2 = filename2.split('_')[-1].split('.')[0]
                    
                    if person1 == person2:
                        pair_lines_filtered.append(f"{person1} {img_num1} {img_num2}\n")
                    else:
                        pair_lines_filtered.append(f"{person1} {img_num1} {person2} {img_num2}\n")
            
            pair_lines = pair_lines_filtered
        
        # Print face validation summary
        print_validation_summary(face_validator)
        
        print(f"\nPair Filtering Statistics:")
        print(f"  Total pairs:     {face_stats['total_pairs']}")
        print(f"  Valid pairs:     {face_stats['valid_pairs']}")
        print(f"  Excluded pairs:  {face_stats['excluded_pairs']} ({face_stats['exclusion_rate']:.2f}%)")
        print(f"  Policy:          {no_face_policy}")
        print(f"{'='*70}\n")
        
        if val_dataset in ['audit_log', 'mapping_val']:
            print(f"Evaluating on {len(validated_pairs_list)} validated pairs...")
        else:
            print(f"Evaluating on {len(pair_lines)} validated pairs...")

    # Process pairs
    predicts = []
    with torch.no_grad():
        if val_dataset in ['audit_log', 'mapping_val']:
            # For audit_log and mapping_val, pairs are already in (path1, path2, is_same) format
            pairs_to_process = validated_pairs_list if (use_validated_pairs and len(validated_pairs_list) > 0) else pair_lines
            for path1, path2, is_same in pairs_to_process:
                try:
                    img1 = Image.open(path1).convert('RGB')
                    img2 = Image.open(path2).convert('RGB')
                except FileNotFoundError:
                    print(f"Warning: Image not found, skipping pair: {path1} or {path2}")
                    continue
                except Exception as e:
                    print(f"Warning: Error loading images, skipping pair: {path1} or {path2} - {e}")
                    continue

                f1 = extract_deep_features(model, img1, device)
                f2 = extract_deep_features(model, img2, device)

                distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
                predicts.append([path1, path2, distance.item(), is_same])
        else:
            # For lfw and celeba, process line-by-line
            for line in pair_lines:
                parts = line.strip().split()

                if val_dataset == 'lfw':
                    if len(parts) == 3:
                        # Positive pair
                        person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                        
                        filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                        filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                        
                        path1 = os.path.join(root, person_name, filename1)
                        path2 = os.path.join(root, person_name, filename2)
                        is_same = '1'
                    elif len(parts) == 4:
                        # Negative pair
                        person1, img_num1, person2, img_num2 = parts
                        
                        filename1 = f'{person1}_{int(img_num1):04d}.jpg'
                        filename2 = f'{person2}_{int(img_num2):04d}.jpg'
                        
                        path1 = os.path.join(root, person1, filename1)
                        path2 = os.path.join(root, person2, filename2)
                        is_same = '0'
                    else:
                        continue
                        
                elif val_dataset == 'celeba':
                    if len(parts) == 2:
                        filename1, filename2 = parts[0], parts[1]
                        
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
    
    predicts = np.array(predicts, dtype=object)
    
    # Basic metric: mean similarity
    similarities = predicts[:, 2].astype(float)
    mean_similarity = np.mean(similarities)
    
    # Initialize metrics dict
    metrics = {
        'mean_similarity': mean_similarity,
        'std_similarity': np.std(similarities)
    }
    
    # Compute full metrics if requested
    if compute_full_metrics:
        # Classification metrics
        classification_metrics = compute_metrics_from_predictions(predicts, threshold)
        metrics.update(classification_metrics)
        
        # ROC metrics
        if save_metrics_path:
            roc_save_path = os.path.join(save_metrics_path, f'{val_dataset}_roc_curve.png')
            roc_metrics = compute_roc_metrics(predicts, save_path=roc_save_path)
            metrics.update(roc_metrics)
            
            # Confusion matrix
            cm_save_path = os.path.join(save_metrics_path, f'{val_dataset}_confusion_matrix.png')
            compute_confusion_matrix(predicts, threshold, save_path=cm_save_path)
        else:
            roc_metrics = compute_roc_metrics(predicts)
            metrics.update(roc_metrics)
    
    return mean_similarity, predicts, metrics