import os
import csv
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm import tqdm
from collections import defaultdict

# Imports específicos de Landmarks
from utils.landmark_annotator import LandmarkAnnotator, extract_landmarks_single_image

# --- FUNÇÕES DE CARREGAMENTO DE DATASETS ---

def load_audit_log_pairs(audit_log_path, val_root):
    """Load image pairs from audit_log.csv file."""
    pairs = []
    
    if not os.path.exists(audit_log_path):
        raise FileNotFoundError(f"Audit log file not found: {audit_log_path}")
    
    def find_image_path(filename, cpf=None):
        search_paths = [
            os.path.join(val_root, filename),
            os.path.join(val_root, 'train', filename),
            os.path.join(val_root, 'val', filename),
        ]
        if cpf:
            search_paths.append(os.path.join(val_root, cpf, filename))
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
            
            is_same = '1' if query_cpf == match_cpf else '0'
            path1 = find_image_path(query_filename, query_cpf)
            path2 = find_image_path(match_filename, match_cpf)
            
            if path1 and path2:
                pairs.append((path1, path2, is_same))
    
    return pairs


def load_mapping_val_pairs(mapping_val_path, max_pairs_per_person=None, negative_ratio=1.0, seed=42):
    """Load image pairs from mapping_val.csv file."""
    pairs = []
    
    if not os.path.exists(mapping_val_path):
        raise FileNotFoundError(f"Mapping val file not found: {mapping_val_path}")
    
    random.seed(seed)
    images_by_cpf = defaultdict(list)
    
    with open(mapping_val_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpf = row.get('cpf', '').strip()
            caminho_imagem = row.get('caminho_imagem', '').strip()
            if not cpf or not caminho_imagem:
                continue
            if os.path.exists(caminho_imagem):
                images_by_cpf[cpf].append(caminho_imagem)
    
    positive_pairs = []
    for cpf, images in images_by_cpf.items():
        if len(images) < 2:
            continue
        person_pairs = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                person_pairs.append((images[i], images[j], '1'))
        if max_pairs_per_person and len(person_pairs) > max_pairs_per_person:
            person_pairs = random.sample(person_pairs, max_pairs_per_person)
        positive_pairs.extend(person_pairs)
    
    pairs.extend(positive_pairs)
    
    num_negative = int(len(positive_pairs) * negative_ratio)
    cpfs_list = list(images_by_cpf.keys())
    negative_pairs = []
    negative_pairs_set = set()
    attempts = 0
    max_attempts = num_negative * 20
    
    while len(negative_pairs) < num_negative and attempts < max_attempts:
        attempts += 1
        cpf1, cpf2 = random.sample(cpfs_list, 2)
        img1 = random.choice(images_by_cpf[cpf1])
        img2 = random.choice(images_by_cpf[cpf2])
        pair_tuple = (img1, img2) if img1 < img2 else (img2, img1)
        if pair_tuple not in negative_pairs_set:
            negative_pairs_set.add(pair_tuple)
            negative_pairs.append((img1, img2, '0'))
    
    pairs.extend(negative_pairs)
    random.shuffle(pairs)
    return pairs


def load_custom_pairs(ann_file, root):
    """Load pairs from custom.txt with real filenames."""
    pairs = []
    with open(ann_file, 'r') as f:
        lines = f.readlines()[1:]
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            class_name, file1, file2 = parts
            path1 = os.path.join(root, class_name, file1)
            path2 = os.path.join(root, class_name, file2)
            is_same = '1'
        elif len(parts) == 4:
            class1, file1, class2, file2 = parts
            path1 = os.path.join(root, class1, file1)
            path2 = os.path.join(root, class2, file2)
            is_same = '0'
        else:
            continue
        pairs.append((path1, path2, is_same))
    
    return pairs


def load_celeba_pairs(ann_file, root, max_pairs=6000):
    """
    Load pairs from celeba_pairs.txt.
    Args:
        max_pairs: Limita o número de pares para evitar validação infinita em arquivos gigantes.
    """
    pairs = []
    
    # Verifica se o arquivo existe
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"CelebA pairs file not found: {ann_file}")

    with open(ann_file, 'r') as f:
        lines = f.readlines()[1:] # Pula o cabeçalho (count)
    
    # CORREÇÃO: Amostragem se o arquivo for muito grande (evita loop infinito)
    if len(lines) > max_pairs:
        print(f"⚠️  CelebA file has {len(lines)} pairs. Randomly sampling {max_pairs} for faster evaluation.")
        random.seed(42)
        lines = random.sample(lines, max_pairs)
    
    for line in lines:
        # CORREÇÃO: split() sem argumentos lida com \t e espaços múltiplos
        parts = line.strip().split()
        
        if len(parts) == 3:
            file1, file2, label = parts
            
            # Tenta encontrar o caminho correto (algumas versões do CelebA tem pastas aninhadas)
            # Tenta estrutura padrão: root/img_align_celeba/img_align_celeba/imagem.jpg
            path1 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', file1)
            path2 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', file2)
            
            # Fallback se não existir: root/imagem.jpg
            if not os.path.exists(path1):
                 path1 = os.path.join(root, file1)
                 path2 = os.path.join(root, file2)
            
            is_same = label
            pairs.append((path1, path2, is_same))
    
    return pairs


# --- FUNÇÕES CORE (Extração de Features com Landmarks) ---

def extract_deep_features(
    model,
    img,
    device,
    use_landmarks=False,
    landmarks=None,
    landmark_detector=None
):
    """
    Extract deep features from an image.
    """
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_landmarks:
            if landmarks is None:
                img_np = np.array(img)
                landmarks = extract_landmarks_single_image(img_np, landmark_detector)
                
                if landmarks is None:
                    landmarks = np.zeros((5, 2), dtype=np.float32)
            
            landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)
            features = model(img_tensor, landmarks_tensor)
        else:
            features = model(img_tensor)
    
    return features.squeeze(0).cpu()


def compute_metrics_from_predictions(predictions, threshold=0.35):
    """
    Compute classification metrics from predictions array
    """
    if len(predictions) == 0:
        return {}
    
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    y_pred = (similarities > threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    median_similarity = np.median(similarities)
    
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
    """
    if len(predictions) == 0:
        return {}
    
    y_true = predictions[:, 3].astype(int)
    similarities = predictions[:, 2].astype(float)
    
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    # TAR@FAR metrics
    tar_at_far_0001 = tpr[np.where(fpr <= 0.0001)[0][-1]] if np.any(fpr <= 0.0001) else 0
    tar_at_far_001 = tpr[np.where(fpr <= 0.001)[0][-1]] if np.any(fpr <= 0.001) else 0
    tar_at_far_01 = tpr[np.where(fpr <= 0.01)[0][-1]] if np.any(fpr <= 0.01) else 0
    tar_at_far_1 = tpr[np.where(fpr <= 0.1)[0][-1]] if np.any(fpr <= 0.1) else 0
    
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
        'TAR@FAR=0.01%': tar_at_far_0001,
        'TAR@FAR=0.1%': tar_at_far_001,
        'TAR@FAR=1%': tar_at_far_01,
        'TAR@FAR=10%': tar_at_far_1,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'far': fpr.mean(),
        'frr': (1 - tpr).mean()
    }


def compute_confusion_matrix(predictions, threshold, save_path=None):
    """
    Compute and optionally plot confusion matrix
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
    no_face_policy='exclude',
    use_landmarks=False,
    landmark_cache_dir='landmark_cache'
):
    """
    Evaluate the model on validation dataset (LFW, CelebA, AuditLog, MappingVal, Custom).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()

    root = val_root
    
    # Inicializa detector de landmarks se necessário
    landmark_detector = None
    landmarks_cache = {}
    
    if use_landmarks:
        try:
            from uniface import RetinaFace
            landmark_detector = RetinaFace()
            
            # Tenta carregar cache de landmarks
            annotator = LandmarkAnnotator(cache_dir=landmark_cache_dir)
            cache_path = annotator._get_cache_path(val_dataset)
            if cache_path.exists():
                import json
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                landmarks_cache = cached_data.get('landmarks', {})
        except ImportError:
            pass
    
    # Select annotation file and load pairs based on dataset
    pair_lines = []
    pairs_preloaded = False # Flag para indicar se pairs já são tuplas (path1, path2, is_same)

    if val_dataset == 'lfw':
        ann_file = os.path.join(root, 'lfw_ann.txt')
        try:
            with open(ann_file) as f:
                lines = f.readlines()
                pair_lines = lines[1:]
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'lfw_ann.txt' not found in '{root}'.")
            return 0.0, np.array([]), {}
            
    elif val_dataset == 'celeba':
        ann_file = os.path.join(root, 'celeba_pairs.txt')
        try:
            # CORREÇÃO: Limita a 6000 pares para não travar a validação
            pair_lines = load_celeba_pairs(ann_file, root, max_pairs=6000)
            pairs_preloaded = True
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'celeba_pairs.txt' not found in '{root}'.")
            return 0.0, np.array([]), {}
            
    elif val_dataset == 'audit_log':
        ann_file = os.path.join(root, 'audit_log.csv')
        try:
            pair_lines = load_audit_log_pairs(ann_file, root)
            pairs_preloaded = True
        except FileNotFoundError:
            print(f"ERROR: 'audit_log.csv' not found in '{root}'.")
            return 0.0, np.array([]), {}
        except Exception as e:
            print(f"ERROR loading audit_log: {e}")
            return 0.0, np.array([]), {}
            
    elif val_dataset == 'mapping_val':
        ann_file = os.path.join(root, 'mapping_val.csv')
        try:
            pair_lines = load_mapping_val_pairs(ann_file)
            pairs_preloaded = True
        except FileNotFoundError:
            print(f"ERROR: 'mapping_val.csv' not found in '{root}'.")
            return 0.0, np.array([]), {}
        except Exception as e:
            print(f"ERROR loading mapping_val: {e}")
            return 0.0, np.array([]), {}
            
    elif val_dataset == 'custom':
        ann_file = os.path.join(root, 'custom.txt')
        try:
            pair_lines = load_custom_pairs(ann_file, root)
            pairs_preloaded = True
        except FileNotFoundError:
            print(f"ERROR: 'custom.txt' not found in '{root}'.")
            return 0.0, np.array([]), {}
        except Exception as e:
            print(f"ERROR loading custom: {e}")
            return 0.0, np.array([]), {}
            
    else:
        raise ValueError(f"Unsupported validation dataset: {val_dataset}.")

    # Face validation with RetinaFace (if enabled)
    use_validated_pairs = False
    validated_pairs_list = []
    
    if face_validator is not None:
        from utils.face_validation import validate_lfw_pairs, validate_audit_log_pairs, print_validation_summary
        
        print(f"\n{'='*70}")
        print("FACE VALIDATION WITH RETINAFACE")
        print(f"{'='*70}")
        
        if val_dataset in ['audit_log', 'mapping_val', 'custom', 'celeba']:
            # Usa validate_audit_log_pairs para datasets que já carregaram tuplas
            validated_pairs_list, excluded_pairs, face_stats = validate_audit_log_pairs(
                validator=face_validator,
                audit_log_pairs=pair_lines,
                policy=no_face_policy
            )
            use_validated_pairs = True
            
        else: # LFW case (ainda são linhas de texto)
            valid_pairs, excluded_pairs, face_stats = validate_lfw_pairs(
                validator=face_validator,
                lfw_root=root,
                ann_file=ann_file,
                policy=no_face_policy
            )
            
            # Reconstrói pair_lines apenas com os válidos no formato original de texto
            pair_lines_filtered = []
            for path1, path2, is_same in valid_pairs:
                if val_dataset == 'lfw':
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
            
        print_validation_summary(face_validator)

    # Definir lista final de pares para processamento
    if use_validated_pairs:
        # Se usou validação nos novos datasets, use a lista validada
        final_pairs_iterator = validated_pairs_list
        is_iterator_tuples = True
    elif pairs_preloaded:
        # Se é novo dataset mas sem validação, use a lista carregada
        final_pairs_iterator = pair_lines
        is_iterator_tuples = True
    else:
        # LFW legado (lista de strings)
        final_pairs_iterator = pair_lines
        is_iterator_tuples = False

    # Process pairs
    predicts = []
    skipped = 0
    
    with torch.no_grad():
        for item in tqdm(final_pairs_iterator, desc="Evaluating pairs", unit="pair"):
            
            # Lógica de Parsing
            if is_iterator_tuples:
                path1, path2, is_same = item
            else:
                # LFW parsing legado
                line = item
                parts = line.strip().split()
                if val_dataset == 'lfw':
                    if len(parts) == 3:
                        person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                        filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                        filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                        path1 = os.path.join(root, person_name, filename1)
                        path2 = os.path.join(root, person_name, filename2)
                        is_same = '1'
                    elif len(parts) == 4:
                        person1, img_num1, person2, img_num2 = parts
                        filename1 = f'{person1}_{int(img_num1):04d}.jpg'
                        filename2 = f'{person2}_{int(img_num2):04d}.jpg'
                        path1 = os.path.join(root, person1, filename1)
                        path2 = os.path.join(root, person2, filename2)
                        is_same = '0'
                    else:
                        continue
                else:
                    continue

            # Carregamento
            try:
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
            except FileNotFoundError:
                skipped += 1
                continue

            # Landmarks e Features
            landmarks1 = None
            landmarks2 = None
            
            if use_landmarks:
                rel_path1 = os.path.relpath(path1, root) if root in path1 else path1
                rel_path2 = os.path.relpath(path2, root) if root in path2 else path2
                
                if rel_path1 in landmarks_cache:
                    landmarks1 = np.array(landmarks_cache[rel_path1], dtype=np.float32)
                
                if rel_path2 in landmarks_cache:
                    landmarks2 = np.array(landmarks_cache[rel_path2], dtype=np.float32)

            f1 = extract_deep_features(
                model, img1, device,
                use_landmarks=use_landmarks,
                landmarks=landmarks1,
                landmark_detector=landmark_detector
            )
            f2 = extract_deep_features(
                model, img2, device,
                use_landmarks=use_landmarks,
                landmarks=landmarks2,
                landmark_detector=landmark_detector
            )

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, distance.item(), is_same])
    
    if skipped > 0:
        print(f"Skipped {skipped} pairs (images not found)")
    
    if len(predicts) == 0:
        print("Warning: No valid pairs were processed.")
        return 0.0, np.array([]), {}
    
    predicts = np.array(predicts, dtype=object)
    
    similarities = predicts[:, 2].astype(float)
    mean_similarity = np.mean(similarities)
    
    metrics = {
        'mean_similarity': mean_similarity,
        'std_similarity': np.std(similarities)
    }
    
    if compute_full_metrics:
        classification_metrics = compute_metrics_from_predictions(predicts, threshold)
        metrics.update(classification_metrics)
        
        if save_metrics_path:
            roc_save_path = os.path.join(save_metrics_path, f'{val_dataset}_roc_curve.png')
            roc_metrics = compute_roc_metrics(predicts, save_path=roc_save_path)
            metrics.update(roc_metrics)
            
            cm_save_path = os.path.join(save_metrics_path, f'{val_dataset}_confusion_matrix.png')
            compute_confusion_matrix(predicts, threshold, save_path=cm_save_path)
        else:
            roc_metrics = compute_roc_metrics(predicts)
            metrics.update(roc_metrics)
    
    return mean_similarity, predicts, metrics