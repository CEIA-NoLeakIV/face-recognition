import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import traceback

try:
    from uniface import RetinaFace
    UNIFACE_AVAILABLE = True
except ImportError:
    UNIFACE_AVAILABLE = False
    print("Warning: uniface not installed. RetinaFace validation will not be available.")


class FaceValidator:
    """
    Validates faces in datasets using RetinaFace detector from UniFace.
    Robust hybrid version (Tuples vs Dicts) + GPU compatibility fix.
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.5,
        cache_dir: str = "face_validation_cache"
    ):
        if not UNIFACE_AVAILABLE:
            raise ImportError("uniface package is required. Install it with: pip install uniface")
        
        self.conf_threshold = conf_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            self.detector = RetinaFace()
        except Exception as e:
            print(f"Warning: Error initializing RetinaFace: {e}")
            # Tenta fallback básico se falhar
            self.detector = RetinaFace()
        
        self.validation_results = {}
        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'no_faces': 0,
            'multiple_faces': 0,
            'failed_images': 0
        }
    
    def detect_face(self, image_path: str) -> Tuple[bool, int, str]:
        """
        Detect faces in a single image (Hybrid Logic).
        Returns: (has_valid_face, num_faces, status_message)
        """
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
            
            raw_result = self.detector.detect(img)
            
            num_valid_faces = 0
            
            if not raw_result:
                return False, 0, "no_face_detected"

            # --- Lógica Híbrida ---
            
            # CASO 1: Lista de Dicionários
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                for face in raw_result:
                    # Confiança pode vir como 'confidence' ou 'score'
                    conf = face.get('confidence', face.get('score', 0.0))
                    if conf >= self.conf_threshold:
                        num_valid_faces += 1

            # CASO 2: Tupla
            elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                boxes, _ = raw_result
                if boxes is not None:
                    for box in boxes:
                        # box: [x1, y1, x2, y2, score]
                        if len(box) >= 5 and box[4] >= self.conf_threshold:
                            num_valid_faces += 1
            
            # --- Resultado Final ---
            if num_valid_faces == 0:
                return False, 0, "no_face_detected"
            elif num_valid_faces == 1:
                return True, 1, "valid"
            else:
                return False, num_valid_faces, "multiple_faces"
                
        except Exception as e:
            # Em caso de erro de leitura de imagem ou corrupção
            return False, 0, "error_processing"
    
    def validate_dataset_images(
        self,
        image_paths: List[str],
        dataset_name: str = "validation",
        force_revalidate: bool = False
    ) -> Dict[str, dict]:
        cache_file = self.cache_dir / f"{dataset_name}_validation.json"
        
        if cache_file.exists() and not force_revalidate:
            print(f"Loading validation results from cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    self.validation_results = json.load(f)
                self._update_stats_from_results()
                return self.validation_results
            except json.JSONDecodeError:
                print("Cache corrupted. Revalidating...")
        
        print(f"Validating {len(image_paths)} images with RetinaFace...")
        self.validation_results = {}
        
        # Reset stats for new validation run
        self.stats = {k: 0 for k in self.stats}
        
        for img_path in tqdm(image_paths, desc="Face Detection"):
            has_face, num_faces, status = self.detect_face(img_path)
            
            self.validation_results[img_path] = {
                'has_valid_face': has_face,
                'num_faces': num_faces,
                'status': status
            }
            
            self.stats['total_images'] += 1
            if status == "valid":
                self.stats['faces_detected'] += 1
            elif status == "no_face_detected":
                self.stats['no_faces'] += 1
            elif status == "multiple_faces":
                self.stats['multiple_faces'] += 1
            else:
                self.stats['failed_images'] += 1
        
        with open(cache_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Validation results cached to: {cache_file}")
        return self.validation_results
    
    def _update_stats_from_results(self):
        self.stats = {
            'total_images': len(self.validation_results),
            'faces_detected': 0,
            'no_faces': 0,
            'multiple_faces': 0,
            'failed_images': 0
        }
        
        for result in self.validation_results.values():
            status = result['status']
            if status == "valid":
                self.stats['faces_detected'] += 1
            elif status == "no_face_detected":
                self.stats['no_faces'] += 1
            elif status == "multiple_faces":
                self.stats['multiple_faces'] += 1
            else:
                self.stats['failed_images'] += 1
    
    def get_validation_stats(self) -> Dict:
        total = self.stats['total_images']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'detection_rate': self.stats['faces_detected'] / total * 100,
            'no_face_rate': self.stats['no_faces'] / total * 100,
            'multiple_faces_rate': self.stats['multiple_faces'] / total * 100,
            'error_rate': self.stats['failed_images'] / total * 100
        }
    
    def filter_valid_images(self, image_paths: List[str], policy: str = 'exclude'):
        if policy == 'include':
            return image_paths, []
        
        valid_images = []
        excluded_images = []
        
        for img_path in image_paths:
            if img_path not in self.validation_results:
                valid_images.append(img_path)
                continue
            
            result = self.validation_results[img_path]
            if result['has_valid_face']:
                valid_images.append(img_path)
            else:
                excluded_images.append(img_path)
        
        return valid_images, excluded_images
    
    def save_validation_report(self, output_path: str, dataset_name: str = "validation"):
        images_by_status = {
            'no_face_detected': [],
            'multiple_faces': [],
            'errors': [],
            'valid': []
        }
        
        for img_path, result in self.validation_results.items():
            status = result['status']
            if 'error' in status:
                images_by_status['errors'].append({'path': img_path, 'error': status})
            elif status == "no_face_detected":
                images_by_status['no_face_detected'].append(img_path)
            elif status == "multiple_faces":
                images_by_status['multiple_faces'].append({'path': img_path, 'num_faces': result['num_faces']})
            else:
                images_by_status['valid'].append(img_path)
        
        report = {
            'dataset_name': dataset_name,
            'statistics': self.get_validation_stats(),
            'detector_config': {'model': 'RetinaFace (UniFace)', 'confidence_threshold': self.conf_threshold},
            'images_without_faces': images_by_status['no_face_detected'],
            'images_with_multiple_faces': images_by_status['multiple_faces'],
            'images_with_errors': images_by_status['errors'],
            'summary': self.stats
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to: {output_path}")
        return report


def validate_lfw_pairs(validator, lfw_root, ann_file, policy='exclude'):
    try:
        with open(ann_file, 'r') as f:
            lines = f.readlines()[1:]
    except FileNotFoundError:
        print(f"Annotation file not found: {ann_file}")
        return [], [], {}

    all_image_paths = set()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            person, img1, img2 = parts
            all_image_paths.add(os.path.join(lfw_root, person, f'{person}_{int(img1):04d}.jpg'))
            all_image_paths.add(os.path.join(lfw_root, person, f'{person}_{int(img2):04d}.jpg'))
        elif len(parts) == 4:
            p1, i1, p2, i2 = parts
            all_image_paths.add(os.path.join(lfw_root, p1, f'{p1}_{int(i1):04d}.jpg'))
            all_image_paths.add(os.path.join(lfw_root, p2, f'{p2}_{int(i2):04d}.jpg'))
    
    validator.validate_dataset_images(list(all_image_paths), dataset_name="lfw")
    
    valid_pairs = []
    excluded_pairs = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            person, img1, img2 = parts
            p1 = os.path.join(lfw_root, person, f'{person}_{int(img1):04d}.jpg')
            p2 = os.path.join(lfw_root, person, f'{person}_{int(img2):04d}.jpg')
            is_same = '1'
        elif len(parts) == 4:
            per1, i1, per2, i2 = parts
            p1 = os.path.join(lfw_root, per1, f'{per1}_{int(i1):04d}.jpg')
            p2 = os.path.join(lfw_root, per2, f'{per2}_{int(i2):04d}.jpg')
            is_same = '0'
        else:
            continue
        
        res1 = validator.validation_results.get(p1, {})
        res2 = validator.validation_results.get(p2, {})
        
        if policy == 'include' or (res1.get('has_valid_face', True) and res2.get('has_valid_face', True)):
            valid_pairs.append((p1, p2, is_same))
        else:
            excluded_pairs.append((p1, p2, is_same, res1, res2))
    
    stats = {
        'total_pairs': len(lines),
        'valid_pairs': len(valid_pairs),
        'excluded_pairs': len(excluded_pairs),
        'exclusion_rate': len(excluded_pairs) / len(lines) * 100 if len(lines) > 0 else 0
    }
    
    return valid_pairs, excluded_pairs, stats


def validate_audit_log_pairs(validator, audit_log_pairs, policy='exclude'):
    """
    Validate faces in audit log pairs using RetinaFace.
    
    Args:
        validator: FaceValidator instance
        audit_log_pairs: List of tuples (path1, path2, is_same) from audit log
        policy: 'exclude' or 'include' for pairs without valid faces
        
    Returns:
        tuple: (valid_pairs, excluded_pairs, stats)
    """
    # Collect all unique image paths
    all_image_paths = set()
    for path1, path2, _ in audit_log_pairs:
        all_image_paths.add(path1)
        all_image_paths.add(path2)
    
    validator.validate_dataset_images(list(all_image_paths), dataset_name="audit_log")
    
    valid_pairs = []
    excluded_pairs = []
    
    for path1, path2, is_same in audit_log_pairs:
        res1 = validator.validation_results.get(path1, {})
        res2 = validator.validation_results.get(path2, {})
        
        if policy == 'include' or (res1.get('has_valid_face', True) and res2.get('has_valid_face', True)):
            valid_pairs.append((path1, path2, is_same))
        else:
            excluded_pairs.append((path1, path2, is_same, res1, res2))
    
    stats = {
        'total_pairs': len(audit_log_pairs),
        'valid_pairs': len(valid_pairs),
        'excluded_pairs': len(excluded_pairs),
        'exclusion_rate': len(excluded_pairs) / len(audit_log_pairs) * 100 if len(audit_log_pairs) > 0 else 0
    }
    
    return valid_pairs, excluded_pairs, stats


def print_validation_summary(validator):
    stats = validator.get_validation_stats()
    print("\n" + "="*70)
    print("FACE DETECTION VALIDATION SUMMARY")
    print("="*70)
    print(f"Total:      {stats['total_images']}")
    print(f"✅ Valid:    {stats['faces_detected']} ({stats.get('detection_rate',0):.2f}%)")
    print(f"❌ No Face:  {stats['no_faces']} ({stats.get('no_face_rate',0):.2f}%)")
    print(f"⚠️  Multiple: {stats['multiple_faces']} ({stats.get('multiple_faces_rate',0):.2f}%)")
    print(f"🔴 Errors:   {stats['failed_images']} ({stats.get('error_rate',0):.2f}%)")
    print("="*70 + "\n")

if __name__ == "__main__":
    validator = FaceValidator(conf_threshold=0.5)
    print("Validator initialized.")