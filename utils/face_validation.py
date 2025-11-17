import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm

try:
    from uniface import RetinaFace
    UNIFACE_AVAILABLE = True
except ImportError:
    UNIFACE_AVAILABLE = False
    print("Warning: uniface not installed. RetinaFace validation will not be available.")


class FaceValidator:
    """
    Validates faces in datasets using RetinaFace detector from UniFace
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.5,
        cache_dir: str = "face_validation_cache"
    ):
        """
        Initialize FaceValidator
        
        Args:
            conf_threshold: Confidence threshold for face detection
            cache_dir: Directory to store validation cache files
        """
        
        if not UNIFACE_AVAILABLE:
            raise ImportError(
                "uniface package is required for RetinaFace validation. "
                "Install it with: pip install uniface"
            )
        
        self.detector = RetinaFace()
        self.conf_threshold = conf_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics
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
        Detect faces in a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (has_valid_face, num_faces, status_message)
        """
        try:
            # Load image
            img = np.array(Image.open(image_path).convert('RGB'))
            
            # Detect faces with uniface RetinaFace
            boxes, landmarks = self.detector.detect(img)
            
            # Check if faces were detected
            if boxes is None or len(boxes) == 0:
                return False, 0, "no_face_detected"
            
            # Filter by confidence (boxes format: [x1, y1, x2, y2, confidence])
            valid_faces = []
            for box in boxes:
                if len(box) >= 5:  # Has confidence score
                    confidence = box[4]
                    if confidence >= self.conf_threshold:
                        valid_faces.append(box)
                else:
                    # No confidence score, include by default
                    valid_faces.append(box)
            
            num_faces = len(valid_faces)
            
            if num_faces == 0:
                return False, 0, "no_face_detected"
            elif num_faces == 1:
                return True, 1, "valid"
            else:
                return False, num_faces, "multiple_faces"
                
        except Exception as e:
            return False, 0, f"error: {str(e)}"
    
    def validate_dataset_images(
        self,
        image_paths: List[str],
        dataset_name: str = "validation",
        force_revalidate: bool = False
    ) -> Dict[str, dict]:
        """
        Validate all images in a list
        
        Args:
            image_paths: List of image paths to validate
            dataset_name: Name for caching purposes
            force_revalidate: If True, ignore cache and revalidate
            
        Returns:
            Dictionary mapping image_path -> validation_result
        """
        cache_file = self.cache_dir / f"{dataset_name}_validation.json"
        
        # Try to load from cache
        if cache_file.exists() and not force_revalidate:
            print(f"Loading validation results from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                self.validation_results = json.load(f)
            self._update_stats_from_results()
            return self.validation_results
        
        # Validate images
        print(f"Validating {len(image_paths)} images with RetinaFace...")
        self.validation_results = {}
        
        for img_path in tqdm(image_paths, desc="Face Detection"):
            has_face, num_faces, status = self.detect_face(img_path)
            
            self.validation_results[img_path] = {
                'has_valid_face': has_face,
                'num_faces': num_faces,
                'status': status
            }
            
            # Update statistics
            self.stats['total_images'] += 1
            if status == "valid":
                self.stats['faces_detected'] += 1
            elif status == "no_face_detected":
                self.stats['no_faces'] += 1
            elif status == "multiple_faces":
                self.stats['multiple_faces'] += 1
            else:
                self.stats['failed_images'] += 1
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"Validation results cached to: {cache_file}")
        return self.validation_results
    
    def _update_stats_from_results(self):
        """Update statistics from loaded validation results"""
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
        """Get validation statistics"""
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
    
    def filter_valid_images(
        self,
        image_paths: List[str],
        policy: str = 'exclude'
    ) -> Tuple[List[str], List[str]]:
        """
        Filter images based on face detection results
        
        Args:
            image_paths: List of image paths to filter
            policy: 'exclude' to remove invalid images, 'include' to keep all
            
        Returns:
            Tuple of (valid_images, excluded_images)
        """
        if policy == 'include':
            return image_paths, []
        
        valid_images = []
        excluded_images = []
        
        for img_path in image_paths:
            if img_path not in self.validation_results:
                # If not validated, include by default
                valid_images.append(img_path)
                continue
            
            result = self.validation_results[img_path]
            if result['has_valid_face']:
                valid_images.append(img_path)
            else:
                excluded_images.append(img_path)
        
        return valid_images, excluded_images
    
    def save_validation_report(
        self,
        output_path: str,
        dataset_name: str = "validation"
    ):
        """
        Save detailed validation report as JSON
        
        Args:
            output_path: Path to save the report
            dataset_name: Name of the dataset
        """
        # Collect images by status
        images_by_status = {
            'no_face_detected': [],
            'multiple_faces': [],
            'errors': [],
            'valid': []
        }
        
        for img_path, result in self.validation_results.items():
            status = result['status']
            if status.startswith('error'):
                images_by_status['errors'].append({
                    'path': img_path,
                    'error': status
                })
            elif status == "no_face_detected":
                images_by_status['no_face_detected'].append(img_path)
            elif status == "multiple_faces":
                images_by_status['multiple_faces'].append({
                    'path': img_path,
                    'num_faces': result['num_faces']
                })
            else:
                images_by_status['valid'].append(img_path)
        
        # Create report
        report = {
            'dataset_name': dataset_name,
            'statistics': self.get_validation_stats(),
            'detector_config': {
                'model': 'RetinaFace (UniFace)',
                'confidence_threshold': self.conf_threshold
            },
            'images_without_faces': images_by_status['no_face_detected'],
            'images_with_multiple_faces': images_by_status['multiple_faces'],
            'images_with_errors': images_by_status['errors'],
            'summary': {
                'total_images': self.stats['total_images'],
                'images_with_valid_face': self.stats['faces_detected'],
                'images_without_face': self.stats['no_faces'],
                'images_with_multiple_faces': self.stats['multiple_faces'],
                'images_with_errors': self.stats['failed_images']
            }
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nValidation report saved to: {output_path}")
        return report


def validate_lfw_pairs(
    validator: FaceValidator,
    lfw_root: str,
    ann_file: str,
    policy: str = 'exclude'
) -> Tuple[List, List, Dict]:
    """
    Validate LFW pairs and filter based on face detection
    
    Args:
        validator: FaceValidator instance
        lfw_root: Root directory of LFW dataset
        ann_file: Path to annotation file
        policy: 'exclude' or 'include' for images without faces
        
    Returns:
        Tuple of (valid_pairs, excluded_pairs, statistics)
    """
    # Read pairs
    with open(ann_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    # Collect all unique image paths
    all_image_paths = set()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Positive pair
            person_name, img1, img2 = parts
            path1 = os.path.join(lfw_root, person_name, f'{person_name}_{int(img1):04d}.jpg')
            path2 = os.path.join(lfw_root, person_name, f'{person_name}_{int(img2):04d}.jpg')
            all_image_paths.add(path1)
            all_image_paths.add(path2)
        elif len(parts) == 4:
            # Negative pair
            person1, img1, person2, img2 = parts
            path1 = os.path.join(lfw_root, person1, f'{person1}_{int(img1):04d}.jpg')
            path2 = os.path.join(lfw_root, person2, f'{person2}_{int(img2):04d}.jpg')
            all_image_paths.add(path1)
            all_image_paths.add(path2)
    
    # Validate all images
    all_image_paths = list(all_image_paths)
    validator.validate_dataset_images(all_image_paths, dataset_name="lfw")
    
    # Filter pairs
    valid_pairs = []
    excluded_pairs = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            person_name, img1, img2 = parts
            path1 = os.path.join(lfw_root, person_name, f'{person_name}_{int(img1):04d}.jpg')
            path2 = os.path.join(lfw_root, person_name, f'{person_name}_{int(img2):04d}.jpg')
            is_same = '1'
        elif len(parts) == 4:
            person1, img1, person2, img2 = parts
            path1 = os.path.join(lfw_root, person1, f'{person1}_{int(img1):04d}.jpg')
            path2 = os.path.join(lfw_root, person2, f'{person2}_{int(img2):04d}.jpg')
            is_same = '0'
        else:
            continue
        
        # Check if both images have valid faces
        result1 = validator.validation_results.get(path1, {})
        result2 = validator.validation_results.get(path2, {})
        
        has_face1 = result1.get('has_valid_face', True)  # Default True if not validated
        has_face2 = result2.get('has_valid_face', True)
        
        if policy == 'include' or (has_face1 and has_face2):
            valid_pairs.append((path1, path2, is_same))
        else:
            excluded_pairs.append((path1, path2, is_same, result1, result2))
    
    stats = {
        'total_pairs': len(lines),
        'valid_pairs': len(valid_pairs),
        'excluded_pairs': len(excluded_pairs),
        'exclusion_rate': len(excluded_pairs) / len(lines) * 100 if len(lines) > 0 else 0
    }
    
    return valid_pairs, excluded_pairs, stats


def print_validation_summary(validator: FaceValidator):
    """Print formatted validation summary"""
    stats = validator.get_validation_stats()
    
    print("\n" + "="*70)
    print("FACE DETECTION VALIDATION SUMMARY (RetinaFace)")
    print("="*70)
    print(f"Total images processed:        {stats['total_images']}")
    print(f"✅ Valid faces detected:       {stats['faces_detected']} ({stats['detection_rate']:.2f}%)")
    print(f"❌ No faces detected:          {stats['no_faces']} ({stats['no_face_rate']:.2f}%)")
    print(f"⚠️  Multiple faces detected:   {stats['multiple_faces']} ({stats['multiple_faces_rate']:.2f}%)")
    print(f"🔴 Processing errors:          {stats['failed_images']} ({stats['error_rate']:.2f}%)")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: Validate LFW dataset
    validator = FaceValidator(gpu_id=0, conf_threshold=0.5)
    
    lfw_root = "data/lfw/val"
    ann_file = os.path.join(lfw_root, "lfw_ann.txt")
    
    # Validate and filter pairs
    valid_pairs, excluded_pairs, stats = validate_lfw_pairs(
        validator, lfw_root, ann_file, policy='exclude'
    )
    
    # Print summary
    print_validation_summary(validator)
    
    print(f"\nPair Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Valid pairs: {stats['valid_pairs']}")
    print(f"  Excluded pairs: {stats['excluded_pairs']} ({stats['exclusion_rate']:.2f}%)")
    
    # Save validation report
    validator.save_validation_report(
        output_path="face_validation_cache/lfw_validation_report.json",
        dataset_name="LFW"
    )