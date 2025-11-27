import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm

try:
    from uniface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("Warning: uniface not installed. Landmark annotation will not be available.")

from utils.general import LOGGER


class LandmarkAnnotator:
    """
    Anota landmarks faciais para todas as imagens de um dataset.
    Salva em cache JSON para evitar reprocessamento.
    """
    
    def __init__(
        self,
        cache_dir: str = "landmark_cache",
        conf_threshold: float = 0.5,
        image_size: int = 112
    ):
        """
        Args:
            cache_dir: Diretório para salvar cache de landmarks
            conf_threshold: Threshold de confiança do RetinaFace
            image_size: Tamanho das imagens (para normalização dos landmarks)
        """
        if not RETINAFACE_AVAILABLE:
            raise ImportError(
                "uniface package required. Install with: pip install uniface"
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.detector = None  # Lazy init
        
        # Estatísticas
        self.stats = {
            'total_images': 0,
            'annotated': 0,
            'failed': 0,
            'failed_paths': []
        }
    
    def _init_detector(self):
        """Inicializa detector apenas quando necessário"""
        if self.detector is None:
            LOGGER.info("Initializing RetinaFace detector...")
            self.detector = RetinaFace()
            LOGGER.info("RetinaFace detector initialized")
    
    def _get_cache_path(self, dataset_name: str) -> Path:
        """Retorna caminho do arquivo de cache para o dataset"""
        return self.cache_dir / f"{dataset_name}_landmarks.json"
    
    def _extract_landmarks(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Extrai landmarks de uma imagem.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Lista de 5 pontos [[x1,y1], [x2,y2], ...] normalizados ou None se falhar
        """
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
            boxes, landmarks = self.detector.detect(img)
            
            # Sem detecção
            if boxes is None or len(boxes) == 0:
                return None
            
            # Filtra por confiança
            valid_idx = None
            for i, box in enumerate(boxes):
                if len(box) >= 5 and box[4] >= self.conf_threshold:
                    valid_idx = i
                    break
            
            if valid_idx is None:
                return None
            
            # Pega landmarks da face mais confiável
            lmks = landmarks[valid_idx]  # Shape: (5, 2)
            
            # Normaliza para [0, 1] baseado no tamanho da imagem
            h, w = img.shape[:2]
            normalized = []
            for point in lmks:
                normalized.append([
                    float(point[0] / w),
                    float(point[1] / h)
                ])
            
            return normalized
            
        except Exception as e:
            return None
    
    def annotate_dataset(
        self,
        dataset_root: str,
        dataset_name: str,
        force_reannotate: bool = False
    ) -> Dict[str, List[List[float]]]:
        """
        Anota todas as imagens de um dataset.
        
        Args:
            dataset_root: Diretório raiz do dataset
            dataset_name: Nome para identificar o cache
            force_reannotate: Se True, ignora cache existente
            
        Returns:
            Dict mapeando image_path -> landmarks
        """
        cache_path = self._get_cache_path(dataset_name)
        
        # Tenta carregar do cache
        if cache_path.exists() and not force_reannotate:
            LOGGER.info(f"Loading landmarks from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            self.stats = cached_data.get('stats', self.stats)
            landmarks_dict = cached_data.get('landmarks', {})
            
            self._log_stats()
            return landmarks_dict
        
        # Inicializa detector
        self._init_detector()
        
        # Coleta todos os caminhos de imagem
        LOGGER.info(f"Scanning dataset: {dataset_root}")
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        self.stats['total_images'] = len(image_paths)
        LOGGER.info(f"Found {len(image_paths)} images to process")
        
        # Processa cada imagem
        landmarks_dict = {}
        self.stats['annotated'] = 0
        self.stats['failed'] = 0
        self.stats['failed_paths'] = []
        
        for img_path in tqdm(image_paths, desc="Extracting landmarks"):
            landmarks = self._extract_landmarks(img_path)
            
            if landmarks is not None:
                # Usa caminho relativo como chave
                rel_path = os.path.relpath(img_path, dataset_root)
                landmarks_dict[rel_path] = landmarks
                self.stats['annotated'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_paths'].append(img_path)
        
        # Salva no cache
        cache_data = {
            'stats': {
                'total_images': self.stats['total_images'],
                'annotated': self.stats['annotated'],
                'failed': self.stats['failed'],
                'failed_paths': self.stats['failed_paths']
            },
            'landmarks': landmarks_dict,
            'config': {
                'conf_threshold': self.conf_threshold,
                'image_size': self.image_size,
                'num_landmarks': 5
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        LOGGER.info(f"Landmarks cached to: {cache_path}")
        self._log_stats()
        
        return landmarks_dict
    
    def _log_stats(self):
        """Loga estatísticas de anotação"""
        total = self.stats['total_images']
        annotated = self.stats['annotated']
        failed = self.stats['failed']
        
        success_rate = (annotated / total * 100) if total > 0 else 0
        
        LOGGER.info("=" * 60)
        LOGGER.info("LANDMARK ANNOTATION STATISTICS")
        LOGGER.info("=" * 60)
        LOGGER.info(f"Total images:      {total}")
        LOGGER.info(f"Successfully annotated: {annotated} ({success_rate:.2f}%)")
        LOGGER.info(f"Failed (excluded):      {failed} ({100 - success_rate:.2f}%)")
        LOGGER.info("=" * 60)
        
        if failed > 0:
            LOGGER.warning(
                f"{failed} images will be EXCLUDED from training due to "
                "failed landmark detection"
            )
    
    def get_valid_samples(
        self,
        samples: List[Tuple[str, int]],
        landmarks_dict: Dict[str, List[List[float]]],
        dataset_root: str
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Filtra samples válidos (que possuem landmarks).
        
        Args:
            samples: Lista de (image_path, label) do dataset original
            landmarks_dict: Dict de landmarks anotados
            dataset_root: Raiz do dataset para calcular caminho relativo
            
        Returns:
            Tuple de (valid_samples, excluded_samples)
        """
        valid = []
        excluded = []
        
        for img_path, label in samples:
            rel_path = os.path.relpath(img_path, dataset_root)
            
            if rel_path in landmarks_dict:
                valid.append((img_path, label))
            else:
                excluded.append((img_path, label))
        
        LOGGER.info(f"Samples with landmarks: {len(valid)}")
        LOGGER.info(f"Samples excluded (no landmarks): {len(excluded)}")
        
        return valid, excluded
    
    def get_landmarks_for_image(
        self,
        image_path: str,
        landmarks_dict: Dict[str, List[List[float]]],
        dataset_root: str
    ) -> Optional[np.ndarray]:
        """
        Obtém landmarks para uma imagem específica.
        
        Args:
            image_path: Caminho absoluto da imagem
            landmarks_dict: Dict de landmarks
            dataset_root: Raiz do dataset
            
        Returns:
            np.ndarray de shape (5, 2) ou None
        """
        rel_path = os.path.relpath(image_path, dataset_root)
        
        if rel_path in landmarks_dict:
            return np.array(landmarks_dict[rel_path], dtype=np.float32)
        
        return None


def extract_landmarks_single_image(
    image: np.ndarray,
    detector: RetinaFace = None,
    conf_threshold: float = 0.5
) -> Optional[np.ndarray]:
    """
    Extrai landmarks de uma única imagem (para inferência).
    
    Args:
        image: Imagem como np.ndarray (RGB)
        detector: Instância do RetinaFace (cria nova se None)
        conf_threshold: Threshold de confiança
        
    Returns:
        np.ndarray de shape (5, 2) normalizado ou None
    """
    if detector is None:
        detector = RetinaFace()
    
    try:
        boxes, landmarks = detector.detect(image)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Encontra face mais confiável
        valid_idx = None
        best_conf = 0
        for i, box in enumerate(boxes):
            if len(box) >= 5 and box[4] >= conf_threshold:
                if box[4] > best_conf:
                    best_conf = box[4]
                    valid_idx = i
        
        if valid_idx is None:
            return None
        
        lmks = landmarks[valid_idx]
        
        # Normaliza
        h, w = image.shape[:2]
        normalized = np.array([
            [lmks[i][0] / w, lmks[i][1] / h]
            for i in range(5)
        ], dtype=np.float32)
        
        return normalized
        
    except Exception:
        return None