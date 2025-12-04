import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import torch
import traceback

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
    Versão Híbrida: Suporta Uniface antigo (tupla) e novo (lista de dicts).
    """
    
    def __init__(
        self,
        cache_dir: str = "landmark_cache",
        conf_threshold: float = 0.5,
        image_size: int = 112
    ):
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
        """Inicializa detector de forma robusta (GPU/CPU)"""
        if self.detector is None:
            LOGGER.info("Initializing RetinaFace detector...")
            try:
                # Tenta usar a GPU atual se disponível
                if torch.cuda.is_available():
                    gpu_id = torch.cuda.current_device()
                    try:
                        # Tenta passar gpu_id (funciona no Docker/Versões novas)
                        self.detector = RetinaFace(gpu_id=gpu_id)
                        LOGGER.info(f"RetinaFace initialized on GPU {gpu_id}")
                    except TypeError:
                        # Se der erro de argumento (Windows/Versões específicas), usa padrão
                        # O onnxruntime-gpu cuidará da aceleração se instalado
                        self.detector = RetinaFace()
                        LOGGER.info("RetinaFace initialized (default mode)")
                else:
                    self.detector = RetinaFace()
                    LOGGER.info("RetinaFace initialized on CPU")
                    
            except Exception as e:
                LOGGER.error(f"Failed to initialize detector: {e}")
                # Fallback final
                self.detector = RetinaFace()
    
    def _get_cache_path(self, dataset_name: str) -> Path:
        """Retorna caminho do arquivo de cache para o dataset"""
        return self.cache_dir / f"{dataset_name}_landmarks.json"
    
    def _extract_landmarks(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Extrai landmarks de uma imagem lidando com diferentes versões da lib.
        """
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
            
            # Captura o resultado sem desempacotar
            raw_result = self.detector.detect(img)
            
            if not raw_result:
                return None

            selected_landmarks = None

            # --- Lógica Híbrida ---

            # CASO 1: Lista de Dicionários
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                best_face = None
                best_conf = -1.0
                
                for face in raw_result:
                    conf = face.get('confidence', face.get('score', 0.0))
                    if conf >= self.conf_threshold and conf > best_conf:
                        best_conf = conf
                        best_face = face
                
                if best_face is not None:
                    selected_landmarks = best_face.get('landmarks', best_face.get('kps'))

            # CASO 2: Tupla
            elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                boxes, landmarks_list = raw_result
                if boxes is not None and len(boxes) > 0:
                    valid_idx = None
                    for i, box in enumerate(boxes):
                        if len(box) >= 5 and box[4] >= self.conf_threshold:
                            valid_idx = i
                            break
                    if valid_idx is not None:
                        selected_landmarks = landmarks_list[valid_idx]

            # --- Se não encontrou nada ---
            if selected_landmarks is None:
                return None

            # --- Normalização ---
            lmks = np.array(selected_landmarks)
            h, w = img.shape[:2]
            
            normalized = []
            for point in lmks:
                normalized.append([
                    float(point[0] / w),
                    float(point[1] / h)
                ])
            
            return normalized
            
        except Exception as e:
            if self.stats['failed'] == 0: 
                print(f"CRITICAL ERROR ON IMAGE {image_path}: {str(e)}")
                traceback.print_exc()
            return None
    
    def annotate_dataset(
        self,
        dataset_root: str,
        dataset_name: str,
        force_reannotate: bool = False
    ) -> Dict[str, List[List[float]]]:
        cache_path = self._get_cache_path(dataset_name)
        
        # Tenta carregar do cache
        if cache_path.exists() and not force_reannotate:
            LOGGER.info(f"Loading landmarks from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                self.stats = cached_data.get('stats', self.stats)
                landmarks_dict = cached_data.get('landmarks', {})
                self._log_stats()
                return landmarks_dict
            except json.JSONDecodeError:
                LOGGER.warning("Cache corrupted. Reannotating...")
        
        self._init_detector()
        
        LOGGER.info(f"Scanning dataset: {dataset_root}")
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        self.stats['total_images'] = len(image_paths)
        LOGGER.info(f"Found {len(image_paths)} images to process")
        
        landmarks_dict = {}
        self.stats['annotated'] = 0
        self.stats['failed'] = 0
        self.stats['failed_paths'] = []
        
        for img_path in tqdm(image_paths, desc="Extracting landmarks"):
            landmarks = self._extract_landmarks(img_path)
            if landmarks is not None:
                rel_path = os.path.relpath(img_path, dataset_root)
                landmarks_dict[rel_path] = landmarks
                self.stats['annotated'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['failed_paths'].append(img_path)
        
        cache_data = {
            'stats': self.stats,
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
        """Loga estatísticas de anotação com formatação visual"""
        total = self.stats['total_images']
        annotated = self.stats['annotated']
        failed = self.stats['failed']
        
        success_rate = (annotated / total * 100) if total > 0 else 0
        fail_rate = (failed / total * 100) if total > 0 else 0
        
        LOGGER.info("\n" + "="*70)
        LOGGER.info("LANDMARK DETECTION SUMMARY (Train Dataset)")
        LOGGER.info("="*70)
        LOGGER.info(f"Total Images:   {total}")
        LOGGER.info(f"✅ Annotated:    {annotated} ({success_rate:.2f}%)")
        LOGGER.info(f"❌ Failed:       {failed} ({fail_rate:.2f}%)")
        LOGGER.info("="*70 + "\n")
        
        if failed > 0:
            LOGGER.warning(f"⚠️  {failed} images will be EXCLUDED from training due to detection failure.")

    def get_valid_samples(self, samples, landmarks_dict, dataset_root):
        valid, excluded = [], []
        for img_path, label in samples:
            rel_path = os.path.relpath(img_path, dataset_root)
            if rel_path in landmarks_dict:
                valid.append((img_path, label))
            else:
                excluded.append((img_path, label))
        return valid, excluded
    
    def get_landmarks_for_image(self, image_path, landmarks_dict, dataset_root):
        rel_path = os.path.relpath(image_path, dataset_root)
        if rel_path in landmarks_dict:
            return np.array(landmarks_dict[rel_path], dtype=np.float32)
        return None


def extract_landmarks_single_image(image, detector=None, conf_threshold=0.5):
    """
    Função helper para extração em imagem única (usada em inferência).
    Também robusta quanto a versão da lib e GPU.
    """
    if detector is None:
        try:
            if torch.cuda.is_available():
                detector = RetinaFace(gpu_id=torch.cuda.current_device())
            else:
                detector = RetinaFace()
        except TypeError:
            detector = RetinaFace()

    try:
        raw_result = detector.detect(image)
        if not raw_result: return None
        
        selected_landmarks = None
        
        # Lógica híbrida simplificada
        if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
            best = max(raw_result, key=lambda x: x.get('confidence', x.get('score', 0)))
            if best.get('confidence', best.get('score', 0)) >= conf_threshold:
                selected_landmarks = best.get('landmarks', best.get('kps'))
        elif isinstance(raw_result, tuple) and len(raw_result) == 2:
            boxes, landmarks_list = raw_result
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if len(box) >= 5 and box[4] >= conf_threshold:
                        selected_landmarks = landmarks_list[i]
                        break

        if selected_landmarks is None: return None
        lmks = np.array(selected_landmarks)
        h, w = image.shape[:2]
        return np.array([[lmks[i][0]/w, lmks[i][1]/h] for i in range(5)], dtype=np.float32)
    except Exception: return None