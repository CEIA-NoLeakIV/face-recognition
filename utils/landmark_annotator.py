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
        """Inicializa detector na GPU ativa se disponível"""
        if self.detector is None:
            LOGGER.info("Initializing RetinaFace detector...")
            try:
                # Tenta usar a GPU atual do PyTorch (respeita o docker run --gpus)
                if torch.cuda.is_available():
                    current_gpu = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_gpu)
                    LOGGER.info(f"Attempting to load RetinaFace on GPU ID {current_gpu}: {gpu_name}")
                    
                    # Usa o ID dinâmico da GPU atual
                    self.detector = RetinaFace(gpu_id=current_gpu)
                else:
                    self.detector = RetinaFace()
            except Exception as e:
                LOGGER.warning(f"Could not force GPU ({e}). Falling back to default initialization.")
                self.detector = RetinaFace()
            
            LOGGER.info("RetinaFace detector initialized")
    
    def _get_cache_path(self, dataset_name: str) -> Path:
        """Retorna caminho do arquivo de cache para o dataset"""
        return self.cache_dir / f"{dataset_name}_landmarks.json"
    
    def _extract_landmarks(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Extrai landmarks de uma imagem lidando com diferentes versões da lib.
        """
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
            
            # --- PONTO CRÍTICO: Captura o resultado sem desempacotar ---
            raw_result = self.detector.detect(img)
            
            if not raw_result:
                return None

            selected_landmarks = None

            # --- CASO 1: Versão Nova (Lista de Dicionários) ---
            # Ex: [{'bbox': [...], 'confidence': 0.99, 'landmarks': [...]}]
            if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
                
                # Encontrar a melhor face baseada na confiança
                best_face = None
                best_conf = -1.0
                
                for face in raw_result:
                    # Algumas versões usam 'score', outras 'confidence'
                    conf = face.get('confidence', face.get('score', 0.0))
                    
                    if conf >= self.conf_threshold and conf > best_conf:
                        best_conf = conf
                        best_face = face
                
                if best_face is not None:
                    # Algumas versões usam 'landmarks', outras 'kps'
                    selected_landmarks = best_face.get('landmarks', best_face.get('kps'))

            # --- CASO 2: Versão Antiga (Tupla de Arrays) ---
            # Ex: (boxes, landmarks)
            elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                boxes, landmarks_list = raw_result
                
                if boxes is not None and len(boxes) > 0:
                    valid_idx = None
                    # boxes formato: [x1, y1, x2, y2, score]
                    for i, box in enumerate(boxes):
                        if len(box) >= 5 and box[4] >= self.conf_threshold:
                            valid_idx = i
                            break
                    
                    if valid_idx is not None:
                        selected_landmarks = landmarks_list[valid_idx]

            # --- Se não encontrou nada em nenhum formato ---
            if selected_landmarks is None:
                return None

            # --- Normalização e Retorno ---
            # Garante que é numpy array
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
            # Imprime erro detalhado apenas na primeira falha para debug
            if self.stats['failed'] == 0: 
                print(f"CRITICAL ERROR ON IMAGE {image_path}: {str(e)}")
                print(f"Retorno do detector (tipo): {type(raw_result) if 'raw_result' in locals() else 'Unknown'}")
                if 'raw_result' in locals():
                    print(f"Retorno (conteúdo): {raw_result}")
                traceback.print_exc()
            return None
    
    def annotate_dataset(
        self,
        dataset_root: str,
        dataset_name: str,
        force_reannotate: bool = False
    ) -> Dict[str, List[List[float]]]:
        """
        Anota todas as imagens de um dataset.
        """
        cache_path = self._get_cache_path(dataset_name)
        
        # Tenta carregar do cache
        if cache_path.exists() and not force_reannotate:
            LOGGER.info(f"Loading landmarks from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                self.stats = cached_data.get('stats', self.stats)
                landmarks_dict = cached_data.get('landmarks', {})
                
                # Sanity check: se o cache está vazio ou corrompido, força reanotação
                if len(landmarks_dict) == 0 and self.stats.get('total_images', 0) > 0:
                    LOGGER.warning("Cache seems empty/failed. Forcing reannotation...")
                else:
                    self._log_stats()
                    return landmarks_dict
            except json.JSONDecodeError:
                LOGGER.warning("Cache corrupted. Reannotating...")
        
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
    Versão Híbrida (Tupla ou Dict).
    """
    if detector is None:
        # Tenta usar GPU ativa se disponível ao criar nova instância
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
        else:
            gpu_id = -1
            
        detector = RetinaFace(gpu_id=gpu_id)
    
    try:
        raw_result = detector.detect(image)
        
        if not raw_result:
            return None
        
        selected_landmarks = None

        # --- Lógica Híbrida (Igual à classe acima) ---
        
        # Caso Lista de Dicts (Novo)
        if isinstance(raw_result, list) and len(raw_result) > 0 and isinstance(raw_result[0], dict):
            best_conf = -1.0
            for face in raw_result:
                conf = face.get('confidence', face.get('score', 0.0))
                if conf >= conf_threshold and conf > best_conf:
                    best_conf = conf
                    selected_landmarks = face.get('landmarks', face.get('kps'))
        
        # Caso Tupla (Antigo)
        elif isinstance(raw_result, tuple) and len(raw_result) == 2:
            boxes, landmarks_list = raw_result
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if len(box) >= 5 and box[4] >= conf_threshold:
                        selected_landmarks = landmarks_list[i]
                        break # Pega o primeiro válido

        if selected_landmarks is None:
            return None
        
        # Normaliza
        lmks = np.array(selected_landmarks)
        h, w = image.shape[:2]
        normalized = np.array([
            [lmks[i][0] / w, lmks[i][1] / h]
            for i in range(5)
        ], dtype=np.float32)
        
        return normalized
        
    except Exception:
        return None