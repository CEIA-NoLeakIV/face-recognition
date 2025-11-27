import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.landmark_annotator import LandmarkAnnotator
from utils.general import LOGGER


class ImageFolder(Dataset):
    """ImageFolder Dataset for loading images organized in a directory structure.

    Args:
        root (str): Root directory containing class subdirectories.
        transform (callable, optional): A function/transform to apply to the images.
        use_landmarks (bool): Se True, retorna landmarks junto com imagem e label.
        landmarks_dict (dict, optional): Dict pré-carregado de landmarks.
        landmark_cache_dir (str): Diretório para cache de landmarks.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        use_landmarks: bool = False,
        landmarks_dict: Optional[Dict[str, List[List[float]]]] = None,
        landmark_cache_dir: str = "landmark_cache"
    ) -> None:
        self.root = root
        self.transform = transform
        self.use_landmarks = use_landmarks
        self.landmarks_dict = landmarks_dict
        
        # Cria dataset inicial
        all_samples = self._make_dataset(root)
        
        if use_landmarks:
            if landmarks_dict is None:
                # Anota landmarks automaticamente
                annotator = LandmarkAnnotator(cache_dir=landmark_cache_dir)
                dataset_name = os.path.basename(root.rstrip('/'))
                self.landmarks_dict = annotator.annotate_dataset(root, dataset_name)
            
            # Filtra apenas samples com landmarks válidos
            self.samples, excluded = self._filter_samples_with_landmarks(
                all_samples, self.landmarks_dict, root
            )
            
            LOGGER.info(f"Dataset loaded: {len(self.samples)} samples with landmarks")
            if len(excluded) > 0:
                LOGGER.warning(f"Excluded {len(excluded)} samples without landmarks")
        else:
            self.samples = all_samples

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._load_image(path)

        if self.transform:
            image = self.transform(image)

        if self.use_landmarks:
            landmarks = self._get_landmarks(path)
            return image, landmarks, label

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    def _get_landmarks(self, image_path: str) -> torch.Tensor:
        """Obtém landmarks para uma imagem."""
        rel_path = os.path.relpath(image_path, self.root)
        
        if rel_path in self.landmarks_dict:
            landmarks = np.array(self.landmarks_dict[rel_path], dtype=np.float32)
        else:
            # Fallback: landmarks zerados (não deveria acontecer após filtro)
            landmarks = np.zeros((5, 2), dtype=np.float32)
        
        return torch.from_numpy(landmarks)

    @staticmethod
    def _filter_samples_with_landmarks(
        samples: List[Tuple[str, int]],
        landmarks_dict: Dict[str, List[List[float]]],
        dataset_root: str
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Filtra samples que possuem landmarks anotados."""
        valid = []
        excluded = []
        
        for img_path, label in samples:
            rel_path = os.path.relpath(img_path, dataset_root)
            
            if rel_path in landmarks_dict:
                valid.append((img_path, label))
            else:
                excluded.append((img_path, label))
        
        return valid, excluded

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Loads an image from the given path."""
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    @staticmethod
    def _make_dataset(directory: str):
        """Creates a dataset of image paths and corresponding labels."""
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        instances = []
        for class_name, class_index in class_to_idx.items():
            class_dir = os.path.join(directory, class_name)

            for root, _, file_names in os.walk(class_dir, followlinks=True):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    if os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png"}:
                        instances.append((path, class_index))

        return instances