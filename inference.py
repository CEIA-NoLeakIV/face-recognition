import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torchvision import transforms

from uniface import RetinaFace

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)
from models.landmark_conditioned import create_landmark_conditioned_model, load_landmark_conditioned_model

from utils.face_utils import compute_similarity
from utils.landmark_annotator import extract_landmarks_single_image


def get_network(model_name: str, use_landmarks: bool = False, landmark_dim: int = 128) -> torch.nn.Module:
    """
    Returns the appropriate model based on the provided model name.

    Args:
        model_name (str): Name of the model architecture.
        use_landmarks (bool): Whether to create landmark-conditioned model.
        landmark_dim (int): Dimension of landmark embedding.

    Returns:
        torch.nn.Module: The selected deep learning model.
    """
    if use_landmarks:
        return create_landmark_conditioned_model(
            network_name=model_name,
            embedding_dim=512,
            num_landmarks=5,
            landmark_dim=landmark_dim
        )
    
    models = {
        "sphere20": sphere20(embedding_dim=512, in_channels=3),
        "sphere36": sphere36(embedding_dim=512, in_channels=3),
        "sphere64": sphere64(embedding_dim=512, in_channels=3),
        "mobilenetv1": MobileNetV1(embedding_dim=512),
        "mobilenetv2": MobileNetV2(embedding_dim=512),
        "mobilenetv3_small": mobilenet_v3_small(embedding_dim=512),
        "mobilenetv3_large": mobilenet_v3_large(embedding_dim=512),
    }

    if model_name not in models:
        raise ValueError(f"Unsupported network '{model_name}'! Available options: {list(models.keys())}")

    return models[model_name]


def load_model(
    model_name: str,
    model_path: str,
    device: torch.device = None,
    use_landmarks: bool = None,
    landmark_dim: int = 128
) -> torch.nn.Module:
    """
    Loads a deep learning model with pre-trained weights.
    
    Args:
        model_name: Name of the model architecture.
        model_path: Path to the checkpoint file.
        device: Device to load model on.
        use_landmarks: Whether model uses landmarks. If None, auto-detect from checkpoint.
        landmark_dim: Dimension of landmark embedding.
        
    Returns:
        Loaded model.
    """
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Auto-detect landmark mode from checkpoint
        if use_landmarks is None:
            use_landmarks = checkpoint.get('use_landmarks', False)
        
        if 'model' in checkpoint:
            if 'args' in checkpoint:
                args = checkpoint['args']
                embedding_dim = getattr(args, 'embedding_dim', 512)
                # Tenta obter landmark_dim dos args salvos
                landmark_dim = getattr(args, 'landmark_dim', landmark_dim)
            else:
                embedding_dim = 512
            
            if use_landmarks:
                model = create_landmark_conditioned_model(
                    network_name=model_name,
                    embedding_dim=embedding_dim,
                    num_landmarks=5,
                    landmark_dim=landmark_dim
                )
            else:
                if model_name == "sphere20":
                    model = sphere20(embedding_dim=embedding_dim, in_channels=3)
                elif model_name == "sphere36":
                    model = sphere36(embedding_dim=embedding_dim, in_channels=3)
                elif model_name == "sphere64":
                    model = sphere64(embedding_dim=embedding_dim, in_channels=3)
                elif model_name == "mobilenetv1":
                    model = MobileNetV1(embedding_dim=embedding_dim)
                elif model_name == "mobilenetv2":
                    model = MobileNetV2(embedding_dim=embedding_dim)
                elif model_name == "mobilenetv3_small":
                    model = mobilenet_v3_small(embedding_dim=embedding_dim)
                elif model_name == "mobilenetv3_large":
                    model = mobilenet_v3_large(embedding_dim=embedding_dim)
                else:
                    raise ValueError(f"Unsupported network '{model_name}'")
            
            model.load_state_dict(checkpoint['model'])
        else:
            # São apenas pesos
            model = get_network(model_name, use_landmarks=use_landmarks, landmark_dim=landmark_dim)
            model.load_state_dict(checkpoint)
        
        model.to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_name}' from {model_path}: {e}")

    return model


def get_transform():
    """
    Returns the image preprocessing transformations.
    """
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def extract_features(
    model,
    device,
    img_path: str,
    use_landmarks: bool = False,
    landmarks: Optional[np.ndarray] = None,
    landmark_detector: Optional[RetinaFace] = None
) -> np.ndarray:
    """
    Extracts face features from an image.
    
    Args:
        model: The model to extract features.
        device: Device to run inference on.
        img_path: Path to the image.
        use_landmarks: Whether model uses landmarks.
        landmarks: Pre-computed landmarks (shape (5, 2) normalized).
        landmark_detector: RetinaFace detector for real-time extraction.
        
    Returns:
        np.ndarray: Feature vector.
    """
    transform = get_transform()

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Error opening image {img_path}: {e}")

    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_landmarks:
            # Obtém landmarks se não fornecidos
            if landmarks is None:
                img_np = np.array(img)
                landmarks = extract_landmarks_single_image(img_np, landmark_detector)
                
                if landmarks is None:
                    # Fallback: usa zeros se não detectar face
                    landmarks = np.zeros((5, 2), dtype=np.float32)
            
            landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)
            features = model(tensor, landmarks_tensor).squeeze().cpu().numpy()
        else:
            features = model(tensor).squeeze().cpu().numpy()
    
    return features


def extract_features_from_image(
    model,
    device,
    img: Image.Image,
    use_landmarks: bool = False,
    landmarks: Optional[np.ndarray] = None,
    landmark_detector: Optional[RetinaFace] = None
) -> np.ndarray:
    """
    Extracts face features from a PIL Image object.
    
    Args:
        model: The model to extract features.
        device: Device to run inference on.
        img: PIL Image object.
        use_landmarks: Whether model uses landmarks.
        landmarks: Pre-computed landmarks (shape (5, 2) normalized).
        landmark_detector: RetinaFace detector for real-time extraction.
        
    Returns:
        np.ndarray: Feature vector.
    """
    transform = get_transform()
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_landmarks:
            if landmarks is None:
                img_np = np.array(img)
                landmarks = extract_landmarks_single_image(img_np, landmark_detector)
                
                if landmarks is None:
                    landmarks = np.zeros((5, 2), dtype=np.float32)
            
            landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)
            features = model(tensor, landmarks_tensor).squeeze().cpu().numpy()
        else:
            features = model(tensor).squeeze().cpu().numpy()
    
    return features


def extract_batch_embeddings(
    model,
    device,
    image_folder: str,
    output_file: str = "assets/embeddings.json",
    use_landmarks: bool = False,
    landmark_detector: Optional[RetinaFace] = None
) -> Dict[str, list]:
    """
    Extracts embeddings from all images in a folder and saves to JSON file.
    
    Args:
        model: Loaded deep learning model
        device: torch device (cpu/cuda)
        image_folder: Path to folder containing images
        output_file: Path to save embeddings JSON file
        use_landmarks: Whether model uses landmarks
        landmark_detector: RetinaFace detector for landmark extraction
    
    Returns:
        dict: Dictionary with image names as keys and embeddings as values
    """
    # Inicializa detector se necessário
    if use_landmarks and landmark_detector is None:
        try:
            landmark_detector = RetinaFace()
            print("RetinaFace detector initialized for landmark extraction")
        except Exception as e:
            print(f"Warning: Could not initialize RetinaFace: {e}")
            print("Landmarks will be zeros (fallback mode)")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
    
    embeddings_dict = {}
    
    print(f"Extracting embeddings from {len(image_files)} images...")
    if use_landmarks:
        print("Mode: Landmark-conditioned")
    
    for img_path in image_files:
        try:
            embedding = extract_features(
                model, device, str(img_path),
                use_landmarks=use_landmarks,
                landmark_detector=landmark_detector
            )
            embeddings_dict[img_path.name] = embedding.tolist()
            print(f"✓ Processed: {img_path.name}")
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    print(f"\nEmbeddings saved to: {output_file}")

    return embeddings_dict


def compare_faces(
    model,
    device,
    img1_path: str,
    img2_path: str,
    threshold: float = 0.35,
    use_landmarks: bool = False,
    landmark_detector: Optional[RetinaFace] = None
) -> Tuple[float, bool]:
    """
    Compares two face images and determines if they belong to the same person.
    
    Args:
        model: The model to use for comparison.
        device: Device to run inference on.
        img1_path: Path to first image.
        img2_path: Path to second image.
        threshold: Similarity threshold.
        use_landmarks: Whether model uses landmarks.
        landmark_detector: RetinaFace detector for landmark extraction.
        
    Returns:
        Tuple of (similarity score, is_same_person).
    """
    # Inicializa detector se necessário
    if use_landmarks and landmark_detector is None:
        try:
            landmark_detector = RetinaFace()
        except Exception as e:
            print(f"Warning: Could not initialize RetinaFace: {e}")
    
    feat1 = extract_features(
        model, device, img1_path,
        use_landmarks=use_landmarks,
        landmark_detector=landmark_detector
    )
    feat2 = extract_features(
        model, device, img2_path,
        use_landmarks=use_landmarks,
        landmark_detector=landmark_detector
    )

    similarity = compute_similarity(feat1, feat2)
    is_same = similarity > threshold

    return similarity, is_same


def compare_faces_with_landmarks(
    model,
    device,
    img1_path: str,
    landmarks1: np.ndarray,
    img2_path: str,
    landmarks2: np.ndarray,
    threshold: float = 0.35
) -> Tuple[float, bool]:
    """
    Compares two face images with pre-computed landmarks.
    
    Args:
        model: The landmark-conditioned model.
        device: Device to run inference on.
        img1_path: Path to first image.
        landmarks1: Landmarks for first image (shape (5, 2) normalized).
        img2_path: Path to second image.
        landmarks2: Landmarks for second image (shape (5, 2) normalized).
        threshold: Similarity threshold.
        
    Returns:
        Tuple of (similarity score, is_same_person).
    """
    feat1 = extract_features(
        model, device, img1_path,
        use_landmarks=True,
        landmarks=landmarks1
    )
    feat2 = extract_features(
        model, device, img2_path,
        use_landmarks=True,
        landmarks=landmarks2
    )

    similarity = compute_similarity(feat1, feat2)
    is_same = similarity > threshold

    return similarity, is_same


if __name__ == "__main__":
    # Example usage with model selection
    model_name = "mobilenetv2"
    model_path = "weights/mobilenetv2_mcp.pth"
    threshold = 0.35
    
    # Set to True if using landmark-conditioned model
    use_landmarks = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (auto-detects landmark mode from checkpoint)
    model = load_model(model_name, model_path, device)
    
    # Check if model uses landmarks
    if hasattr(model, 'landmark_encoder'):
        use_landmarks = True
        print("Loaded landmark-conditioned model")

    # Compare faces
    similarity, is_same = compare_faces(
        model, device,
        img1_path="assets/b_01.jpg",
        img2_path="assets/b_02.jpg",
        threshold=threshold,
        use_landmarks=use_landmarks
    )

    print(f"Similarity: {similarity:.4f} - {'same' if is_same else 'different'} (Threshold: {threshold})")