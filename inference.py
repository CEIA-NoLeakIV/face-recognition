# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import numpy as np
from PIL import Image
import os
import json
from pathlib import Path

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

from utils.face_utils import compute_similarity


def get_network(model_name: str) -> torch.nn.Module:
    """
    Returns the appropriate model based on the provided model name.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        torch.nn.Module: The selected deep learning model.
    """
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


def load_model(model_name: str, model_path: str, device: torch.device = None) -> torch.nn.Module:
    """
    Loads a deep learning model with pre-trained weights.
    """
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model' in checkpoint:
            if 'args' in checkpoint:
                args = checkpoint['args']
                embedding_dim = getattr(args, 'embedding_dim', 512)
            else:
                embedding_dim = 512
            
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
            # S�o apenas pesos
            model = get_network(model_name)
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


def extract_features(model, device, img_path: str) -> np.ndarray:
    """
    Extracts face features from an image.
    """
    transform = get_transform()

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Error opening image {img_path}: {e}")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor).squeeze().cpu().numpy()
    return features

def extract_batch_embeddings(model, device, image_folder: str, output_file: str = "assets/embeddings.json") -> dict:
    """
    Extracts embeddings from all images in a folder and saves to JSON file.
    
    Args:
        model: Loaded deep learning model
        device: torch device (cpu/cuda)
        image_folder: Path to folder containing images
        output_file: Path to save embeddings JSON file
    
    Returns:
        dict: Dictionary with image names as keys and embeddings as values
    """

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
    
    embeddings_dict = {}
    
    print(f"Extracting embeddings from {len(image_files)} images...")
    
    for img_path in image_files:
        try:
            embedding = extract_features(model, device, str(img_path))
            embeddings_dict[img_path.name] = embedding.tolist()
            print(f"✓ Processed: {img_path.name}")
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    print(f"\nEmbeddings saved to: {output_file}")

    return embeddings_dict


def compare_faces(model, device, img1_path: str, img2_path: str, threshold: float = 0.35) -> tuple[float, bool]:
    """
    Compares two face images and determines if they belong to the same person.
    """
    feat1 = extract_features(model, device, img1_path)
    feat2 = extract_features(model, device, img2_path)

    similarity = compute_similarity(feat1, feat2)
    is_same = similarity > threshold

    return similarity, is_same


if __name__ == "__main__":
    # Example usage with model selection
    model_name = "mobilenetv2"
    model_path = "weights/mobilenetv2_mcp.pth"
    threshold = 0.35

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_name, model_path, device)

    # Compare faces
    similarity, is_same = compare_faces(
        model, device,
        img1_path="assets/b_01.jpg",
        img2_path="assets/b_02.jpg",
        threshold=threshold
    )

    print(f"Similarity: {similarity:.4f} - {'same' if is_same else 'different'} (Threshold: {threshold})")
