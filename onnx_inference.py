import cv2
import uniface
import numpy as np
from pathlib import Path

from models import ONNXFaceEngine
from utils.face_utils import compute_similarity

import warnings
warnings.filterwarnings("ignore")


def compare_faces(
        model: ONNXFaceEngine,
        img1: np.ndarray,
        landmarks1: np.ndarray,
        img2: np.ndarray,
        landmarks2: np.ndarray,
        threshold: float = 0.30
) -> tuple:
    """
    Compares two face images and determines if they belong to the same person.

    Args:
        model (ONNXFaceEngine): The face recognition model instance.
        img1 (np.ndarray): First face image (BGR format).
        landmarks1 (np.ndarray): Facial landmarks for img1, shape (5, 2).
                                 Order: left_eye, right_eye, nose, left_mouth, right_mouth.
        img2 (np.ndarray): Second face image (BGR format).
        landmarks2 (np.ndarray): Facial landmarks for img2, shape (5, 2).
        threshold (float): Similarity threshold for face matching. Default: 0.30.

    Returns:
        tuple[float, bool]: Similarity score and match result (True/False).
    """
    feat1 = model.get_embedding(img1, landmarks1)
    feat2 = model.get_embedding(img2, landmarks2)
    similarity = compute_similarity(feat1, feat2)
    is_match = similarity > threshold
    return similarity, is_match


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk with validation.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        np.ndarray: Loaded image in BGR format.
        
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be loaded.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return img


def detect_face_landmarks(detector, image: np.ndarray, image_name: str = "image") -> np.ndarray:
    """
    Detect face landmarks in an image with validation.
    
    Args:
        detector: UniFace detector instance.
        image (np.ndarray): Input image in BGR format.
        image_name (str): Name of the image for error messages.
        
    Returns:
        np.ndarray: Facial landmarks, shape (5, 2).
        
    Raises:
        ValueError: If no face or multiple faces are detected.
    """
    boxes, landmarks = detector.detect(image)
    
    if len(landmarks) == 0:
        raise ValueError(f"No face detected in {image_name}")
    
    if len(landmarks) > 1:
        print(f"⚠️  Warning: Multiple faces detected in {image_name}, using the first one")
    
    return landmarks[0]


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("ONNX Face Recognition - Inference Example")
    print("="*70)
    
    # Configuration
    model_path = "weights/mobilenetv2_mcp.onnx"  # Adjust path as needed
    img1_path = "assets/b_01.jpg"
    img2_path = "assets/b_02.jpg"
    threshold = 0.30
    
    try:
        # Initialize face detector
        print("\n📦 Initializing face detector...")
        uniface_inference = uniface.RetinaFace(
            model="retinaface_mnet_v2", 
            conf_thresh=0.45
        )
        print("✓ Face detector initialized")
        
        # Initialize face recognition model
        print(f"\n📦 Loading ONNX model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        face_recognizer = ONNXFaceEngine(model_path)
        print("✓ ONNX model loaded successfully")
        
        # Load images
        print(f"\n📂 Loading images...")
        img1 = load_image(img1_path)
        print(f"✓ Loaded: {img1_path}")
        
        img2 = load_image(img2_path)
        print(f"✓ Loaded: {img2_path}")
        
        # Detect faces and landmarks
        print(f"\n🔍 Detecting faces...")
        landmarks1 = detect_face_landmarks(uniface_inference, img1, img1_path)
        print(f"✓ Face detected in {img1_path}")
        
        landmarks2 = detect_face_landmarks(uniface_inference, img2, img2_path)
        print(f"✓ Face detected in {img2_path}")
        
        # Compare faces
        print(f"\n🔄 Comparing faces (threshold={threshold})...")
        similarity, is_same = compare_faces(
            face_recognizer, 
            img1, landmarks1, 
            img2, landmarks2, 
            threshold=threshold
        )
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Threshold:        {threshold:.4f}")
        print(f"Match Result:     {'✓ SAME PERSON' if is_same else '✗ DIFFERENT PERSON'}")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease check that the following files exist:")
        print(f"  - ONNX model: {model_path}")
        print(f"  - Image 1:    {img1_path}")
        print(f"  - Image 2:    {img2_path}")
        
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPossible causes:")
        print("  - Image file is corrupted")
        print("  - No face detected in the image")
        print("  - Face is too small or poorly lit")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()