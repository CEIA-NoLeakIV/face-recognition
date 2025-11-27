import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LandmarkEncoder(nn.Module):
    """
    Codifica landmarks faciais (5 pontos) em um vetor de features.
    
    Input: (batch, 5, 2) - 5 pontos normalizados [0,1]
    Output: (batch, landmark_dim)
    """
    
    def __init__(
        self,
        num_landmarks: int = 5,
        landmark_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        input_dim = num_landmarks * 2  # 5 pontos x 2 coordenadas = 10
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(num_parameters=64),
            nn.Dropout(dropout),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout),
            
            nn.Linear(128, landmark_dim),
            nn.BatchNorm1d(landmark_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            landmarks: (batch, 5, 2) ou (batch, 10) - landmarks normalizados
            
        Returns:
            (batch, landmark_dim) - embedding dos landmarks
        """
        # Flatten se necessário
        if landmarks.dim() == 3:
            landmarks = landmarks.view(landmarks.size(0), -1)
        
        return self.encoder(landmarks)


class FeatureFusion(nn.Module):
    """
    Funde features visuais com features de landmarks.
    
    Estratégia: Concatenação + MLP para projetar de volta à dimensão original
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        landmark_dim: int = 128,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        fused_dim = visual_dim + landmark_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.PReLU(num_parameters=fused_dim),
            nn.Dropout(dropout),
            
            nn.Linear(fused_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        landmark_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, visual_dim) - embedding do backbone
            landmark_features: (batch, landmark_dim) - embedding dos landmarks
            
        Returns:
            (batch, output_dim) - embedding fundido
        """
        # Concatena
        fused = torch.cat([visual_features, landmark_features], dim=1)
        
        # Projeta
        return self.fusion(fused)


class LandmarkConditionedModel(nn.Module):
    """
    Wrapper que adiciona condicionamento por landmarks a qualquer backbone.
    
    Arquitetura:
        Imagem → Backbone → Visual Embedding (512d)
                                    ↓
        Landmarks → LandmarkEncoder → Landmark Embedding (128d)
                                    ↓
                            FeatureFusion
                                    ↓
                        Final Embedding (512d)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 512,
        num_landmarks: int = 5,
        landmark_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            backbone: Modelo base (sphere20, mobilenetv2, etc.)
            embedding_dim: Dimensão do embedding visual (saída do backbone)
            num_landmarks: Número de pontos de landmarks (padrão: 5)
            landmark_dim: Dimensão do embedding de landmarks
            dropout: Taxa de dropout
        """
        super().__init__()
        
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.num_landmarks = num_landmarks
        
        # Codificador de landmarks
        self.landmark_encoder = LandmarkEncoder(
            num_landmarks=num_landmarks,
            landmark_dim=landmark_dim,
            dropout=dropout
        )
        
        # Módulo de fusão
        self.fusion = FeatureFusion(
            visual_dim=embedding_dim,
            landmark_dim=landmark_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) - imagens
            landmarks: (batch, 5, 2) ou (batch, 10) - landmarks normalizados [0,1]
            
        Returns:
            (batch, embedding_dim) - embedding condicionado
        """
        # Extrai features visuais
        visual_embedding = self.backbone(x)
        
        # Codifica landmarks
        landmark_embedding = self.landmark_encoder(landmarks)
        
        # Funde
        fused_embedding = self.fusion(visual_embedding, landmark_embedding)
        
        return fused_embedding
    
    def forward_visual_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass usando apenas a imagem (sem landmarks).
        Útil para comparação ou fallback.
        
        Args:
            x: (batch, 3, H, W) - imagens
            
        Returns:
            (batch, embedding_dim) - embedding visual apenas
        """
        return self.backbone(x)
    
    def get_visual_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias para forward_visual_only"""
        return self.forward_visual_only(x)


def create_landmark_conditioned_model(
    network_name: str,
    embedding_dim: int = 512,
    num_landmarks: int = 5,
    landmark_dim: int = 128,
    dropout: float = 0.1,
    **backbone_kwargs
) -> LandmarkConditionedModel:
    """
    Factory function para criar modelo condicionado por landmarks.
    
    Args:
        network_name: Nome do backbone ('sphere20', 'mobilenetv2', etc.)
        embedding_dim: Dimensão do embedding
        num_landmarks: Número de landmarks
        landmark_dim: Dimensão do embedding de landmarks
        dropout: Taxa de dropout
        **backbone_kwargs: Argumentos adicionais para o backbone
        
    Returns:
        LandmarkConditionedModel configurado
    """
    from models import (
        sphere20, sphere36, sphere64,
        MobileNetV1, MobileNetV2,
        mobilenet_v3_small, mobilenet_v3_large
    )
    
    networks = {
        'sphere20': sphere20,
        'sphere36': sphere36,
        'sphere64': sphere64,
        'mobilenetv1': MobileNetV1,
        'mobilenetv2': MobileNetV2,
        'mobilenetv3_small': mobilenet_v3_small,
        'mobilenetv3_large': mobilenet_v3_large
    }
    
    if network_name not in networks:
        raise ValueError(f"Unknown network: {network_name}. Available: {list(networks.keys())}")
    
    # Cria backbone
    backbone = networks[network_name](embedding_dim=embedding_dim, **backbone_kwargs)
    
    # Wrapa com condicionamento de landmarks
    model = LandmarkConditionedModel(
        backbone=backbone,
        embedding_dim=embedding_dim,
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim,
        dropout=dropout
    )
    
    return model


def load_landmark_conditioned_model(
    checkpoint_path: str,
    network_name: str,
    device: torch.device = None,
    embedding_dim: int = 512,
    num_landmarks: int = 5,
    landmark_dim: int = 128
) -> LandmarkConditionedModel:
    """
    Carrega modelo condicionado de um checkpoint.
    
    Args:
        checkpoint_path: Caminho do checkpoint
        network_name: Nome do backbone
        device: Device para carregar o modelo
        embedding_dim: Dimensão do embedding
        num_landmarks: Número de landmarks
        landmark_dim: Dimensão do embedding de landmarks
        
    Returns:
        Modelo carregado
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cria modelo
    model = create_landmark_conditioned_model(
        network_name=network_name,
        embedding_dim=embedding_dim,
        num_landmarks=num_landmarks,
        landmark_dim=landmark_dim
    )
    
    # Carrega pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model