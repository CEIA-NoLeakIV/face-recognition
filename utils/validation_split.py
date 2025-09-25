import torch
import random
from torch.utils.data import Subset

def create_validation_split(dataset, val_split=0.1, random_seed=42):
    
    # Configurar seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Organizar samples por classe
    class_to_indices = {}
    for idx, (_, target) in enumerate(dataset.samples):
        path, label = dataset.samples[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # Split estratificado por classe
    for class_id, indices in class_to_indices.items():
        if len(indices) < 2:  # Pular classes com poucos samples
            train_indices.extend(indices)
            continue
            
        # Embaralhar indices da classe
        random.shuffle(indices)
        
        # Calcular split
        val_size = max(1, int(len(indices) * val_split))
        train_size = len(indices) - val_size
        
        # Dividir
        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:])
    
    # Criar subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset