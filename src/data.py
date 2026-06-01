import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

class JointTransform:
    """
    Transformación conjunta para imágenes y sus correspondientes máscaras de segmentación.
    Redimensiona ambas imágenes y convierte la máscara a un tensor de clases {0, 1, 2}.
    """
    def __init__(self, size=(128, 128)):
        self.size = size

    def __call__(self, img, target):
        # Redimensionado de la imagen (bilineal) y de la máscara (vecino más cercano)
        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, self.size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Convertir imagen a tensor y normalizar con la media y desviación estándar de ImageNet
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # En Oxford-IIIT Pet, la máscara tiene valores {1, 2, 3}:
        # 1: Objeto de interés (mascota), 2: Fondo, 3: Contorno.
        # Restamos 1 para mapearlo a clases 0-indexadas {0, 1, 2}
        target_np = np.array(target)
        target_np = np.clip(target_np - 1, 0, 2)
        target = torch.as_tensor(target_np, dtype=torch.long)
        
        return img, target

def get_dataloaders(data_dir='./data', batch_size=16, size=(128, 128), train_subset_size=320, val_subset_size=160):
    """
    Descarga el dataset Oxford-IIIT Pet y crea los DataLoaders para entrenamiento y validación.
    Para acelerar las iteraciones del optimizador bayesiano, se seleccionan subconjuntos reducidos.
    
    Parámetros:
    -----------
    data_dir : str
        Carpeta donde descargar y almacenar los datos.
    batch_size : int
        Tamaño del lote de entrenamiento.
    size : tuple (int, int)
        Tamaño de redimensionado de las imágenes.
    train_subset_size : int
        Número de muestras de entrenamiento a utilizar (subconjunto).
    val_subset_size : int
        Número de muestras de validación a utilizar (subconjunto).
        
    Retorna:
    --------
    train_loader, val_loader : DataLoader, DataLoader
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = JointTransform(size=size)
    
    # Descargar y cargar sets trainval y test de Oxford-IIIT Pet
    # trainval se usará para entrenamiento, test para validación
    train_dataset = OxfordIIITPet(
        root=data_dir,
        split='trainval',
        target_types='segmentation',
        download=True,
        transforms=transform
    )
    
    val_dataset = OxfordIIITPet(
        root=data_dir,
        split='test',
        target_types='segmentation',
        download=True,
        transforms=transform
    )
    
    # Tomar subconjuntos de datos de forma aleatoria fija para reproducibilidad
    rng = np.random.default_rng(42)
    
    if train_subset_size < len(train_dataset):
        train_indices = rng.choice(len(train_dataset), size=train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        
    if val_subset_size < len(val_dataset):
        val_indices = rng.choice(len(val_dataset), size=val_subset_size, replace=False)
        val_dataset = Subset(val_dataset, val_indices)
        
    # Crear los DataLoaders (con num_workers=0 por compatibilidad y estabilidad en Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
