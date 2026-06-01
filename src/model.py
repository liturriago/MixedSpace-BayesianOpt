import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Bloque de doble convolución: (Conv2d -> BatchNorm2d -> ReLU) x 2.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    """
    U-Net ligera optimizada para ejecución rápida en CPU/GPU sin perder la estructura
    de conexiones de salto (skip connections) de la arquitectura clásica.
    """
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        # Encoder (reducción)
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(16, 32)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )
        
        # Decoder (expansión)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)  # 32 (up) + 32 (skip) = 64
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(32, 16)  # 16 (up) + 16 (skip) = 32
        
        # Capa final de proyección a clases
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Conexiones de salto en la etapa de decodificación
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase.
    Mapea el hiperparámetro continuo 'alpha_focal' en un vector de ponderación balanceado.
    """
    def __init__(self, alpha_focal, gamma=2.0):
        super().__init__()
        self.alpha_focal = alpha_focal
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W)
        # targets: (B, H, W)
        
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)
        
        # Obtener log_p y p correspondientes a la clase real (target) en cada píxel
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Definición de pesos para las 3 clases:
        # Clase 0 (Mascota) recibe el peso 'alpha_focal'
        # Clase 1 (Fondo) y Clase 2 (Contorno) se reparten equitativamente el peso restante: (1 - alpha_focal) / 2
        weights = torch.tensor(
            [self.alpha_focal, (1.0 - self.alpha_focal) / 2.0, (1.0 - self.alpha_focal) / 2.0],
            device=inputs.device,
            dtype=inputs.dtype
        )
        
        # Extraer el peso correspondiente a cada píxel según su target real
        weight_t = weights[targets]
        
        # Ecuación de Focal Loss: FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = - weight_t * ((1.0 - p_t) ** self.gamma) * log_p_t
        return loss.mean()

def compute_dice_score(preds, targets, num_classes=3):
    """
    Calcula el coeficiente Dice Score promedio (macro Dice) sobre las 3 clases.
    """
    dice_scores = []
    for c in range(num_classes):
        p_c = (preds == c).float()
        t_c = (targets == c).float()
        
        intersection = (p_c * t_c).sum()
        total = p_c.sum() + t_c.sum()
        
        if total == 0:
            # Si la clase no está presente ni en predicciones ni en targets, su concordancia es perfecta
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection / total).item())
            
    return np.mean(dice_scores)

def evaluate_pipeline(learning_rate, alpha_focal, optimizer_type, train_loader, val_loader, device='cpu'):
    """
    Pipeline de evaluación de caja negra para la optimización bayesiana.
    Entrena la U-Net por 1 época y mide el Dice Score de validación y el tiempo de ejecución.
    
    Parámetros:
    -----------
    learning_rate : float
        Tasa de aprendizaje continua.
    alpha_focal : float
        Hiperparámetro de balance de Focal Loss.
    optimizer_type : float
        Índice relajado del optimizador a mapear en {0: Adam, 1: RMSprop, 2: SGD_Momentum}.
    train_loader, val_loader : DataLoader
    device : str
    
    Retorna:
    --------
    val_dice : float
        Dice Score de validación obtenido (métrica a maximizar).
    epoch_time : float
        Tiempo en segundos que tardó en procesarse la época (métrica de restricción).
    """
    # 1. Instanciar el modelo ligero y la función de pérdida
    model = SimpleUNet(in_channels=3, num_classes=3).to(device)
    criterion = FocalLoss(alpha_focal=alpha_focal, gamma=2.0)
    
    # 2. Mapear y configurar el optimizador según la relajación indexada
    optimizer_idx = int(np.clip(np.round(optimizer_type), 0, 2))
    if optimizer_idx == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_idx == 1:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
    # 3. Entrenamiento (1 Época) y medición de tiempo
    model.train()
    start_time = time.time()
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
    epoch_time = time.time() - start_time
    
    # 4. Evaluación en el conjunto de validación
    model.eval()
    all_dices = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            dice = compute_dice_score(preds, masks, num_classes=3)
            all_dices.append(dice)
            
    val_dice = np.mean(all_dices)
    
    return float(val_dice), float(epoch_time)
