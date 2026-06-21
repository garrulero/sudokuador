import torch.nn as nn
import torch.nn.functional as F

class LectorSudoku(nn.Module):
    """
    CNN mejorada para reconocimiento de dígitos de Sudoku (0-9, donde 0 = vacío).
    
    Mejoras respecto a la v1:
      - Más filtros: 32 y 64 (antes 16 y 32) → detecta más patrones
      - BatchNorm: estabiliza y acelera el entrenamiento
      - Dropout: previene overfitting → generaliza mejor a fotos reales
      - FC más amplia: 256 neuronas (antes 128) → más capacidad de decisión
    """
    def __init__(self):
        super(LectorSudoku, self).__init__()
        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pooling y regularización
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        # Capas fully-connected
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28x28 -> 14x14
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14x14 -> 7x7
        x = self.dropout_conv(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x