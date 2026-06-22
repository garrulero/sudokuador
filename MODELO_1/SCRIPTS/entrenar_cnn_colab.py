# =============================================================================
# SCRIPT DE ENTRENAMIENTO CNN v3 - PARA GOOGLE COLAB
# =============================================================================
# Copia y pega TODO este script en una celda de Google Colab.
# Usa Runtime > Change runtime type > T4 GPU para entrenar rapido.
#
# MEJORAS respecto a la v2:
#   1. Arquitectura mas potente: 32/64 filtros + BatchNorm + Dropout
#   2. Mejor augmentacion: blur, brillo/contraste, borrado aleatorio
#   3. 30 epocas con scheduler coseno (convergencia mas suave)
#   4. Mas datos sinteticos con variaciones de grosor y tamanho
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image, ImageFilter
import random
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# =============================================================================
# PASO 1: ARQUITECTURA MEJORADA
# =============================================================================
class LectorSudoku(nn.Module):
    def __init__(self):
        super(LectorSudoku, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# =============================================================================
# PASO 2: DATASETS
# =============================================================================

class MNISTInvertido(Dataset):
    """MNIST digitos 1-9, invertidos a oscuro-sobre-blanco."""
    def __init__(self, train=True, transform=None):
        mnist = datasets.MNIST(root='./data', train=train, download=True)
        mascara = mnist.targets > 0
        self.datos = mnist.data[mascara]
        self.etiquetas = mnist.targets[mascara]
        self.transform = transform

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        img = self.datos[idx].numpy()
        img = 255 - img
        img_pil = Image.fromarray(img, mode='L')
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, self.etiquetas[idx].item()


class CeldasVacias(Dataset):
    """Casillas vacias sinteticas (clase 0)."""
    def __init__(self, num_muestras=8000, transform=None):
        self.num_muestras = num_muestras
        self.transform = transform

    def __len__(self):
        return self.num_muestras

    def __getitem__(self, idx):
        color_fondo = random.randint(180, 255)
        img = np.full((28, 28), color_fondo, dtype=np.uint8)
        ruido = np.clip(np.random.normal(0, 15, (28, 28)), -40, 40).astype(np.int16)
        img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

        # Simular lineas residuales del grid
        if random.random() < 0.3:
            grosor = random.randint(1, 3)
            color_linea = random.randint(80, 180)
            lado = random.choice(['t', 'b', 'l', 'r'])
            if lado == 't': img[0:grosor, :] = color_linea
            elif lado == 'b': img[-grosor:, :] = color_linea
            elif lado == 'l': img[:, 0:grosor] = color_linea
            elif lado == 'r': img[:, -grosor:] = color_linea

        img_pil = Image.fromarray(img, mode='L')
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, 0


class DigitosSinteticos(Dataset):
    """Digitos con fuentes OpenCV, mas variaciones para cubrir confusiones 1/7, 4/1, 8/7."""
    def __init__(self, num_por_clase=800, transform=None):
        self.imagenes = []
        self.etiquetas = []
        self.transform = transform

        fuentes = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        ]

        for clase in range(1, 10):
            for _ in range(num_por_clase):
                color_fondo = random.randint(190, 255)
                img = np.full((28, 28), color_fondo, dtype=np.uint8)
                ruido = np.clip(np.random.normal(0, 10, (28, 28)), -20, 20).astype(np.int16)
                img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

                fuente = random.choice(fuentes)
                escala = random.uniform(0.5, 1.0)
                grosor = random.randint(1, 3)
                color_texto = random.randint(0, 80)

                (w, h), _ = cv2.getTextSize(str(clase), fuente, escala, grosor)
                x = int((28 - w) / 2) + random.randint(-4, 4)
                y = int((28 + h) / 2) + random.randint(-4, 4)
                cv2.putText(img, str(clase), (x, y), fuente, escala, color_texto, grosor)

                self.imagenes.append(img)
                self.etiquetas.append(clase)

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        img_pil = Image.fromarray(self.imagenes[idx], mode='L')
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, self.etiquetas[idx]


# =============================================================================
# PASO 3: TRANSFORMACIONES (augmentacion agresiva)
# =============================================================================

transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.8, 1.2), shear=8),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# =============================================================================
# PASO 4: MONTAJE DE DATOS
# =============================================================================

print("Descargando MNIST y generando datos sinteticos...")

mnist_train      = MNISTInvertido(train=True, transform=transform_train)
vacias_train     = CeldasVacias(num_muestras=8000, transform=transform_train)
sintetico_train  = DigitosSinteticos(num_por_clase=800, transform=transform_train)

mnist_test   = MNISTInvertido(train=False, transform=transform_test)
vacias_test  = CeldasVacias(num_muestras=1000, transform=transform_test)

dataset_train = ConcatDataset([mnist_train, vacias_train, sintetico_train])
dataset_test  = ConcatDataset([mnist_test, vacias_test])

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=2)

print(f"Entrenamiento: {len(dataset_train)} muestras")
print(f"Test:          {len(dataset_test)} muestras")

# =============================================================================
# PASO 5: ENTRENAMIENTO (30 epocas con CosineAnnealing)
# =============================================================================

modelo = LectorSudoku().to(device)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=30)

mejor_precision = 0.0
EPOCAS = 30

print(f"\nIniciando entrenamiento ({EPOCAS} epocas)...")
print("-" * 65)

for epoca in range(EPOCAS):
    modelo.train()
    perdida_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizador.zero_grad()
        salida = modelo(imgs)
        perdida = criterio(salida, labels)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()

    scheduler.step()
    lr_actual = optimizador.param_groups[0]['lr']

    # Validacion
    modelo.eval()
    aciertos = 0
    total = 0
    aciertos_clase = [0] * 10
    total_clase = [0] * 10

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, predicciones = torch.max(modelo(imgs), 1)
            total += labels.size(0)
            aciertos += (predicciones == labels).sum().item()
            for i in range(labels.size(0)):
                c = labels[i].item()
                total_clase[c] += 1
                if predicciones[i].item() == c:
                    aciertos_clase[c] += 1

    precision = 100 * aciertos / total
    loss_media = perdida_total / len(train_loader)

    marca = ""
    if precision > mejor_precision:
        mejor_precision = precision
        torch.save(modelo.state_dict(), "cnn_digits.pt")
        marca = " << MEJOR"

    print(f"Epoca {epoca+1:2d}/{EPOCAS} | Loss: {loss_media:.4f} | "
          f"Acc: {precision:.2f}% | LR: {lr_actual:.6f}{marca}")

# =============================================================================
# PASO 6: INFORME FINAL
# =============================================================================

print("\n" + "=" * 65)
print("PRECISION POR CLASE (ultima epoca)")
print("=" * 65)
nombres = ["Vacio", "Uno", "Dos", "Tres", "Cuatro",
           "Cinco", "Seis", "Siete", "Ocho", "Nueve"]
for c in range(10):
    if total_clase[c] > 0:
        pct = 100 * aciertos_clase[c] / total_clase[c]
        barra = "#" * int(pct / 2)
        print(f"  Clase {c} ({nombres[c]:>7s}): {pct:5.1f}% ({aciertos_clase[c]:>4d}/{total_clase[c]:>4d}) |{barra}")

print(f"\n  MEJOR PRECISION GLOBAL: {mejor_precision:.2f}%")

# =============================================================================
# PASO 7: MATRIZ DE CONFUSION (para ver errores 1<->7, 4<->1, 8<->7)
# =============================================================================

print("\n" + "=" * 65)
print("MATRIZ DE CONFUSION (filas=real, columnas=predicho)")
print("=" * 65)

modelo.load_state_dict(torch.load("cnn_digits.pt", map_location=device, weights_only=True))
modelo.eval()

confusion = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, preds = torch.max(modelo(imgs), 1)
        for real, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[real][pred] += 1

# Cabecera
print("       ", end="")
for j in range(10):
    print(f"  {j:>4d}", end="")
print()
print("       " + "-" * 50)

for i in range(10):
    total_fila = confusion[i].sum()
    print(f"  {i} ({nombres[i]:>7s}) |", end="")
    for j in range(10):
        val = confusion[i][j]
        if i == j:
            print(f" [{val:>3d}]", end="")
        elif val > 0:
            print(f"  {val:>3d} ", end="")
        else:
            print(f"    . ", end="")
    pct = 100 * confusion[i][i] / total_fila if total_fila > 0 else 0
    print(f"  | {pct:.1f}%")

# =============================================================================
# PASO 8: VERIFICACION RAPIDA
# =============================================================================

print("\n" + "=" * 65)
print("VERIFICACION CON DIGITOS SINTETICOS NUEVOS")
print("=" * 65)

fuentes_test = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

for clase in range(10):
    resultados = []
    for _ in range(15):
        if clase == 0:
            bg = random.randint(210, 250)
            img = np.full((28, 28), bg, dtype=np.uint8)
            ruido = np.clip(np.random.normal(0, 10, (28, 28)), -20, 20).astype(np.int16)
            img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
        else:
            bg = random.randint(210, 250)
            img = np.full((28, 28), bg, dtype=np.uint8)
            ruido = np.clip(np.random.normal(0, 8, (28, 28)), -15, 15).astype(np.int16)
            img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
            fuente = random.choice(fuentes_test)
            grosor = random.randint(1, 2)
            escala = random.uniform(0.6, 0.9)
            (w, h), _ = cv2.getTextSize(str(clase), fuente, escala, grosor)
            x = int((28 - w) / 2) + random.randint(-2, 2)
            y = int((28 + h) / 2) + random.randint(-2, 2)
            cv2.putText(img, str(clase), (x, y), fuente, escala, 30, grosor)

        tensor = torch.tensor((img / 255.0 - 0.5) / 0.5, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, pred = torch.max(modelo(tensor), 1)
        resultados.append(pred.item())

    correcto = sum(1 for r in resultados if r == clase)
    estado = "OK" if correcto >= 12 else "!!"
    print(f"  Clase {c} ({nombres[clase]:>7s}): {resultados} -> {correcto}/15 [{estado}]")

# =============================================================================
# PASO 9: DESCARGA
# =============================================================================
print("\n" + "=" * 65)
print("Modelo guardado como: cnn_digits.pt")
print("Copialo a: CODE/MODELOS/MODELO_2/cnn_digits.pt")
print("=" * 65)

# Descomenta para descarga automatica en Colab:
# from google.colab import files
# files.download("cnn_digits.pt")
