"""
Script de diagnóstico para verificar si el modelo CNN funciona correctamente.
Genera dígitos sintéticos (idénticos al entrenamiento) y los pasa por la red.
"""
import cv2
import numpy as np
import torch
import random
from modelo_cnn import LectorSudoku

# =========================================================
# 1. Cargar el modelo
# =========================================================
RUTA_MODELO = "C:/BOOT/garrulero/sudokuador/CODE/MODELOS/MODELO_2/cnn_digits.pt"

lector = LectorSudoku()
lector.load_state_dict(torch.load(RUTA_MODELO, map_location=torch.device('cpu'), weights_only=True))
lector.eval()

print("=" * 60)
print("DIAGNÓSTICO DEL MODELO CNN")
print("=" * 60)

# =========================================================
# 2. Generar dígitos sintéticos IDÉNTICOS al entrenamiento
# =========================================================
fuentes = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
]

def generar_digito_sintetico(clase):
    """Genera una imagen 28x28 exactamente como fabricar_datos.py"""
    tamano = 28
    color_fondo = random.randint(200, 255)
    img = np.full((tamano, tamano), color_fondo, dtype=np.uint8)
    ruido = np.random.normal(0, 15, (tamano, tamano)).astype(np.uint8)
    img = cv2.add(img, ruido)
    
    if clase != 0:
        fuente = random.choice(fuentes)
        escala = random.uniform(0.6, 0.9)
        grosor = random.randint(1, 2)
        color_texto = random.randint(0, 60)
        (w, h), _ = cv2.getTextSize(str(clase), fuente, escala, grosor)
        x = int((tamano - w) / 2) + random.randint(-3, 3)
        y = int((tamano + h) / 2) + random.randint(-3, 3)
        cv2.putText(img, str(clase), (x, y), fuente, escala, color_texto, grosor)
    
    return img

def normalizar_como_entrenamiento(img_28):
    """Replica transforms.ToTensor() + Normalize((0.5,), (0.5,))"""
    img_norm = img_28 / 255.0
    img_norm = (img_norm - 0.5) / 0.5
    return torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# =========================================================
# 3. TEST A: ¿El modelo reconoce sus propios datos sintéticos?
# =========================================================
print("\n--- TEST A: Dígitos sintéticos (como en entrenamiento) ---")
aciertos = 0
total = 0
errores = []

for clase in range(10):
    for _ in range(20):  # 20 muestras por clase
        img = generar_digito_sintetico(clase)
        tensor = normalizar_como_entrenamiento(img)
        
        with torch.no_grad():
            logits = lector(tensor)
            _, pred = torch.max(logits, 1)
            predicho = pred.item()
        
        if predicho == clase:
            aciertos += 1
        else:
            errores.append((clase, predicho))
        total += 1

precision = 100 * aciertos / total
print(f"  Precisión en datos sintéticos: {aciertos}/{total} = {precision:.1f}%")
if precision < 80:
    print("  [!] El modelo NO funciona bien ni con sus propios datos!")
    print("  -> El modelo puede estar mal entrenado o el archivo .pt es incorrecto.")
else:
    print("  [OK] El modelo funciona bien con datos sinteticos.")
    print("  -> El problema esta en la DIFERENCIA entre datos reales y sinteticos (domain gap).")

if errores:
    print(f"\n  Errores ({len(errores)}):")
    for real, pred in errores[:10]:
        print(f"    Real: {real} -> Predicho: {pred}")

# =========================================================
# 4. TEST B: Simular qué pasa con una imagen "realista"
# =========================================================
print("\n--- TEST B: Simulación de imagen de cámara ---")

def simular_casilla_real(clase, con_threshold=False):
    """Simula una casilla de una foto real de un sudoku"""
    tamano = 50  # Tamaño típico de una casilla extraída
    
    # Fondo más grisáceo (como papel fotografiado con sombras)
    color_fondo = random.randint(160, 200)
    img = np.full((tamano, tamano), color_fondo, dtype=np.uint8)
    
    # Ruido de cámara más agresivo
    ruido = np.random.normal(0, 25, (tamano, tamano)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    
    if clase != 0:
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.2
        grosor = 2
        color_texto = random.randint(20, 80)
        (w, h), _ = cv2.getTextSize(str(clase), fuente, escala, grosor)
        x = int((tamano - w) / 2) + random.randint(-2, 2)
        y = int((tamano + h) / 2) + random.randint(-2, 2)
        cv2.putText(img, str(clase), (x, y), fuente, escala, color_texto, grosor)
    
    # Simular el preprocesamiento actual de main.py
    margen = int(tamano * 0.15)
    recortada = img[margen:tamano-margen, margen:tamano-margen]
    
    if con_threshold:
        # Con binarización adaptativa
        blur = cv2.GaussianBlur(recortada, (3, 3), 0)
        binaria = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        img_28 = cv2.resize(binaria, (28, 28))
    else:
        # Preprocesamiento actual: NORM_MINMAX
        normalizada = cv2.normalize(recortada, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        if np.mean(normalizada) < 127:
            normalizada = cv2.bitwise_not(normalizada)
        img_28 = cv2.resize(normalizada, (28, 28))
    
    return normalizar_como_entrenamiento(img_28)

# Test sin threshold (método actual)
print("\n  Método ACTUAL (NORM_MINMAX, sin threshold):")
aciertos_actual = 0
total_b = 0
for clase in range(10):
    preds = []
    for _ in range(10):
        tensor = simular_casilla_real(clase, con_threshold=False)
        with torch.no_grad():
            _, pred = torch.max(lector(tensor), 1)
            preds.append(pred.item())
            if pred.item() == clase:
                aciertos_actual += 1
            total_b += 1
    print(f"    Clase {clase}: predicciones = {preds}")

print(f"  Precisión: {100*aciertos_actual/total_b:.1f}%")

# Test con threshold (método propuesto)
print("\n  Método PROPUESTO (con binarización adaptativa):")
aciertos_threshold = 0
total_c = 0
for clase in range(10):
    preds = []
    for _ in range(10):
        tensor = simular_casilla_real(clase, con_threshold=True)
        with torch.no_grad():
            _, pred = torch.max(lector(tensor), 1)
            preds.append(pred.item())
            if pred.item() == clase:
                aciertos_threshold += 1
            total_c += 1
    print(f"    Clase {clase}: predicciones = {preds}")

print(f"  Precisión: {100*aciertos_threshold/total_c:.1f}%")

print("\n" + "=" * 60)
print("CONCLUSIÓN:")
print("=" * 60)
