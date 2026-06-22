import cv2
import numpy as np
import os
import random

# 1. Configuración de nuestro dataset sintético
# =========================================================
# Generaremos carpetas del 0 al 9 con 1000 imágenes cada una
ruta_base = "C:/BOOT/garrulero/sudokuador/CODE/DATASET_SINTETICO"
num_muestras_por_clase = 1000
tamano = 28 # Píxeles (28x28)

# Tipografías base de OpenCV para simular distintos tipos de impresión
fuentes = [
    cv2.FONT_HERSHEY_SIMPLEX, 
    cv2.FONT_HERSHEY_COMPLEX, 
    cv2.FONT_HERSHEY_TRIPLEX, 
    cv2.FONT_HERSHEY_DUPLEX  # <-- Aquí faltaba el "_HERSHEY_"
]

# Creamos las carpetas (del 0 al 9) automáticamente
for i in range(10):
    os.makedirs(os.path.join(ruta_base, str(i)), exist_ok=True)

print(f"Iniciando la fábrica de datos... Se generarán {num_muestras_por_clase * 10} imágenes.")

# 2. Bucle de fabricación
# =========================================================
for clase in range(10):
    for i in range(num_muestras_por_clase):
        
        # A. Crear fondo base (blanco sucio / gris claro simulando papel real)
        color_fondo = random.randint(200, 255)
        img = np.full((tamano, tamano), color_fondo, dtype=np.uint8)

        # B. Añadir "ruido" al fondo (simulando la textura del papel y la cámara)
        # NOTA: np.random.normal genera negativos → clip antes de uint8 para evitar overflow
        ruido = np.clip(np.random.normal(0, 15, (tamano, tamano)), -30, 30).astype(np.int16)
        img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

        # C. Si la clase NO es 0 (celda vacía), dibujamos un número
        if clase != 0:
            fuente = random.choice(fuentes)
            escala = random.uniform(0.6, 0.9)  # Tamaños ligeramente distintos
            grosor = random.randint(1, 2)      # Tinta normal o negrita
            color_texto = random.randint(0, 60) # Negro o gris muy oscuro
            
            # Calcular cuánto ocupa el texto para centrarlo (con una ligera desviación aleatoria)
            (w, h), _ = cv2.getTextSize(str(clase), fuente, escala, grosor)
            x = int((tamano - w) / 2) + random.randint(-3, 3)
            y = int((tamano + h) / 2) + random.randint(-3, 3)
            
            # Estampar el número en la imagen
            cv2.putText(img, str(clase), (x, y), fuente, escala, color_texto, grosor)

        # D. Guardar la imagen en su carpeta correspondiente
        nombre_archivo = f"{clase}_{i}.jpg"
        ruta_img = os.path.join(ruta_base, str(clase), nombre_archivo)
        cv2.imwrite(ruta_img, img)
        
    print(f"  -> Clase {clase} terminada ({num_muestras_por_clase} imágenes).")

print(f"\n¡Proceso finalizado! Tu dataset sintético te espera en: {ruta_base}")