import os
import shutil

# 1. Definimos dónde están las cosas (rutas extraídas de tu árbol)
ruta_robo = r"C:\BOOT\GARRULERO\SUDOKUADOR\MODELO_1\DATA\RAW\roboflow\Sudoku Vision.v2-v2-2021-12-11-1-01am.yolov8"
ruta_bridge = r"C:\BOOT\GARRULERO\SUDOKUADOR\MODELO_1\DATA\RAW\bridge\drive-download-20260620T185558Z-3-001"

# 2. Definimos dónde queremos el Super-Dataset
ruta_salida = r"C:\BOOT\GARRULERO\SUDOKUADOR\MODELO_1\DATA\dataset_fusionado"

# 3. Creamos las carpetas maestras vacías
subconjuntos = ['train', 'valid', 'test']
for sub in subconjuntos:
    os.makedirs(os.path.join(ruta_salida, 'images', sub), exist_ok=True)
    os.makedirs(os.path.join(ruta_salida, 'labels', sub), exist_ok=True)

def copiar_con_prefijo(ruta_img, ruta_lbl, sub_destino, prefijo):
    if not os.path.exists(ruta_img): 
        return
        
    archivos = os.listdir(ruta_img)
    for img_name in archivos:
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue
        
        # Sacamos el nombre del txt asociado
        nombre_base = os.path.splitext(img_name)[0]
        txt_name = nombre_base + ".txt"
        
        if os.path.exists(os.path.join(ruta_lbl, txt_name)):
            # Bautizamos los archivos para evitar colisiones
            nuevo_img = f"{prefijo}_{img_name}"
            nuevo_txt = f"{prefijo}_{txt_name}"
            
            # Copiamos a la carpeta maestra
            shutil.copy(os.path.join(ruta_img, img_name), os.path.join(ruta_salida, 'images', sub_destino, nuevo_img))
            shutil.copy(os.path.join(ruta_lbl, txt_name), os.path.join(ruta_salida, 'labels', sub_destino, nuevo_txt))

# ==========================================
# ¡Que empiece la magia!
# ==========================================

print("Fusionando Dataset de Roboflow (Pete)...")
# La estructura de Pete es: train/images y train/labels
copiar_con_prefijo(os.path.join(ruta_robo, 'train', 'images'), os.path.join(ruta_robo, 'train', 'labels'), 'train', 'robo')
copiar_con_prefijo(os.path.join(ruta_robo, 'valid', 'images'), os.path.join(ruta_robo, 'valid', 'labels'), 'valid', 'robo')
copiar_con_prefijo(os.path.join(ruta_robo, 'test', 'images'), os.path.join(ruta_robo, 'test', 'labels'), 'test', 'robo')

print("Fusionando Dataset de The Bridge (Profesores)...")
# La estructura de Bridge es: images/train y labels/train (¡y usan 'val' en vez de 'valid'!)
copiar_con_prefijo(os.path.join(ruta_bridge, 'images', 'train'), os.path.join(ruta_bridge, 'labels', 'train'), 'train', 'bridge')
copiar_con_prefijo(os.path.join(ruta_bridge, 'images', 'val'), os.path.join(ruta_bridge, 'labels', 'val'), 'valid', 'bridge')
copiar_con_prefijo(os.path.join(ruta_bridge, 'images', 'test'), os.path.join(ruta_bridge, 'labels', 'test'), 'test', 'bridge')

print(f"¡Fusión completada! Tu super-dataset te espera en:\n{ruta_salida}")