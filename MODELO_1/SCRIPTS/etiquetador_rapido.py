import os
import sys
import glob
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Ajustar PYTHONPATH
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from variables import RUTA_MODELO_YOLO, RUTA_MODELO_CNN, TAMANO_TABLERO
from vision import extraer_tablero, trocear_cuadricula
from modelo_cnn import LectorSudoku

# Configuración de rutas
DIR_PROYECTO = os.path.dirname(directorio_actual)
DIR_IMAGENES = os.path.join(DIR_PROYECTO, "data", "DATA") 
DIR_GUARDADO = os.path.join(DIR_PROYECTO, "data", "hard_samples")

print(f"Buscando imágenes en: {DIR_IMAGENES}")
print(f"Las muestras se guardarán en: {DIR_GUARDADO}")

# Cargar modelos
print("Cargando modelo YOLO...")
modelo_yolo = YOLO(RUTA_MODELO_YOLO)

print("Cargando modelo CNN...")
lector = LectorSudoku()
lector.load_state_dict(torch.load(RUTA_MODELO_CNN, map_location=torch.device('cpu'), weights_only=True))
lector.eval()

# Buscar imágenes
patrones = ["*.jpg", "*.jpeg", "*.png"]
rutas_imagenes = []
for pat in patrones:
    rutas_imagenes.extend(glob.glob(os.path.join(DIR_IMAGENES, "**", pat), recursive=True))

if not rutas_imagenes:
    print(f"No se encontraron imágenes en {DIR_IMAGENES}.")
    sys.exit(1)

buffer_global = []
img_cursor = 0
idx_actual = 0
salir_del_programa = False

def procesar_siguiente_imagen():
    global img_cursor, buffer_global
    if img_cursor >= len(rutas_imagenes): return False
    
    ruta_img = rutas_imagenes[img_cursor]
    print(f"\nProcesando imagen con YOLO: {os.path.basename(ruta_img)}...")
    img_cursor += 1
    
    imagen_cv = cv2.imread(ruta_img)
    if imagen_cv is None: return procesar_siguiente_imagen() # recursivo si hay error de lectura
        
    resultados = modelo_yolo(imagen_cv, verbose=False)
    caja = resultados[0].boxes.xyxy.cpu().numpy()
    
    if len(caja) == 0: return procesar_siguiente_imagen()
        
    try:
        tablero_gris = extraer_tablero(imagen_cv, caja, TAMANO_TABLERO)
        casillas = trocear_cuadricula(tablero_gris, 9, 9)
    except Exception as e:
        print(f"Error extrayendo tablero: {e}")
        return procesar_siguiente_imagen()

    # Preprocesar las 81 casillas y guardarlas en el buffer
    for idx, imagen_casilla in enumerate(casillas):
        alto, ancho = imagen_casilla.shape
        margen_y = int(alto * 0.1)
        margen_x = int(ancho * 0.1)
        casilla_sin_bordes = imagen_casilla[margen_y:alto-margen_y, margen_x:ancho-margen_x]
        
        blur = cv2.GaussianBlur(casilla_sin_bordes, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contorno_digito = None
        max_area = 0
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w > alto * 0.75 or h > alto * 0.75: continue
            if area < 20: continue
            if area > max_area:
                max_area = area
                contorno_digito = cnt

        lienzo_final = np.zeros((28, 28), dtype=np.uint8)
        tiene_digito = False

        if contorno_digito is not None:
            x, y, w, h = cv2.boundingRect(contorno_digito)
            kernel = np.ones((2, 2), np.uint8)
            digito_soldado = cv2.morphologyEx(thresh[y:y+h, x:x+w], cv2.MORPH_CLOSE, kernel)
            
            lado_max = 20
            if w > 0 and h > 0:
                ratio = min(lado_max / w, lado_max / h)
                nuevo_w = max(1, int(w * ratio))
                nuevo_h = max(1, int(h * ratio))
                digito_resized = cv2.resize(digito_soldado, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
                
                fondo = np.zeros((28, 28), dtype=np.uint8)
                inicio_y = (28 - nuevo_h) // 2
                inicio_x = (28 - nuevo_w) // 2
                fondo[inicio_y:inicio_y+nuevo_h, inicio_x:inicio_x+nuevo_w] = digito_resized
                lienzo_final = cv2.GaussianBlur(fondo, (3, 3), 0)
                tiene_digito = True
        
        # Inferencia CNN
        tensor_casilla = torch.tensor(lienzo_final, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_casilla = 1.0 - (tensor_casilla / 127.5)
        
        prediccion_num = 0
        if tiene_digito:
            with torch.no_grad():
                prediccion = lector(tensor_casilla)
                _, pronostico = torch.max(prediccion, 1)
                prediccion_num = pronostico.item()
                
            buffer_global.append({
                'img': lienzo_final,
                'cnn': prediccion_num,
                'tiene_digito': tiene_digito,
                'origen': os.path.basename(ruta_img),
                'etiqueta': None,
                'ruta_guardada': None
            })
    return True

cv2.namedWindow("Etiquetador Rapido Sudoku", cv2.WINDOW_AUTOSIZE)

while True:
    if idx_actual >= len(buffer_global):
        if not procesar_siguiente_imagen():
            # Ya no hay mas imagenes que procesar
            break
        continue
        
    celda = buffer_global[idx_actual]
    
    # --- UI: Panel Principal ---
    # Tamaño de la imagen principal 300x300
    lienzo_mostrar = cv2.resize(celda['img'], (300, 300), interpolation=cv2.INTER_NEAREST)
    lienzo_mostrar = cv2.cvtColor(lienzo_mostrar, cv2.COLOR_GRAY2BGR)
    
    # Textos simples y gigantes
    color_cnn = (0, 255, 0) if celda['tiene_digito'] else (0, 165, 255)
    cv2.putText(lienzo_mostrar, f"CNN: {celda['cnn']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_cnn, 3)
    cv2.putText(lienzo_mostrar, f"{celda['origen'][:15]}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Texto de progreso
    texto_progreso = f"Progreso: {idx_actual + 1} | Img: {img_cursor}/{len(rutas_imagenes)}"
    cv2.putText(lienzo_mostrar, texto_progreso, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Texto de atajos (pequeño y discreto)
    cv2.putText(lienzo_mostrar, "[1-9] Guardar | [0/ESP] Saltar | [+] Rotar", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(lienzo_mostrar, "[Z] Deshacer | [Q] Salir", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # --- UI: Tira de Historial ---
    # Queremos mostrar las ultimas 5 decisiones debajo de la imagen principal
    historico = buffer_global[max(0, idx_actual - 5):idx_actual]
    tira_historial = np.zeros((80, 300, 3), dtype=np.uint8)
    
    # Cada thumbnail ocupará 60x60
    for i, celda_h in enumerate(historico):
        thumb = cv2.resize(celda_h['img'], (60, 60), interpolation=cv2.INTER_NEAREST)
        thumb_c = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        x_offset = i * 60
        tira_historial[0:60, x_offset:x_offset+60] = thumb_c
        
        # Etiqueta que el usuario eligió
        etq = celda_h['etiqueta'] if celda_h['etiqueta'] is not None else "Salt."
        color_etq = (0, 255, 0) if etq != "Salt." else (150, 150, 150)
        cv2.putText(tira_historial, f"-> {etq}", (x_offset + 5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_etq, 1)

    # Juntar principal y tira historial (300x300 + 300x80 = 300x380)
    pantalla_final = np.vstack((lienzo_mostrar, tira_historial))
    
    cv2.imshow("Etiquetador Rapido Sudoku", pantalla_final)
    key = cv2.waitKey(0)
    
    if key in [27, ord('q'), ord('Q')]: # ESC o Q
        break
    elif key in [ord('z'), ord('Z'), 8]: # Z o Backspace (Deshacer)
        if idx_actual > 0:
            idx_actual -= 1
            # Eliminar la etiqueta guardada físicamente
            celda_previa = buffer_global[idx_actual]
            if celda_previa['ruta_guardada'] and os.path.exists(celda_previa['ruta_guardada']):
                try:
                    os.remove(celda_previa['ruta_guardada'])
                    print(f"Deshecho: borrado {celda_previa['ruta_guardada']}")
                except Exception as e:
                    print(f"Error borrando deshacer: {e}")
            celda_previa['ruta_guardada'] = None
            celda_previa['etiqueta'] = None
    elif key == ord('+'): # Rotar 90 grados
        celda['img'] = cv2.rotate(celda['img'], cv2.ROTATE_90_CLOCKWISE)
        # Volver a pasar la CNN para actualizar la predicción de la imagen rotada
        tensor_rotado = torch.tensor(celda['img'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_rotado = 1.0 - (tensor_rotado / 127.5)
        with torch.no_grad():
            pred = lector(tensor_rotado)
            _, pronostico = torch.max(pred, 1)
            celda['cnn'] = pronostico.item()
    elif key in [32, ord('b'), ord('0')]: # ESPACIO o 0 (Saltar)
        celda['etiqueta'] = "Salt."
        idx_actual += 1
    elif ord('1') <= key <= ord('9'): # Numeros 1-9 (Clasificar)
        digito_real = chr(key)
        dir_destino = os.path.join(DIR_GUARDADO, digito_real)
        os.makedirs(dir_destino, exist_ok=True)
        ruta_guardado = os.path.join(dir_destino, f"etiquetador_{int(time.time()*1000)}_{idx_actual}.png")
        cv2.imwrite(ruta_guardado, celda['img'])
        
        celda['ruta_guardada'] = ruta_guardado
        celda['etiqueta'] = digito_real
        idx_actual += 1

cv2.destroyAllWindows()
print("\n¡Etiquetado finalizado!")
