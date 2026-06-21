import os
import sys
import base64
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Añadir el directorio actual de CODE al path para importar módulos correctamente
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from variables import RUTA_MODELO_YOLO, RUTA_MODELO_CNN, TAMANO_TABLERO
from funciones import resolver_sudoku, corregir_con_reglas_sudoku
from vision import extraer_tablero, trocear_cuadricula
from modelo_cnn import LectorSudoku

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Sudokuador API",
    description="Backend personalizado de visión artificial y resolución de Sudokus"
)

# Habilitar CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los modelos una sola vez al iniciar la aplicación para rendimiento óptimo
print("Cargando modelo YOLO...")
try:
    modelo_yolo = YOLO(RUTA_MODELO_YOLO)
    print("¡YOLO cargado correctamente!")
except Exception as e:
    print(f"ERROR cargando YOLO: {e}")
    modelo_yolo = None

print("Cargando modelo CNN...")
try:
    lector = LectorSudoku()
    lector.load_state_dict(torch.load(RUTA_MODELO_CNN, map_location=torch.device('cpu'), weights_only=True))
    lector.eval()
    print("¡CNN cargada correctamente!")
except Exception as e:
    print(f"ERROR cargando CNN: {e}")
    lector = None


class SolveRequest(BaseModel):
    grid: list[list[int]]


def codificar_imagen_base64(imagen_np):
    """Codifica una imagen NumPy a string base64 en formato JPEG."""
    exito, buffer = cv2.imencode('.jpg', imagen_np)
    if not exito:
        return ""
    return base64.b64encode(buffer).decode('utf-8')


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if modelo_yolo is None or lector is None:
        raise HTTPException(
            status_code=500,
            detail="Los modelos de IA no se cargaron correctamente en el servidor."
        )

    try:
        # 1. Leer los bytes del archivo subido
        contenido = await file.read()
        nparr = np.frombuffer(contenido, np.uint8)
        imagen_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen_cv is None:
            raise HTTPException(
                status_code=400,
                detail="El archivo proporcionado no es una imagen válida."
            )

        # 2. Ejecutar la detección YOLO para encontrar el tablero
        resultados = modelo_yolo(imagen_cv)
        caja = resultados[0].boxes.xyxy.cpu().numpy()

        if len(caja) == 0:
            raise HTTPException(
                status_code=400,
                detail="No se ha detectado el tablero de Sudoku en la imagen. Intenta con una foto más centrada y con mejor iluminación."
            )

        # 3. Recortar el tablero y pasarlo a escala de grises
        tablero_gris = extraer_tablero(imagen_cv, caja, TAMANO_TABLERO)
        tablero_b64 = codificar_imagen_base64(tablero_gris)

        # 4. Trocear el tablero en 81 casillas
        casillas = trocear_cuadricula(tablero_gris, 9, 9)

        # 5. Inferencia con la CNN de PyTorch
        matriz_detectada = np.zeros((9, 9), dtype=int)
        matriz_confianzas = np.zeros((9, 9), dtype=float)
        casillas_para_depurar_b64 = []
        registro_logs = []

        for idx, imagen_casilla in enumerate(casillas):
            # Preprocesamiento de la casilla: Buscar el dígito dentro y centrarlo
            alto, ancho = imagen_casilla.shape
            margen_y = int(alto * 0.1)
            margen_x = int(ancho * 0.1)
            casilla_sin_bordes = imagen_casilla[margen_y:alto-margen_y, margen_x:ancho-margen_x]
            
            # 1. Binarización Adaptativa Equilibrada
            # Quitamos la Normalización porque "sucio" las casillas vacías convirtiendo el ruido en manchas negras.
            # Un C=5 es el "Punto Dulce": no tan bajo como para rellenar los agujeros de los 6 y 9,
            # ni tan alto como para borrar el número entero.
            blur = cv2.GaussianBlur(casilla_sin_bordes, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            
            # 2. Encontrar contornos
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contorno_digito = None
            max_area = 0
            
            for cnt in contornos:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                
                # Ignorar si es una raya de la cuadrícula (toca los bordes del recorte)
                if w > alto * 0.75 or h > alto * 0.75:
                    continue
                
                # Ignorar motas de polvo / sombras muy pequeñas
                if area < 20:
                    continue
                
                # Quedarse siempre con el manchón más grande (el número real)
                if area > max_area:
                    max_area = area
                    contorno_digito = cnt

            lienzo_final = np.zeros((28, 28), dtype=np.uint8)
            tiene_digito = False

            if contorno_digito is not None:
                x, y, w, h = cv2.boundingRect(contorno_digito)
                # Usamos el recorte directamente del adaptiveThreshold original
                # No re-binarizamos con OTSU porque dentro del bounding box estrecho
                # el histograma no es bimodal y OTSU rellenaría los huecos de los 8s y 9s engordando la letra.
                # Opcional para imágenes muy borrosas: soldar pequeños huecos en el trazo y suavizar.
                kernel = np.ones((2, 2), np.uint8)
                digito_soldado = cv2.morphologyEx(thresh[y:y+h, x:x+w], cv2.MORPH_CLOSE, kernel)
                
                # MNIST tiene dígitos de ~20x20
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
                    
                    # Simular el Anti-Aliasing (bordes suaves) de MNIST/Fuentes con un ligero Blur final
                    lienzo_final = cv2.GaussianBlur(fondo, (3, 3), 0)
                    tiene_digito = True
            
            # Guardar imagen para depuración
            casillas_para_depurar_b64.append(codificar_imagen_base64(lienzo_final))

            # Convertimos al rango [-1, 1] que espera la red.
            # ¡CUIDADO! La red se entrenó en Colab (MNIST y Sintéticos) con fondo blanco (1.0) y tinta negra (-1.0).
            # Nuestro lienzo_final tiene fondo negro (0) y tinta blanca (255).
            # Por lo tanto, invertimos la matemática:
            # 0 (fondo negro) -> se vuelve +1.0
            # 255 (tinta blanca) -> se vuelve -1.0
            tensor_casilla = torch.tensor(lienzo_final, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor_casilla = 1.0 - (tensor_casilla / 127.5)
            
            if not tiene_digito:
                # Si no encontramos dígito, forzamos la predicción a 0 artificialmente
                prediccion_forzada = torch.zeros((1, 10))
                prediccion_forzada[0, 0] = 15.0 # Muy alta confianza para 0
                prediccion_forzada[0, 1:] = -5.0
                prediccion = prediccion_forzada
                numero_leido = 0
            else:
                # Inferencia real
                with torch.no_grad():
                    prediccion = lector(tensor_casilla)
                    _, pronostico = torch.max(prediccion, 1)
                    numero_leido = pronostico.item()
            
            # Almacenar en la matriz
            fila = idx // 9
            columna = idx % 9
            matriz_detectada[fila][columna] = numero_leido
            matriz_confianzas[fila][columna] = prediccion[0][numero_leido].item()

            # Guardar logs
            logits = [round(val, 2) for val in prediccion[0].tolist()]
            registro_logs.append({
                "casilla": idx,
                "fila": fila,
                "col": columna,
                "veredicto": numero_leido,
                "confianza": prediccion[0][numero_leido].item(),
                "logits": logits,
                "pixel_min": tensor_casilla.min().item(),
                "pixel_max": tensor_casilla.max().item(),
                "pixel_media": tensor_casilla.mean().item()
            })

        # 6. Autocorrección usando reglas del Sudoku
        correcciones = corregir_con_reglas_sudoku(matriz_detectada, matriz_confianzas)

        return {
            "status": "success",
            "grid": matriz_detectada.tolist(),
            "confidences": matriz_confianzas.tolist(),
            "corrections": correcciones,
            "board_image": f"data:image/jpeg;base64,{tablero_b64}",
            "cells_images": [f"data:image/jpeg;base64,{cell}" for cell in casillas_para_depurar_b64],
            "logs": registro_logs
        }

    except Exception as e:
        print(f"Error procesando imagen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando el Sudoku: {str(e)}"
        )


@app.post("/api/solve")
async def solve_sudoku_endpoint(req: SolveRequest):
    grid = np.array(req.grid, dtype=int)
    if grid.shape != (9, 9):
        raise HTTPException(
            status_code=400,
            detail="La cuadrícula del Sudoku debe ser exactamente de 9x9."
        )

    # Verificar si es válido inicialmente (por si hay duplicados introducidos por el usuario)
    for f in range(9):
        for c in range(9):
            num = grid[f, c]
            if num != 0:
                # Comprobamos temporalmente sin ese número
                grid[f, c] = 0
                valido = resolver_sudoku_temporal_validez(grid, f, c, num)
                grid[f, c] = num
                if not valido:
                    raise HTTPException(
                        status_code=400,
                        detail=f"El Sudoku ingresado contiene conflictos iniciales (por ejemplo, el número {num} está repetido en la fila, columna o bloque de la casilla Fila {f+1}, Col {c+1})."
                    )

    # Intentar resolver
    matriz_resuelta = grid.copy()
    exito = resolver_sudoku(matriz_resuelta)

    if exito:
        return {
            "status": "success",
            "solved": True,
            "grid": matriz_resuelta.tolist()
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Este Sudoku no tiene solución lógica válida. Revisa los números detectados o editados."
        )


def resolver_sudoku_temporal_validez(tablero, fila, col, num):
    """Comprobación simple de validez inicial."""
    if num in tablero[fila, :]:
        return False
    if num in tablero[:, col]:
        return False
    inicio_fila = (fila // 3) * 3
    inicio_col = (col // 3) * 3
    cuadrante = tablero[inicio_fila:inicio_fila+3, inicio_col:inicio_col+3]
    if num in cuadrante:
        return False
    return True


# Servir la interfaz web estática en la raíz
ruta_estaticos = os.path.join(directorio_actual, "static")
if not os.path.exists(ruta_estaticos):
    os.makedirs(ruta_estaticos, exist_ok=True)

app.mount("/", StaticFiles(directory=ruta_estaticos, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # Ejecutar en localhost puerto 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
