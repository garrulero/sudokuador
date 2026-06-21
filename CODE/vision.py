import cv2
import numpy as np

def _ordenar_puntos(pts):
    """Ordena los 4 vértices: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extraer_tablero(imagen_cv, coordenadas_yolo, tamano_final=450):
    """
    Toma la imagen original y las coordenadas de YOLO,
    intenta enderezar el Sudoku con una transformación de perspectiva
    y devuelve la imagen en escala de grises de 450x450.
    """
    # 1. Sacamos las coordenadas de la caja que detectó YOLO
    x1, y1, x2, y2 = map(int, coordenadas_yolo[0])
    
    # Expandimos un pelín el recorte de YOLO para asegurarnos de no cortar el borde exterior
    alto_img, ancho_img = imagen_cv.shape[:2]
    margen = 5
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(ancho_img, x2 + margen)
    y2 = min(alto_img, y2 + margen)
    
    recorte = imagen_cv[y1:y2, x1:x2]
    recorte_gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    
    # 2. Intentamos buscar el cuadrado exacto para hacer Corrección de Perspectiva
    recorte_blur = cv2.GaussianBlur(recorte_gris, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(recorte_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        # Ordenamos los contornos por área, buscamos el más grande que parezca un cuadrado
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
        for c in contornos:
            perimetro = cv2.arcLength(c, True)
            aproximacion = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            
            # Si el contorno tiene 4 vértices y es razonablemente grande
            if len(aproximacion) == 4 and cv2.contourArea(c) > (recorte_gris.shape[0] * recorte_gris.shape[1]) * 0.2:
                pts_origen = aproximacion.reshape(4, 2).astype(np.float32)
                pts_origen = _ordenar_puntos(pts_origen)
                
                pts_destino = np.array([
                    [0, 0],
                    [tamano_final - 1, 0],
                    [tamano_final - 1, tamano_final - 1],
                    [0, tamano_final - 1]
                ], dtype=np.float32)
                
                matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
                tablero_cuadrado = cv2.warpPerspective(recorte_gris, matriz, (tamano_final, tamano_final))
                return tablero_cuadrado
    
    # 3. FALLBACK: Si no encontramos el cuadrado, hacemos un resize normal
    tablero_cuadrado = cv2.resize(recorte_gris, (tamano_final, tamano_final))
    return tablero_cuadrado

def trocear_cuadricula(tablero_gris, filas=9, columnas=9):
    """
    Divide la imagen del tablero en 81 casillas individuales.
    Primero intenta detectar las líneas reales de la cuadrícula con morfología
    para hacer un corte preciso. Si no lo consigue, usa división equitativa.
    """
    # Intentar detección inteligente de las líneas del grid
    bordes_h, bordes_v = _detectar_lineas_grid(tablero_gris)
    
    if bordes_h is not None and bordes_v is not None:
        # Recorte PRECISO: usamos las líneas reales detectadas
        casillas = []
        for i in range(filas):
            for j in range(columnas):
                y1 = bordes_h[i]
                y2 = bordes_h[i + 1]
                x1 = bordes_v[j]
                x2 = bordes_v[j + 1]
                casilla = tablero_gris[y1:y2, x1:x2]
                casillas.append(casilla)
        return casillas
    
    # FALLBACK: división equitativa simple (por si falla la detección)
    return _division_simple(tablero_gris, filas, columnas)


def _detectar_lineas_grid(tablero_gris):
    """
    Detecta las 10 líneas horizontales y 10 verticales de la cuadrícula de Sudoku
    usando operaciones morfológicas (apertura con kernels lineales).
    
    Devuelve dos listas de 10 posiciones cada una, o (None, None) si falla.
    """
    alto, ancho = tablero_gris.shape
    
    # 1. Binarizar con Otsu (las líneas del grid se vuelven blancas)
    _, binaria = cv2.threshold(tablero_gris, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Detectar líneas HORIZONTALES
    #    Usamos un kernel horizontal largo: solo sobreviven estructuras que
    #    crucen al menos 1/4 del ancho de la imagen (= una línea de cuadrícula)
    largo_h = ancho // 4
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (largo_h, 1))
    mascara_h = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_h)
    
    # 3. Detectar líneas VERTICALES (mismo concepto, kernel vertical)
    largo_v = alto // 4
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, largo_v))
    mascara_v = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_v)
    
    # 4. Proyectar para encontrar las posiciones Y (horizontales) y X (verticales)
    #    Sumamos píxeles blancos a lo largo de cada eje
    proyeccion_h = np.sum(mascara_h, axis=1) / 255  # Suma por filas → posición Y
    proyeccion_v = np.sum(mascara_v, axis=0) / 255  # Suma por columnas → posición X
    
    # 5. Umbral: una línea real debe cubrir al menos el 20% del ancho/alto
    umbral_h = ancho * 0.2
    umbral_v = alto * 0.2
    
    pos_h = np.where(proyeccion_h > umbral_h)[0]
    pos_v = np.where(proyeccion_v > umbral_v)[0]
    
    # 6. Agrupar píxeles cercanos en líneas individuales
    #    (una línea gruesa ocupa varios píxeles, los agrupamos y tomamos el centro)
    distancia_agrupacion = max(alto, ancho) // 20  # ~5% del tamaño
    lineas_h = _agrupar_en_lineas(pos_h, distancia_agrupacion)
    lineas_v = _agrupar_en_lineas(pos_v, distancia_agrupacion)
    
    # 7. Validar: un Sudoku tiene exactamente 10 líneas en cada dirección
    #    (borde superior + 8 interiores + borde inferior)
    if len(lineas_h) == 10 and len(lineas_v) == 10:
        return lineas_h, lineas_v
    
    # Si no detectamos exactamente 10, devolvemos None para usar el fallback
    return None, None


def _agrupar_en_lineas(posiciones, distancia_min=10):
    """
    Agrupa posiciones de píxeles consecutivas/cercanas en una sola línea.
    Devuelve el punto MEDIO de cada grupo (= centro de cada línea de la cuadrícula).
    """
    if len(posiciones) == 0:
        return []
    
    grupos = []
    inicio_grupo = posiciones[0]
    fin_grupo = posiciones[0]
    
    for i in range(1, len(posiciones)):
        if posiciones[i] - fin_grupo <= distancia_min:
            # Este píxel pertenece al mismo grupo (misma línea)
            fin_grupo = posiciones[i]
        else:
            # Nuevo grupo → guardar el anterior
            grupos.append((inicio_grupo + fin_grupo) // 2)
            inicio_grupo = posiciones[i]
            fin_grupo = posiciones[i]
    
    # Guardar el último grupo
    grupos.append((inicio_grupo + fin_grupo) // 2)
    return grupos


def _division_simple(tablero_gris, filas, columnas):
    """
    Fallback: divide el tablero con aritmética simple (división equitativa).
    Se usa cuando la detección de líneas no encuentra exactamente 10+10 líneas.
    """
    casillas = []
    alto_celda = tablero_gris.shape[0] // filas
    ancho_celda = tablero_gris.shape[1] // columnas
    
    for i in range(filas):
        for j in range(columnas):
            y_inicio = i * alto_celda
            y_fin = (i + 1) * alto_celda
            x_inicio = j * ancho_celda
            x_fin = (j + 1) * ancho_celda
            
            casilla = tablero_gris[y_inicio:y_fin, x_inicio:x_fin]
            casillas.append(casilla)
            
    return casillas