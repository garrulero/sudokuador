import os
import sys
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO
import streamlit as st

# Añadir el directorio actual al path
directorio_actual = os.path.dirname(os.path.abspath(__file__))
if directorio_actual not in sys.path:
    sys.path.append(directorio_actual)

from variables import RUTA_MODELO_YOLO, RUTA_MODELO_CNN, TAMANO_TABLERO
from funciones import resolver_sudoku, corregir_con_reglas_sudoku
from vision import extraer_tablero, trocear_cuadricula
from modelo_cnn import LectorSudoku

# Configuración de página
st.set_page_config(page_title="SUDOKUADOR", page_icon="🧩", layout="wide")

# Inyección de CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Animaciones Galácticas / Mágicas (Light) */
    @keyframes hyperdrive {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes levitateLight {
        0% { transform: translateY(0px) scale(1); opacity: 0.8; }
        50% { transform: translateY(-10px) scale(1.05); opacity: 1; }
        100% { transform: translateY(0px) scale(1); opacity: 0.8; }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(0, 119, 255, 0.3), inset 0 0 10px rgba(0, 119, 255, 0.1); }
        50% { box-shadow: 0 0 35px rgba(0, 119, 255, 0.5), inset 0 0 20px rgba(0, 119, 255, 0.3); }
        100% { box-shadow: 0 0 15px rgba(0, 119, 255, 0.3), inset 0 0 10px rgba(0, 119, 255, 0.1); }
    }

    .stApp {
        background: linear-gradient(135deg, #ffffff, #e6f0fa, #f4f9ff, #d4e8ff);
        background-size: 400% 400%;
        animation: hyperdrive 12s ease infinite;
        color: #0a192f !important;
    }

    /* Partículas espaciales versión Light (Polvo de estrellas luminoso) */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: 
            radial-gradient(circle, rgba(0, 119, 255, 0.15) 1px, transparent 1px),
            radial-gradient(circle, rgba(0, 200, 255, 0.1) 2px, transparent 2px);
        background-size: 40px 40px, 60px 60px;
        background-position: 0 0, 20px 20px;
        animation: levitateLight 10s ease-in-out infinite alternate;
        z-index: 0;
        pointer-events: none;
    }

    .block-container {
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4, label {
        color: #0044aa !important; /* Azul profundo futurista Jedi */
        text-align: center;
        font-weight: 800 !important;
        letter-spacing: 3px;
        font-family: 'Outfit', sans-serif !important;
        text-shadow: 0 0 15px rgba(0, 119, 255, 0.3);
    }
    
    /* El Dropzone (Botón Mágico de subida holográfico) */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(10px);
        border: 2px dashed #0077ff !important;
        border-radius: 15px !important;
        padding: 50px !important;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        animation: pulseGlow 4s infinite;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-color: #00aaff !important;
        border-style: solid !important;
        transform: scale(1.03) translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 119, 255, 0.4), inset 0 0 25px rgba(0, 119, 255, 0.3) !important;
    }
    
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploadDropzone"] div[data-testid="stText"],
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] svg,
    [data-testid="stFileUploadDropzone"] span {
        display: none !important;
    }

    [data-testid="stFileUploadDropzone"]::before {
        content: "TRANSMITIR HOLOGRAMA";
        color: #0055ff !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 4px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        display: block;
        text-shadow: 0 0 8px rgba(0, 119, 255, 0.4);
    }

    .stButton > button {
        background: linear-gradient(45deg, #0077ff, #00ccff) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        letter-spacing: 3px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 5px 15px rgba(0, 119, 255, 0.4);
        padding: 15px !important;
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 119, 255, 0.6);
        background: linear-gradient(45deg, #0055dd, #00aaff) !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("SUDOKUADOR")

@st.cache_resource
def load_models():
    # Cargar YOLO
    try:
        yolo = YOLO(RUTA_MODELO_YOLO)
    except Exception as e:
        st.error(f"Error cargando YOLO: {e}")
        yolo = None
        
    # Cargar CNN
    try:
        cnn = LectorSudoku()
        cnn.load_state_dict(torch.load(RUTA_MODELO_CNN, map_location=torch.device('cpu'), weights_only=True))
        cnn.eval()
    except Exception as e:
        st.error(f"Error cargando CNN: {e}")
        cnn = None
        
    return yolo, cnn

modelo_yolo, lector = load_models()

if modelo_yolo is None or lector is None:
    st.warning("Los modelos no están disponibles. Revisa las rutas en variables.py.")
    st.stop()

uploaded_file = st.file_uploader("INICIALIZAR DESENCRIPTACIÓN DE IMAGEN", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen original
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imagen_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner("Procesando imagen con IA..."):
        # 1. Ejecutar YOLO
        resultados = modelo_yolo(imagen_cv)
        caja = resultados[0].boxes.xyxy.cpu().numpy()

        if len(caja) == 0:
            st.error("No se ha detectado ningún tablero de Sudoku en la imagen. Intenta con una foto más clara.")
            st.stop()

        # 2. Recortar y grises
        tablero_gris = extraer_tablero(imagen_cv, caja, TAMANO_TABLERO)
        
        mejor_score = -float('inf')
        mejor_tablero = tablero_gris
        mejores_resultados = None

        # 3. Probar rotaciones
        for angulo in [0, 90, 180, 270]:
            if angulo == 90:
                tablero_rot = cv2.rotate(tablero_gris, cv2.ROTATE_90_CLOCKWISE)
            elif angulo == 180:
                tablero_rot = cv2.rotate(tablero_gris, cv2.ROTATE_180)
            elif angulo == 270:
                tablero_rot = cv2.rotate(tablero_gris, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                tablero_rot = tablero_gris

            casillas = trocear_cuadricula(tablero_rot, 9, 9)

            matriz_detectada_tmp = np.zeros((9, 9), dtype=int)
            matriz_confianzas_tmp = np.zeros((9, 9), dtype=float)
            
            suma_confianzas = 0.0
            num_digitos = 0

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

                tensor_casilla = torch.tensor(lienzo_final, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                tensor_casilla = 1.0 - (tensor_casilla / 127.5)
                
                if tiene_digito and tensor_casilla.min().item() > -0.5:
                    tiene_digito = False
                
                if not tiene_digito:
                    numero_leido = 0
                else:
                    with torch.no_grad():
                        prediccion = lector(tensor_casilla)
                        _, pronostico = torch.max(prediccion, 1)
                        numero_leido = pronostico.item()
                        suma_confianzas += prediccion[0][numero_leido].item()
                        num_digitos += 1
                
                fila = idx // 9
                columna = idx % 9
                matriz_detectada_tmp[fila][columna] = numero_leido
                if tiene_digito:
                    matriz_confianzas_tmp[fila][columna] = prediccion[0][numero_leido].item()

            score_rotacion = (suma_confianzas / num_digitos) if num_digitos > 0 else -100.0

            if score_rotacion > mejor_score:
                mejor_score = score_rotacion
                mejor_tablero = tablero_rot
                mejores_resultados = (matriz_detectada_tmp, matriz_confianzas_tmp)
        
        # 4. Extraer mejores resultados y aplicar reglas de sudoku
        matriz_detectada, matriz_confianzas = mejores_resultados
        corregir_con_reglas_sudoku(matriz_detectada, matriz_confianzas)
        
        # Estado para saber si se ha pulsado resolver
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.solve_clicked = False
            st.session_state.last_uploaded = uploaded_file.name

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h4 style='text-align:center;'>1. Original</h4>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.markdown("<h4 style='text-align:center;'>2. Recortado</h4>", unsafe_allow_html=True)
        st.image(mejor_tablero, use_container_width=True, channels="GRAY")

    with col3:
        st.markdown("<h4 style='text-align:center;'>3. IA Resolución</h4>", unsafe_allow_html=True)
        
        # Contenedor para el botón o la animación
        action_container = st.empty()
        
        if not st.session_state.solve_clicked:
            # Mostrar botón centrado
            st.markdown("<br><br>", unsafe_allow_html=True) # Espaciado simulado para centrar
            if action_container.button("🚀 Resolver", type="primary", use_container_width=True):
                st.session_state.solve_clicked = True
                st.rerun()

        if st.session_state.solve_clicked:
            matriz_a_resolver = matriz_detectada.copy()
            matriz_original = matriz_detectada.copy()
            exito = resolver_sudoku(matriz_a_resolver)
            
            if exito:
                vacios = [(f, c) for f in range(9) for c in range(9) if matriz_original[f, c] == 0]
                current_anim_grid = matriz_original.copy()
                
                # Renderizar primero los detectados (efecto escaneo)
                html_table_res = "<table style='border-collapse: collapse; margin: auto; box-shadow: 0 0 25px rgba(0,119,255,0.3); background: rgba(255,255,255,0.8); backdrop-filter: blur(5px); border-radius: 10px; overflow: hidden;'>"
                for fi in range(9):
                    html_table_res += "<tr>"
                    for ci in range(9):
                        border_right = "2px solid #0077ff" if (ci+1)%3==0 and ci<8 else "1px solid rgba(0,119,255,0.2)"
                        border_bottom = "2px solid #0077ff" if (fi+1)%3==0 and fi<8 else "1px solid rgba(0,119,255,0.2)"
                        num = current_anim_grid[fi, ci]
                        color = "#0a192f" if num != 0 else "transparent"
                        text_disp = num if num != 0 else ""
                        html_table_res += f"<td style='border-right: {border_right}; border-bottom: {border_bottom}; width: 35px; height: 35px; text-align: center; color: {color}; font-weight: 800; font-size: 18px;'>{text_disp}</td>"
                    html_table_res += "</tr>"
                html_table_res += "</table>"
                action_container.markdown(html_table_res, unsafe_allow_html=True)
                
                time.sleep(0.5) # Pausa dramática para ver lo detectado
                
                # Animación Matrix de resolución
                for (f, c) in vacios:
                    current_anim_grid[f, c] = matriz_a_resolver[f, c]
                    
                    html_table_res = "<table style='border-collapse: collapse; margin: auto; box-shadow: 0 0 25px rgba(0,119,255,0.3); background: rgba(255,255,255,0.8); backdrop-filter: blur(5px); border-radius: 10px; overflow: hidden;'>"
                    for fi in range(9):
                        html_table_res += "<tr>"
                        for ci in range(9):
                            border_right = "2px solid #0077ff" if (ci+1)%3==0 and ci<8 else "1px solid rgba(0,119,255,0.2)"
                            border_bottom = "2px solid #0077ff" if (fi+1)%3==0 and fi<8 else "1px solid rgba(0,119,255,0.2)"
                            
                            num = current_anim_grid[fi, ci]
                            es_original = matriz_original[fi, ci] != 0
                            
                            if num == 0:
                                color = "transparent"
                                text_disp = ""
                                font_weight = "normal"
                            else:
                                color = "#0a192f" if es_original else "#0077ff"
                                font_weight = "800" if es_original else "600"
                                text_disp = num
                                
                            html_table_res += f"<td style='border-right: {border_right}; border-bottom: {border_bottom}; width: 35px; height: 35px; text-align: center; color: {color}; font-weight: {font_weight}; font-size: 18px;'>{text_disp}</td>"
                        html_table_res += "</tr>"
                    html_table_res += "</table>"
                    
                    action_container.markdown(html_table_res, unsafe_allow_html=True)
                    time.sleep(0.01)
                    
                st.toast("¡Sudoku resuelto y desencriptado con éxito!", icon="✅")
            else:
                action_container.error("El tablero extraído no tiene solución lógica. La IA pudo haber leído mal un número.")
