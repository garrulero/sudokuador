import numpy as np

def es_valido(tablero, fila, col, num):
    """
    Comprueba si es legal colocar un 'num' en la posición (fila, col) del tablero.
    Asume que 'tablero' es un array de NumPy de 9x9.
    """
    # 1. Comprobar la fila: si el número ya está en esa fila, no es válido
    if num in tablero[fila, :]:
        return False

    # 2. Comprobar la columna: si el número ya está en esa columna, no es válido
    if num in tablero[:, col]:
        return False

    # 3. Comprobar el cuadrante 3x3
    # Calculamos en qué índice empieza el cuadrante actual usando división entera (//)
    inicio_fila = (fila // 3) * 3
    inicio_col = (col // 3) * 3
    
    # Extraemos la submatriz 3x3
    cuadrante = tablero[inicio_fila : inicio_fila + 3, inicio_col : inicio_col + 3]
    
    if num in cuadrante:
        return False

    # Si pasa todas las pruebas, el movimiento es legal
    return True


def encontrar_vacio(tablero):
    """
    Recorre el tablero buscando la primera casilla que contenga un 0.
    Devuelve una tupla (fila, columna) o None si el tablero está lleno.
    """
    for f in range(9):
        for c in range(9):
            if tablero[f, c] == 0:
                return (f, c)
    return None


def resolver_sudoku(tablero):
    """
    Función recursiva que resuelve el Sudoku utilizando el algoritmo de Backtracking.
    Modifica el array 'tablero' in-place.
    Devuelve True si encuentra la solución, o False si se atasca.
    """
    # Paso 1: Buscar la siguiente casilla vacía
    vacio = encontrar_vacio(tablero)
    
    # Condición de parada: Si no hay vacíos, el Sudoku está resuelto
    if not vacio:
        return True
    
    fila, col = vacio

    # Paso 2: Probar números del 1 al 9 en la casilla vacía
    for num in range(1, 10):
        if es_valido(tablero, fila, col, num):
            
            # Si es válido, lo colocamos "temporalmente"
            tablero[fila, col] = num
            
            # Paso 3: Llamada recursiva (magia del Backtracking)
            # Volvemos a llamar a la función para intentar rellenar la siguiente casilla
            if resolver_sudoku(tablero):
                return True
            
            # Paso 4: Dar marcha atrás (Backtrack)
            # Si el camino que hemos tomado no lleva a la solución,
            # borramos el número (volvemos a poner 0) y el bucle probará el siguiente
            tablero[fila, col] = 0

    # Si probamos del 1 al 9 y ninguno funciona, este camino es un callejón sin salida
    return False


def corregir_con_reglas_sudoku(matriz, confianzas):
    """
    Usa las reglas del Sudoku para detectar conflictos (duplicados en fila, columna o caja)
    y corrige el dígito con MENOR confianza, reemplazándolo por 0 (vacío).

    Parámetros:
        matriz:      np.array 9x9 con los dígitos detectados (0 = vacío)
        confianzas:  np.array 9x9 con la confianza (logit máximo) de cada predicción
    
    Devuelve:
        correcciones: lista de strings describiendo cada corrección aplicada
    """
    correcciones = []
    
    def _resolver_conflicto(celdas_con_numero, numero, zona_nombre):
        """Si hay duplicados de 'numero' en una zona, borra el de menor confianza."""
        if len(celdas_con_numero) <= 1:
            return
        # Encontrar la celda con MENOR confianza
        peor = min(celdas_con_numero, key=lambda pos: confianzas[pos[0], pos[1]])
        f, c = peor
        conf = confianzas[f, c]
        matriz[f, c] = 0  # Marcar como vacío
        correcciones.append(
            f"Conflicto en {zona_nombre}: dos '{numero}' -> "
            f"Casilla ({f},{c}) borrada (confianza {conf:.2f})"
        )
    
    # 1. Comprobar FILAS
    for fila in range(9):
        for num in range(1, 10):
            posiciones = [(fila, c) for c in range(9) if matriz[fila, c] == num]
            _resolver_conflicto(posiciones, num, f"Fila {fila}")
    
    # 2. Comprobar COLUMNAS
    for col in range(9):
        for num in range(1, 10):
            posiciones = [(f, col) for f in range(9) if matriz[f, col] == num]
            _resolver_conflicto(posiciones, num, f"Columna {col}")
    
    # 3. Comprobar CAJAS 3x3
    for bf in range(3):
        for bc in range(3):
            for num in range(1, 10):
                posiciones = []
                for f in range(bf * 3, bf * 3 + 3):
                    for c in range(bc * 3, bc * 3 + 3):
                        if matriz[f, c] == num:
                            posiciones.append((f, c))
                _resolver_conflicto(posiciones, num, f"Caja ({bf},{bc})")
    
    return correcciones