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