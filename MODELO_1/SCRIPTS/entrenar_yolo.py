from ultralytics import YOLO

# Cargar el modelo base ligero de YOLOv8
model = YOLO('yolov8n.pt') 

# Entrenar el modelo con nuestro SUPER DATASET
# Usamos la ruta absoluta al yaml que acabas de crear
ruta_yaml = r"C:\BOOT\GARRULERO\SUDOKUADOR\MODELO_1\DATA\dataset_fusionado\data.yaml"

# Iniciamos el entrenamiento
resultados = model.train(data=ruta_yaml, epochs=25, imgsz=640)

print("¡Entrenamiento finalizado! Tu cerebro artificial está listo.")