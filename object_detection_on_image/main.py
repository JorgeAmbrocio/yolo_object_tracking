# importar la clase de la aplicación
from app_class_mode import ObjectDetection

# Inicializar la aplicación
config_path = 'model\yolov3.cfg'
weights_path = 'model\yolov3.weights'
labels_path = 'model\coco.names'
app = ObjectDetection(config_path, weights_path, labels_path)

# Detectar objetos en la imagen
image_path = 'images\horse.jpg'
boxes, confidences, classIDs, idxs = app.detect(image_path)

# Mostrar la imagen con los objetos detectados
image = app.draw(image_path, boxes, confidences, classIDs, idxs)

# Mostrar la imagen
app.show(image)