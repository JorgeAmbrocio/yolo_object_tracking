import cv2
import numpy as np

# Clase para detectar objetos en una imagen
class ObjectDetection:
    # Constructor
    def __init__(self, config_path, weights_path, labels_path):
        # Cargar configuración y pesos de YOLO, DNN es Deep Neural Network
        self.config = config_path
        self.weights = weights_path

        # Cargar las etiquetas de COCO names, que son las clases que detecta YOLO
        self.LABELS = open(labels_path).read().strip().split('\n')

        # Cargar los colores para cada clase
        self.colors = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype='uint8')

        # Cargar el modelo de YOLO
        self.net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
    
    # Método para detectar objetos en una imagen
    def detect(self, image): # image_path
        #image = cv2.imread(image_path)

        # Obtener las dimensiones de la imagen
        (h, w) = image.shape[:2]

        # Crar un blob de la imagen, blob es un contenedor de imágenes
        # blobFromImage(image, scalefactor, size, mean, swapRB, crop)
        # scalefactor: escala de la imagen
        # size: tamaño de la imagen en pixeles
        # swapRB: cambiar de BGR a RGB, Red Green Blue, el sistema usa por defecto BGR, por eso intercambiamos los canales R y B
        # crop: recortar la imagen al terminar de procesarla
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Pasar el blob a través de la red y obtener las detecciones y predicciones
        ln = self.net.getLayerNames()

        # Obtener las capas de salida de YOLO
        #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        # Inicializar las listas de detecciones, confianzas y cuadros delimitadores
        boxes = []
        confidences = []
        classIDs = []

        # Recorrer cada capa de salida
        for output in layerOutputs:
            # Recorrer cada detección
            for detection in output:
                # Extraer la clase ID y la confianza (probabilidad) de la detección
                scores = detection[5:] # Las primeras 5 posiciones son las coordenadas del cuadro delimitador
                classID = np.argmax(scores)
                confidence = scores[classID]

                # Filtrar las detecciones débiles, asegurando que la confianza sea mayor que la confianza mínima
                if confidence > 0.5:
                    # Escalar las coordenadas del cuadro delimitador de la imagen a las dimensiones de la imagen
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype('int')

                    # Usar las coordenadas del centro (x, y) para ubicar el cuadro delimitador
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Actualizar las listas de cuadros delimitadores, confianzas y clases
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        # Aplicar la supresión no máxima para eliminar los cuadros delimitadores débiles y superpuestos
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # Devovler las detecciones
        return boxes, confidences, classIDs, idxs
    

    # Función para dibujar los cuadros delimitadores y las etiquetas en la imagen
    def draw(self, image, boxes, confidences, classIDs, idxs):
        # image = cv2.imread(image_path)
        # Si hay al menos una detección
        if len(idxs) > 0:
            # Recorrer los índices de las detecciones
            for i in idxs.flatten():
                # Extraer las coordenadas del cuadro delimitador para la detección, luego dibujar el cuadro delimitador
                # y la etiqueta en la imagen
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Devolver la imagen con los cuadros delimitadores y las etiquetas
        return image
    
    # Función para mostrar la imagen
    def show(self, image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()