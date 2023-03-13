import cv2
import numpy as np

# Cargar configuración y pesos de YOLO, DNN es Deep Neural Network
config = 'model/yolov3.cfg'
weights = 'model/yolov3.weights'

# Cargar las etiquetas de COCO names, que son las clases que detecta YOLO
LABELS = open('model/coco.names').read().strip().split('\n')

# Cargar los colores para cada clase
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# Cargar el modelo de YOLO
net = cv2.dnn.readNetFromDarknet(config, weights)

# Cargar la imagen
image = cv2.imread('images/horse.jpg')

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
ln = net.getLayerNames()

# Obtener las capas de salida de YOLO
#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

net.setInput(blob)
layerOutputs = net.forward(ln)

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
        classID = np.argmax(scores) # Obtener el ID de la clase con mayor confianza
        confidence = scores[classID] # Obtener la confianza de la clase

        # Filtrar las detecciones débiles, descartando aquellas que tienen una confianza menor al 50%
        if confidence > 0.5:            
            # Escalar las coordenadas del cuadro delimitador para que coincidan con el tamaño de la imagen
            # YOLO devuelve las coordenadas del centro (x, y) del cuadro delimitador seguido por el ancho y alto
            # del cuadro delimitador
            box = detection[0:4] * np.array([w, h, w, h]) # Multiplicar las coordenadas por el tamaño de la imagen original
            (centerX, centerY, width, height) = box.astype('int')

            # Usar las coordenadas (x, y) para el cuadro delimitador
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Actualizar las listas de cuadros delimitadores, confianzas y IDs de clases
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Aplicar la supresión no máxima para eliminar cuadros delimitadores con mucha intersección
# y dejar solo los cuadros delimitadores con mayor confianza
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

if len(idxs) > 0:
    # Recorrer los índices que se mantienen después de aplicar la supresión no máxima
    for i in idxs.flatten():
        # Imrimir la etiqueta y la confianza en la consola
        print('detectado: ', LABELS[classIDs[i]], 'con confianza de: ', confidences[i])

        # Extraer las coordenadas (x, y) para el cuadro delimitador
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # Dibujar el cuadro delimitador y la etiqueta en la imagen
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


cv2.imshow("Imagen",image)
cv2.waitKey(0)
cv2.destroyAllWindows()