import cv2

# Importar detección de objetos desde la carpeta object_detection_on_image
import sys
sys.path.append('object_detection_on_image')
from app_class import ObjectDetection as od

# Clase que se encarga de analizar el video y detectar los objetos
class VideoAnalyzer:
    # Constructor
    def __init__(self, config_path, weights_path, labels_path):
        # Inicializar la aplicación
        self.od = od(config_path, weights_path, labels_path)
    
    # Método para analizar el video
    def analyze(self, video_path):
        # Leer el video
        cap = cv2.VideoCapture(video_path)
        
        # Leer el primer frame
        ret, frame = cap.read()
        
        # Mientras haya frames
        index = 0
        while ret:
            index += 1
            if index % 25 != 0:
                ret, frame = cap.read()
                continue

            # Reducir el tamaño del frame al 50%
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Detectar los objetos en el frame
            boxes, confidences, classIDs, idxs = self.od.detect(frame)
            
            # Dibujar los cuadros delimitadores
            frame = self.od.draw(frame, boxes, confidences, classIDs, idxs)
            
            # Mostrar el frame
            cv2.imshow('Frame', frame)
            
            # Leer el siguiente frame
            ret, frame = cap.read()
            
            # Si se presiona la tecla ESC, salir
            if cv2.waitKey(37) & 0xFF == ord('q'):
                break
        
        # Liberar la captura y cerrar las ventanas
        cap.release()
        cv2.destroyAllWindows()
