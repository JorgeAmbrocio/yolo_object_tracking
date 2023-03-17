# Importar el analizador de video
from video_analyzer import VideoAnalyzer

# Inicializar el analizador de video
config_path = 'object_detection_on_image\\model\\yolov3.cfg'
weights_path = 'object_detection_on_image\\model\\yolov3.weights'
labels_path = 'object_detection_on_image\\model\\coco.names'
video_analyzer = VideoAnalyzer(config_path, weights_path, labels_path)

# Analizar el video
video_path = 'object_detection_on_video\\videos\\pasillo_Trim_1.mp4'
video_path = 'object_detection_on_video\\videos\\pasillo_exterior_1.mp4'
video_analyzer.analyze(video_path)