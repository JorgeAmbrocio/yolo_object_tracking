import torch 
import torch.nn as nn
import yaml
import cv2
import numpy as np
import PIL
import math

# imports from yolov6
#from yolov6.utils.torch_utils import fuse_model
#from yolov6.utils.nms import non_max_suppression
#from yolov6.core.inferer import Inferer
from torch_utils import fuse_model
from nms import non_max_suppression

from typing import List, Optional

class DetectBackend(nn.Module):
    def __init__(self, weights='yolov6s.pt', device=None, dnn=True):
        super().__init__()
        model = load_checkpoint(weights, map_location=device)
        stride = int(model.stride.max())
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):
        y, _ = self.model(im)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return y


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    #LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        #LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model

def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)
    
def process_image_url(path, img_size, stride, half):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    img_src = np.asarray(Image.open(path))
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    print(e)

  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src

def process_image(img_src, img_size, stride, half):
  '''Process image before image inference.'''
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src

def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0].clamp_(0, target_shape[1])  # x1
    boxes[:, 1].clamp_(0, target_shape[0])  # y1
    boxes[:, 2].clamp_(0, target_shape[1])  # x2
    boxes[:, 3].clamp_(0, target_shape[0])  # y2

    return boxes

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color

def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
        
def inference_url(url, model, img_size, stride, half, device):
    """Inference image."""
    img_size = check_img_size(img_size, s=stride)

    img, img_src = process_image_url(url, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes:Optional[List[int]] = None # the classes to keep
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round() # Inferer.
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)
            label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
            plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=generate_colors(class_num, True)) # Inferer.
    img_out = PIL.Image.fromarray(img_ori)
    return img_out

def inference_image(image, model, img_size, stride, half, device):
    """Inference image."""
    img_size = check_img_size(img_size, s=stride)

    img, img_src = process_image(image, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes:Optional[List[int]] = None # the classes to keep
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    if len(det):
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round() # Inferer.
        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)
            label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
            plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=generate_colors(class_num, True)) # Inferer.
    img_out = PIL.Image.fromarray(img_ori)
    return img_out

def load_model(weights, device_, dnn):
    """Load model."""
    # cargar el modelo
    cuda = device_ != 'cpu' and torch.cuda.is_available()
    device_ = torch.device(f'cuda:{device_}' if cuda else 'cpu')

    model = DetectBackend(weights=weights, device=device_, dnn=False)
    class_names = load_yaml("./data/coco.yaml")['names']
    stride = model.stride
    model.model.float()
    half = False
    
    return model, class_names, stride, half, device_

def trabajar_image_url(model):
    # cargar el modelo
    #model, class_names, stride, half, device = load_model(weights='yolov6n.pt', device='cpu', dnn= False)

    url:str = "./data/images/image1.jpg"  #"https://i.imgur.com/1IWZX69.jpg" #@param {type:"string"}
    img_size:int = 640#@param {type:"integer"}

    # Inference image with url
    img_prossesed = inference_url(url, model, img_size, stride, half, device)
    img_prossesed.show()

def trabajar_image(model):
    # dirección del video y de la imágen de prueba
    image_path = 'data/images/image2.jpg'
    
    # cargar el modelo
    #model, class_names, stride, half, device = load_model(weights='yolov6n.pt', device='cpu', dnn= False)

    img_size:int = 640#@param {type:"integer"}

    # Inference image with image
    image = np.asarray(PIL.Image.open(image_path))
    #image.show()
    img_prossesed = inference_image(image, model, img_size, stride, half, device)
    img_prossesed.show()

def trabajar_video(model):
    # dirección del video y de la imágen de prueba
    video_path = 'street.mp4'
    
    img_size:int = 640#@param {type:"integer"}

    # Obtener el video desde la ruta
    video = cv2.VideoCapture(video_path)
    # Obtener el ancho y el alto del video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Obtener el número de frames del video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Obtener el FPS del video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # ciclo que recorre cada frame del video
    ret, frame = video.read()
    counter = 0
    while ret:
        
        # leer el frame
        ret, frame = video.read()
        
        if not ret:
            break

        counter += 1
        if counter % 3 != 0:
            continue

        # reducir el tamaño del frame a 0.5
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.35)
        #frame = cv2.resize(frame, (320, 320))
        

        # Inference video
        img_prossesed = inference_image(frame, model, img_size, stride, half, device)
        # mostrar el frame
        #img_prossesed_resized = cv2.resize(np.asarray(img_prossesed), (0,0), fx=2, fy=2)
        cv2.imshow('frame', np.asarray(img_prossesed))

        # esperar 1 milisegundo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # dirección del video y de la imágen de prueba
    video_path = 'street.mp4'
    image_path = 'data/images/image2.jpg'
    
    url:str = "./data/images/image1.jpg"  #"https://i.imgur.com/1IWZX69.jpg" #@param {type:"string"}
    hide_labels: bool = False #@param {type:"boolean"}
    hide_conf: bool = False #@param {type:"boolean"}
    img_size:int = 640#@param {type:"integer"}

    conf_thres: float =.25 #@param {type:"number"}
    iou_thres: float =.45 #@param {type:"number"}
    max_det:int =  1000#@param {type:"integer"}
    agnostic_nms: bool = False #@param {type:"boolean"}
    str_device: str = '0' #@param {type:"string"}
    str_weights: str = 'yolov6n.pt' #@param {type:"string"}

    # cargar el modelo
    model, class_names, stride, half, device = load_model(weights=str_weights, device_=str_device, dnn= False)

    trabajar_video(model)

