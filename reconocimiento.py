# ***********************************************************************
# Programa de Detección de Personas con Intel RealSense para Kria KV260
# ***********************************************************************
# Descripción:
# Este programa utiliza una cámara Intel RealSense para capturar imágenes RGB
# y de profundidad, y un modelo de detección de objetos entrenado para reconocer
# personas específicas. Dependiendo de la persona reconocida, se activa un LED
# o un zumbador. El modelo se ejecuta en una DPU (Deep Processing Unit) utilizando
# la biblioteca PYNQ.
#
# Requisitos:
# - Python 3.x
# - PYNQ
# - OpenCV
# - pyrealsense2
# - numpy
# - matplotlib
# - colorsys
# - random
#
# Configuración:
# - Asegúrate de tener instaladas las bibliotecas necesarias:
#   pip install pyrealsense2 open3d numpy opencv-python matplotlib
# - Conecta la cámara Intel RealSense y asegúrate de que esté funcionando correctamente.
# - Configura el modelo y los archivos necesarios en las rutas especificadas.
# - Conecta los dispositivos GPIO para el LED y el zumbador.
#
# Ejecución:
# - Ejecuta el programa y observa la salida en la ventana de OpenCV.
# - Presiona 'q' para salir del programa.
#
# ***********************************************************************
# ***********************************************************************
# Import Packages
# ***********************************************************************
import os
import time
import numpy as np
import cv2
import random
import colorsys
import pyrealsense2 as rs
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from pynq_dpu import DpuOverlay
from pynq import Overlay
from pynq.lib import AxiGPIO

# ***********************************************************************
# input file names
# ***********************************************************************
dpu_model   = os.path.abspath("dpu.bit")
cnn_xmodel  = os.path.join("./"        , "modelo.xmodel") #modelo optimizado
labels_file = os.path.join("./img"     , "personas.txt") # nombre de las personas a reconocer + otra persona
# ***********************************************************************
# Prepare the Overlay and load the "cnn.xmodel"
# ***********************************************************************
overlay = DpuOverlay(dpu_model)
overlay.load_model(cnn_xmodel)
ol = overlay

# ***********************************************************************
# Utility Functions
# ***********************************************************************
def preprocess(image, input_size, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_image = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_image = np.ones(input_size, dtype=np.uint8) * 114

    ratio = min(input_size[0] / image.shape[0],
                input_size[1] / image.shape[1])
    resized_image = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = resized_image.astype(np.uint8)

    padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                    ratio)] = resized_image
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, ratio

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def postprocess(
    outputs,
    img_size,
    ratio,
    nms_th,
    nms_score_th,
    max_width,
    max_height,
    p6=False,
):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = sigmoid(predictions[:, 4:5]) * softmax(predictions[:, 5:])
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=nms_th,
        score_thr=nms_score_th,
    )

    bboxes, scores, class_ids = [], [], []
    if dets is not None:
        bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for bbox in bboxes:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], max_width)
            bbox[3] = min(bbox[3], max_height)

    return bboxes, scores, class_ids

def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(
    boxes,
    scores,
    nms_thr,
    score_thr,
    class_agnostic=True,
):
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware

    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr

        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [
                        valid_boxes[keep], valid_scores[keep, None],
                        cls_inds
                    ],
                    1,
                )
                final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr,
                                    score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr

    if valid_score_mask.sum() == 0:
        return None

    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)

    dets = None
    if keep:
        dets = np.concatenate([
            valid_boxes[keep],
            valid_scores[keep, None],
            valid_cls_inds[keep, None],
        ], 1)

    return dets

def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
class_names = get_class(labels_file)
num_classes = len(class_names)

hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
              

colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

random.seed(0)
random.shuffle(colors)
random.seed(None)

def draw_bbox(image, bboxes, classes):
    image_h, image_w, _ = image.shape
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(1.8 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image

# ***********************************************************************
# Use VART APIs
# ***********************************************************************

dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)
shapeOut0 = (tuple(outputTensors[0].dims))
shapeOut1 = (tuple(outputTensors[1].dims))
shapeOut2 = (tuple(outputTensors[2].dims))
outputSize0 = int(outputTensors[0].get_data_size() / shapeIn[0])
outputSize1 = int(outputTensors[1].get_data_size() / shapeIn[0])
outputSize2 = int(outputTensors[2].get_data_size() / shapeIn[0])
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"), 
               np.empty(shapeOut1, dtype=np.float32, order="C"),
               np.empty(shapeOut2, dtype=np.float32, order="C")]
image = input_data[0]

def run(input_image, section_i, display=False):
    input_shape=(416, 416)
    class_score_th=0.3
    nms_th=0.45
    nms_score_th=0.1

    image_size = input_image.shape[:2]
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    image_data, ratio = preprocess(input_image, input_shape)
    
    image[0,...] = image_data.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    outputs = np.concatenate([output.reshape(1, -1, output.shape[-1]) for output in output_data], axis=1)
    bboxes, scores, class_ids = postprocess(
        outputs,
        input_shape,
        ratio,
        nms_th,
        nms_score_th,
        image_width,
        image_height,
    )
    
    bboxes_with_scores_and_classes = []
    for i in range(len(bboxes)):
        bbox = bboxes[i].tolist() + [scores[i], class_ids[i]]
        bboxes_with_scores_and_classes.append(bbox)
    bboxes_with_scores_and_classes = np.array(bboxes_with_scores_and_classes)
    display = draw_bbox(input_image, bboxes_with_scores_and_classes, class_names)

    # Activar LED o zumbador según la persona reconocida
    if 0 in class_ids:  # Suponiendo que la persona específica tiene la clase ID 0
        gpio_out.write(0x4, mask)  # Encender LED
    elif 1 in class_ids:  # Suponiendo que otra persona tiene la clase ID 1
        gpio_out.write(0x2, mask)  # Encender zumbador
    else:
        gpio_out.write(0x0, mask)  # Apagar todos

# Configuración de la cámara Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# LED(GPIO)_set
gpio_0_ip = ol.ip_dict['axi_gpio_0']
gpio_out = AxiGPIO(gpio_0_ip).channel1
mask = 0xffffffff

# Inicializar variables para el cálculo de FPS promedio
frame_count = 0
avg_start_time = time.time()

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        start_time = time.time()
        height, width, _ = color_image.shape

        sections = [color_image]

        for i, section in enumerate(sections):
            run(section, i+1, display=True)
            if display:
                cv2.imshow(f"Section {i+1}", section)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            gpio_out.write(0x00, mask)  # Apagar todos los GPIO
            break

        end_time = time.time()
        print("Total run time: {:.4f} seconds".format(end_time - start_time))
        print("Performance: {} FPS".format(1/(end_time - start_time)))
        print(" ")

        frame_count += 1
        if frame_count % 100 == 0:
            avg_end_time = time.time()
            elapsed_time = avg_end_time - avg_start_time
            fps = frame_count / elapsed_time
            print(" ")
            print("Avg_FPS:", fps)
            print(" ")
            frame_count = 0
            avg_start_time = time.time()

finally:
    pipeline.stop()
    cap.release()
    cv2.destroyAllWindows()

# ***********************************************************************
# Clean up
# ***********************************************************************
del overlay
del dpu
