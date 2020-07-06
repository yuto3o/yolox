

# More Than YOLO

English | [中文](./README_CN.md)

TensorFlow & Keras & Python

YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny

[**Unofficial**] YOLOv4-tiny, YOLOX

**requirements:** TensorFlow 2.1 (not test on 1.x), OpenCV, Numpy, PyYAML

---

## News !

- Small batch size is used, because available GPU (8G) has small memory.  Please use big batch size as possible.
- Online High Level augmentation will slow down training speed.
- When I tried to train yolov3 or yolov4, 'NaN' problem made me crazy.
- Support AccumOptimizer, Similar to  'subdivisions' in darknet.
- When I tried to train yolov3 or yolov4, I found that if I set weight decay to 5e-4，the result is unsatisfactory; if I set it to 0, everything is OK.
- Note that both yolov3-tiny and yolov4-tiny don't use anchor 0, so they use only anchors 1-7.[[link]](https://github.com/hunglc007/tensorflow-yolov4-tflite/issues/111)
---

This repository have done:

- [x] Backbone for YOLO (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny, Unofficial-YOLOv4-tiny)
- [x] YOLOv3 Head
- [x] Keras Callbacks for Online Evaluation
- [x] Load Official Weight File
- [x] Data Format Converter(COCO and Pascal VOC)
- [x] K-Means for Anchors
- [x] Fight with 'NaN'
- [x] Train (Strategy and Model Config)
  - Define simple training in [train.py](./train.py)
  - Use YAML as config file in [cfgs](./cfgs)
  - [x] Cosine Annealing LR
  - [x] Warm-up
  - [x] AccumOptimizer
- [ ] Data Augmentation
  - [x] Standard Method: Random Flip, Random Crop, Zoom, Random Grayscale, Random Distort, Rotate
  - [x] Hight Level: Cut Mix, Mix Up, Mosaic （These Online Augmentations is Slow）
  - [ ] More, I can be so much more ... 
- [ ] For Loss
  - [x] Label Smoothing
  - [x] Focal Loss
  - [x] L2, D-IoU, G-IoU, C-IoU
  - [ ] ...

---

[toc]

## 0. Please Read Source Code for More Details

You can get official weight files from https://github.com/AlexeyAB/darknet/releases or https://pjreddie.com/darknet/yolo/.

## 1. Samples

### 1.1 Data File

We use a special data format like that,

```txt
path/to/image1 x1,y1,x2,y2,label x1,y1,x2,y2,label 
path/to/image2 x1,y1,x2,y2,label 
...
```

Convert your data format firstly. We present [a script for Pascal VOC](./data/pascal_voc/voc_convert.py) and [a script for COCO](./data/coco/coco_convert.py).

More details and a simple dataset could be got from https://github.com/YunYang1994/yymnist.

### 1.2 Configure 

```yaml
# coco_yolov4.yaml
yolo:
  type: "yolov4"  # must be 'yolov3', 'yolov3_tiny', 'yolov4', 'yolov4_tiny' ‘unofficial_yolov4_tiny’ and 'yolox'.
  iou_threshold: 0.5
  score_threshold: 0.005
  max_boxes: 100
  strides: "32,16,8"
  anchors: "12,16 19,36 40,28 36,75 76,55 72,146 142,110 192,243 459,401"
  mask: "6,7,8 3,4,5 0,1,2"
  name_path: "./data/coco/coco.name"

train:
  label: "coco_yolov4" # any thing you like
  anno_path: "./data/coco/train2017.txt"
  image_size: "320,352,384,416,448,480,512,544,576,608"  # "416" for single mini batch size, "352,384,416,448,480" for Dynamic mini batch size.

  batch_size: 4
  init_weight_path: "./ckpts/yolov4.weights"
  save_weight_path: "./ckpts"

  loss_type: "CIoU+FL" # Must be "L2", "DIoU", "GIoU", "CIoU" or something like "L2+FL" for focal loss
  
  # turn on hight level data augmentation
  mosaic: false
  label_smoothing: false
  normal_method: true

  ignore_threshold: 0.7

test:
  anno_path: "./data/coco/val2017.txt"
  image_size: "608"
  batch_size: 1
  init_weight_path: "./ckpts/yolov4.weights"
```

### 1.3 K-Means

Edit the kmeans.py as you like

```python
# kmeans.py
# Key Parameters
K = 6 # num of clusters
image_size = 416
dataset_path = './data/pascal_voc/train.txt'
```

### 1.4 Inference

#### A Simple Script for Video, Device or Image

Only support mp4, avi, device id, rtsp, png, jpg (Based on OpenCV) 

![gif](./misc/street.gif)

```shell
python detector.py --config=./cfgs/coco_yolov4.yaml --media=./misc/street.mp4 --gpu=false
```

#### A simple demo for YOLOv4

![yolov4](./misc/dog_v4.jpg)

```python
from core.utils import decode_cfg, load_weights
from core.model.one_stage.yolov4 import YOLOv4
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, Shader

import numpy as np
import cv2
import time

# read config
cfg = decode_cfg('cfgs/coco_yolov4.yaml')
names = cfg['yolo']['names']

model, eval_model = YOLOv4(cfg)
eval_model.summary()

# assign colors for difference labels
shader = Shader(cfg['yolo']['num_classes'])

# load weights
load_weights(model, cfg['test']['init_weight_path'])

img_raw = read_image('./misc/dog.jpg')
img = preprocess_image(img_raw, (512, 512))
imgs = img[np.newaxis, ...]

tic = time.time()
boxes, scores, classes, valid_detections = eval_model.predict(imgs)
toc = time.time()
print((toc - tic)*1000, 'ms')

# for single image, batch size is 1
valid_boxes = boxes[0][:valid_detections[0]]
valid_score = scores[0][:valid_detections[0]]
valid_cls = classes[0][:valid_detections[0]]

img, valid_boxes = postprocess_image(img, img_raw.shape[1::-1], valid_boxes)
img = draw_bboxes(img, valid_boxes, valid_score, valid_cls, names, shader)

cv2.imshow('img', img[..., ::-1])
cv2.imwrite('./misc/dog_v4.jpg', img)
cv2.waitKey()
```

## 2. Train

!!! Please Read the above guide (e.g. 1.1, 1.2).

```shell
python train.py --config=./cfgs/coco_yolov4.yaml
```

## 3. Experiment

### 3.1 Speed

**i7-9700F+16GB**

| Model       | 416x416 | 512x512 | 608x608 |
| ----------- | ------- | ------- | ------- |
| YOLOv3      | 219 ms  | 320 ms  | 429 ms  |
| YOLOv3-tiny | 49 ms   | 63 ms   | 78 ms   |
| YOLOv4      | 344 ms  | 490 ms  | 682 ms  |
| YOLOv4-tiny | 51 ms   | 66 ms   | 83 ms  |
| Unofficial-YOLOv4-tiny | 64 ms   | 86 ms   | 110 ms  |
| YOLOX | 67 ms | 83 ms | 104 ms |

**i7-9700F+16GB / RTX 2070S+8G**

| Model       | 416x416 | 512x512 | 608x608 |
| ----------- | ------- | ------- | ------- |
| YOLOv3      | 59 ms   | 66 ms   | 83 ms   |
| YOLOv3-tiny | 28 ms   | 30 ms   | 33 ms   |
| YOLOv4      | 73 ms   | 74 ms   | 91 ms   |
| YOLOv4-tiny | 30 ms   | 32 ms   | 35 ms  |
| Unofficial-YOLOv4-tiny | 30 ms   | 31 ms   | 34 ms   |
| YOLOX | 42 ms | 45 ms | 50 ms |

### 3.2 Logs

**Augmentations**

| Name                    | Abbr |
| ----------------------- | ---- |
| Standard Method         | SM   |
| Dynamic mini batch size | DM   |
| Label Smoothing         | LS   |
| Focal Loss              | FL   |
| Mosaic                  | M    |
| Warm-up LR              | W    |
| Cosine Annealing LR     | CA   |

Standard Method Package includes Flip left and right,  Crop and Zoom(jitter=0.3), Grayscale, Distort, Rotate(angle=7).

**YOLOv3-tiny**(Pretrained on COCO; Trained on VOC)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    |      |      |      |      | L2   | 26.6 | 61.8 | 17.2  |
| ✔    | ✔    |      |      |      | L2   | 27.3 | 62.4 | 17.9  |
| ✔    | ✔    | ✔    |      |      | L2   | 26.7 | 61.7  | 17.1 |
| ✔    | ✔    |      |      |      | CIoU | 30.9 | 64.2  | 25.0 |
| ✔    | ✔    |      | ✔    |      | CIoU | 32.3 | 65.7 | 27.6  |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |  |  |  |

**YOLOv3**(TODO; Pretrained on COCO; Trained on VOC; only 15 epochs)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    | ✔    |      | ✔    |      | CIoU | 46.5 | 80.0  | 49.0  |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |      |       |       |

**YOLOv4-tiny**(TODO; Pretrained on COCO;  Trained on VOC)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    | ✔    |      | ✔    |      | CIoU | 35.1 | 70.2  | 30.0  |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |      |       |       |

**YOLOv4**(TODO; Pretrained on COCO;  Trained on VOC)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    | ✔    |      | ✔    |      | CIoU |      |       |       |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |      |       |       |

**Unofficial-YOLOv4-tiny**(TODO; Pretrained on COCO, part of YOLOv3-tiny weights; Trained on VOC)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    | ✔    |      | ✔    |      | CIoU | 35.0 | 65.7  | 33.8  |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |      |       |       |

**YOLOX**(TODO; Pretrained on COCO, part of YOLOv4-tiny weights; Trained on VOC)

| SM   | DM   | LS   | FL   | M    | Loss | AP   | AP@50 | AP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- |
| ✔    | ✔    |      | ✔    |      | CIoU | 40.6 | 72.2  | 40.3  |
| ✔    | ✔    |      | ✔    | ✔    | CIoU |      |       |       |

### 3.3 Details

#### Tiny Version

| Stage | Freeze Backbone | LR                   | Steps   |
| ----- | --------------- | -------------------- | ------- |
| 0     | Yes             | 1e-3 (w/ W)          | 4000    |
| 1     | Yes             | -                    | 32*4000 |
| 2     | No              | 1e-3 to 1e-6 (w/ CA) | 48*4000 |

#### Common Version

| Stage | Freeze Backbone | LR                   | Steps    |
| ----- | --------------- | -------------------- | -------- |
| 0     | Yes             | 1e-3 (w/ W)          | 4000     |
| 1     | Yes             | -                    | 80*4000  |
| 2     | No              | 1e-3 to 1e-6 (w/ CA) | 120*4000 |

Training a complete network is time-consuming ...

## 4. Reference

- https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
- https://github.com/hunglc007/tensorflow-yolov4-tflite
- https://github.com/experiencor/keras-yolo3

## 5. History

- Slim version: https://github.com/yuto3o/yolox/tree/slim
- Tensorflow2.0-YOLOv3: https://github.com/yuto3o/yolox/tree/yolov3-tf2
