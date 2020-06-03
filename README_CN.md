

# More Than YOLO

TensorFlow & Keras Implementations & Python

YOLOv3, YOLOv3-tiny, YOLOv4

YOLOv4-tiny（YOLOv4未提出，非官方）

**requirements:** TensorFlow 2.x (not test on 1.x), OpenCV, Numpy, PyYAML

---

## 被训练支配的恐惧 !

- 因为目前可用的卡是一张游戏卡RzaiTX 2070S(8 G)，因此在训练时使用了较小的batch size。
- 本项目的数据增强均使用在线形式，高级的数据增强方式会大大拖慢训练速度。
- 训练过程中，Tiny版问题不大，而完整版模型容易NaN或者收敛慢，还在调参中。

---

This repository have done:

- [x] Backbone for YOLO (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny[unofficial])
- [x] YOLOv3 Head
- [x] Keras Callbacks for Online Evaluation
- [x] Load Official Weight File
- [x] K-Means for Anchors
- [x] Fight with 'NaN'
- [x] Train (Strategy and Model Config)
  - Define simple training in [train.py](./train.py)
  - Use YAML as config file in [cfgs](./cfgs)
  - [x] Cosine Annealing LR
  - [x] Warm-up
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

## 0. 在提问前直接先看源码可更好帮助理解

相关的darknet权重可从官方渠道获取： https://github.com/AlexeyAB/darknet/releases 或者 https://pjreddie.com/darknet/yolo/.

## 1. Samples

### 1.1 Data File

本项目使用了不同于VOC和COCO的数据存储格式：

```txt
path/to/image1 x1,y1,x2,y2,label x1,y1,x2,y2,label 
path/to/image2 x1,y1,x2,y2,label 
...
```

当然本项目也提供了一个简单的VOC格式[转换脚本](./data/pascal_voc/voc_convert.py)。

也可以从其他大佬的项目中看到这种格式的运用，甚至可以得到一个简单的入门级目标检测数据集： https://github.com/YunYang1994/yymnist.

### 1.2 Configure 

```yaml
# voc_yolov3_tiny.yaml
yolo:
  type: "yolov3_tiny" # 当前只能是 'yolov3', 'yolov3_tiny', 'yolov4', 'yolov4_tiny'
  iou_threshold: 0.45
  score_threshold: 0.5
  max_boxes: 100
  strides: "32,16"
  anchors: "10,14 23,27 37,58 81,82 135,169 344,319"
  mask: "3,4,5 0,1,2"

train:
  label: "voc_yolov3_tiny" # 决定了LOG的根目录名，比较随意
  name_path: "./data/pascal_voc/voc.name"
  anno_path: "./data/pascal_voc/train.txt"
  # 当甚至为单一值时，比如"416"，表示使用单一图像训练尺度； 而"352,384,416,448,480" 则使用动态多尺度训练策略。
  image_size: "416" 

  batch_size: 4
  # 在载入权重前，你需要尽量保证网络结果一致，特别是darknet权重；而使用keras权重时，支持按层名导入。如果你想在官方COCO权重的基础上训练，可以直接使用COCO的网络配置，或者是先将darknet权重转为keras形式（只需向网络载入一次darknet权重，再保存权重就完成了转换）。
  init_weight_path: "./ckpts/yolov3-tiny.h5"
  save_weight_path: "./ckpts"

  # 支持 "L2", "DIoU", "GIoU", "CIoU"，或者以+分格的"L2+FL"开启Focal Loss
  loss_type: "L2" 
  
  # 一些策略的开关
  mix_up: false
  cut_mix: false
  mosaic: false
  label_smoothing: false
  normal_method: true

  ignore_threshold: 0.5

test:
  anno_path: "./data/pascal_voc/test.txt"
  image_size: "416" # 验证模型时的图像尺寸
  batch_size: 1 # 占位，还不支持
  init_weight_path: ""
```

### 1.3 K-Means

简单的编辑这一些超参，

```python
# kmeans.py
# Key Parameters
K = 6 # num of clusters
image_size = 416
dataset_path = './data/pascal_voc/train.txt'
```

### 1.4 Inference

#### 简单的面向图像、设备以及视频的测试脚本

目前只支持格式 mp4, avi, device id, rtsp, png, jpg (基于OpenCV) 

![gif](./misc/street.gif)

```shell
python detector.py --config=./cfgs/coco_yolov4.yaml --media=./misc/street.mp4 --gpu=false
```

#### YOLOv4的简单测试

![yolov4](./misc/dog_v4.jpg)

```python
from core.utils import decode_cfg, load_weights
from core.model.one_stage.yolov4 import YOLOv4
from core.image import draw_bboxes, preprocess_image, preprocess_image_inv, read_image, Shader

import numpy as np
import cv2
import time

# read config
cfg = decode_cfg('cfgs/coco_yolov4.yaml')

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

valid_boxes *= 512
img, valid_boxes = preprocess_image_inv(img, img_raw.shape[1::-1], valid_boxes)
img = draw_bboxes(img, valid_boxes, valid_score, valid_cls, names, shader)

cv2.imshow('img', img[..., ::-1])
cv2.imwrite('./misc/dog_v4.jpg', img)
cv2.waitKey()
```

## 2. Train

!!! 请先阅读上一节的内容 (e.g. 1.1, 1.2).

```shell
python train.py --config=./cfgs/voc_yolov4.yaml
```

## 3. Experiment

### 3.1 Speed

**i7-9700F+16GB**

| Model       | 416x416 | 512x512 | 608x608 |
| ----------- | ------- | ------- | ------- |
| YOLOv3      |         |         |         |
| YOLOv3-tiny |         |         |         |
| YOLOv4      |         |         |         |
| YOLOv4-tiny |         |         |         |

**i7-9700F+16GB / RTX 2070S+8G**

| Model       | 416x416 | 512x512 | 608x608 |
| ----------- | ------- | ------- | ------- |
| YOLOv3      |         |         |         |
| YOLOv3-tiny |         |         |         |
| YOLOv4      | 61 ms   |         |         |
| YOLOv4-tiny | 29 ms   |         |         |

### 3.2 Logs

**Augmentations**

| Name                    | Abbr |
| ----------------------- | ---- |
| Standard Method         | SM   |
| Dynamic mini batch size | DM   |
| Label Smoothing         | LS   |
| Focal Loss              | FL   |
| Mix Up                  | MU   |
| Cut Mix                 | CM   |
| Mosaic                  | M    |

Standard Method Package 包括 Flip left and right,  Crop and Zoom(jitter=0.3), Grayscale, Distort, Rotate(angle=7).

**YOLOv3-tiny**(Pretrained on COCO)

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
|      |      |      |      |      |      |      | L2   | 18.5 | 44.9   | 10.4   |
| ✔    |      |      |      |      |      |      | L2   | 22.0 | 49.1   | 15.2   |
| ✔    | ✔    |      |      |      |      |      | L2   | 22.8 | 49.8   | 16.3   |
| ✔    | ✔    | ✔    |      |      |      |      | L2   | 21.9 | 48.5   | 15.4   |
| ✔    | ✔    |      |      |      |      |      | CIoU | 25.3 | 50.5   | 21.8   |
| ✔    | ✔    |      | ✔    |      |      |      | CIoU | 25.6 | 49.4   | 23.6   |
| ✔    | ✔    |      | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    |      | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    |      | ✔    |      |      | ✔    | CIoU | 23.7 | 46.1   | 21.3   |

可能是，轮数还不充足，LS在小模型上并未展现出优势，甚至M有负面效果。

**YOLOv3**(TODO; Pretrained on COCO)

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
| ✔    | ✔    | ✔    | ✔    |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      | ✔    | CIoU |      |        |        |

**YOLOv4-tiny**(TODO; Pretrained on COCO, part of YOLOv3-tiny weights)

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
| ✔    | ✔    |      | ✔    |      |      |      | CIoU | 27.6 | 48.3   | 28.9   |
| ✔    | ✔    |      | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    |      | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    |      | ✔    |      |      | ✔    | CIoU |      |        |        |

**YOLOv4**(TODO; Pretrained on COCO)

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
| ✔    | ✔    | ✔    | ✔    |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      | ✔    | CIoU |      |        |        |

### 3.3 训练细节

针对于Tiny版本， 实验使用了预训练的YOLOv3-Tiny权重（因为v3与v4主干参数结构上基本是一致的），在冻结了主干参数并保持学习率为1e-4情况下先训练了30轮，接着解冻所有参数并又使用1e-5的学习率训练另外的50轮。 

大模型的训练还在进行中，暂定完全冻结主干部分，先使用warm-up策略训练3轮，再使用余弦退火策略训练另外180轮，初始学习率为5e-4，终止学习率为1e-6。

## 4. Reference

- https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
- https://github.com/hunglc007/tensorflow-yolov4-tflite
- https://github.com/experiencor/keras-yolo3

## 5. History

- Slim version: https://github.com/yuto3o/yolox/tree/slim
- Tensorflow2.0-YOLOv3: https://github.com/yuto3o/yolox/tree/yolov3-tf2