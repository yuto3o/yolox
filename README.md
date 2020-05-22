# More Than YOLO

TensorFlow & Keras Implementations & Python

YOLOv3, YOLOv3-tiny, YOLOv4

YOLOv4-tiny(unofficial)

**requirements:** TensorFlow 2.x (not test on 1.x), OpenCV, Numpy, PyYAML

---

This repository have done:

- [x] Backbone for YOLO (YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv4-tiny[unofficial])
- [x] YOLOv3 Head
- [x] Keras Callbacks for Online Evaluation
- [x] Load Official Weight File
- [x] K-Means for Anchors
- [x] Training Script (Strategy and Model Config)
  - Define simple training in [train.py](./train.py)
  - Use YAML as config file in [cfgs](./cfgs)
- [ ] Data Augmentation
  - [x] Standard Method: Random Flip, Random Crop, Zoom, Random Grayscale, Random Distort
  - [x] Hight Level: Cut Mix, Mix Up, Mosaic
  - [ ] More, I can be so much more ... 
- [ ] For Loss
  - [x] Label Smoothing
  - [x] Focal Loss
  - [x] L2, D-IoU, G-IoU, C-IoU
  - [ ] ...

---

[toc]

## 0. Read Source Code for More Details

You can get official weight files from https://github.com/AlexeyAB/darknet/releases or https://pjreddie.com/darknet/yolo/.

## 1. Samples

### 1.1 Data File

We use a special data format like that,

```txt
path/to/image1 x1,y1,x2,y2,label x1,y1,x2,y2,label 
path/to/image2 x1,y1,x2,y2,label 
...
```

Convert your data format firstly. We present a script for Pascal VOC in https://github.com/yuto3o/yolox/blob/master/data/pascal_voc/voc_convert.py

More details and a simple dataset could be got from https://github.com/YunYang1994/yymnist.

### 1.2 Configure 

```yaml
# voc_yolov3_tiny.yaml
yolo:
  type: "yolov3_tiny" # must be 'yolov3', 'yolov3_tiny', 'yolov4', 'yolov4_tiny'.
  iou_threshold: 0.45
  score_threshold: 0.5
  max_boxes: 100
  strides: "32,16"
  anchors: "10,14 23,27 37,58 81,82 135,169 344,319"
  mask: "3,4,5 0,1,2"

train:
  label: "voc_yolov3_tiny" # any thing you like
  name_path: "./data/pascal_voc/voc.name"
  anno_path: "./data/pascal_voc/train.txt"
  # "416" for single mini batch size, "352,384,416,448,480" for Dynamic mini batch size.
  image_size: "416" 

  batch_size: 4
  # if you want to load .weights file, you should use something like coco.yaml.
  init_weight_path: "./ckpts/yolov3-tiny.h5"
  sample_rate: 5 # eval for every 5 epochs
  save_weight_path: "./ckpts"

  # Must be "L2", "CIou", "GIou", "CIou" or something like "L2+FL" for focal loss
  loss_type: "L2" 
  
  # turn on hight level data augmentation
  mix_up: false
  cut_mix: false
  mosaic: false
  label_smoothing: false

  ignore_threshold: 0.5

test:
  anno_path: "./data/pascal_voc/test.txt"
  image_size: "416" # image size for test
  batch_size: 1
  init_weight_path: ""
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

A simple demo for YOLOv4

![yolov4](./misc/dog_v4.jpg)

```python
# read config
cfg = decode_cfg("cfgs/coco_yolov4.yaml")

model, eval_model = YoloV4(cfg)
eval_model.summary()

# assign colors for difference labels
names = cfg["train"]["names"]
shader = Shader(len(names))

# load weights
load_weights(model, cfg["test"]["init_weight_path"])

img_raw = read_image(r'./misc/dog.jpg')
img = preprocess_image(img_raw, (512, 512))
imgs = img[np.newaxis, ...]

tic = time.time()
boxes, scores, classes, valid_detections = eval_model.predict(imgs)
print(time.time() - tic, 's')

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

## 2. Experiment

We freeze backbone for first 30 epochs, and then finetune  all of the trainable variables for another 50 epochs. 

| Name                    | Abbr |
| ----------------------- | ---- |
| Standard Method         | SM   |
| Dynamic mini batch size | DM   |
| Label Smoothing         | LS   |
| Focal Loss              | FL   |
| Mix Up                  | MU   |
| Cut Mix                 | CM   |
| Mosaic                  | M    |

**YOLOv3-tiny**

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
|      |      |      |      |      |      |      | L2   | 18.5 | 44.9   | 10.4   |
| ✔    |      |      |      |      |      |      | L2   | 22.0 | 49.1   | 15.2   |
| ✔    | ✔    |      |      |      |      |      | L2   |      |        |        |
| ✔    | ✔    | ✔    |      |      |      |      | L2   |      |        |        |
| ✔    | ✔    | ✔    |      |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      | ✔    | CIoU |      |        |        |



**YOLOv4-tiny**

| SM   | DM   | LS   | FL   | MU   | CM   | M    | Loss | mAP  | mAP@50 | mAP@75 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ |
|      |      |      |      |      |      |      | L2   |      |        |        |
| ✔    |      |      |      |      |      |      | L2   |      |        |        |
| ✔    | ✔    |      |      |      |      |      | L2   |      |        |        |
| ✔    | ✔    | ✔    |      |      |      |      | L2   |      |        |        |
| ✔    | ✔    | ✔    |      |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    | ✔    |      |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      | ✔    |      | CIoU |      |        |        |
| ✔    | ✔    | ✔    | ✔    |      |      | ✔    | CIoU |      |        |        |

## 3. Reference

- https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
- https://github.com/hunglc007/tensorflow-yolov4-tflite
- https://github.com/experiencor/keras-yolo3

## 4. History

- Slim version: https://github.com/yuto3o/yolox/tree/slim
- Tensorflow2.0-YOLOv3: https://github.com/yuto3o/yolox/tree/yolov3-tf2