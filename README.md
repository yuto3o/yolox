# KISS YOLOv3



May be finished in the future.

- [x] **ConvertScript**: Convert original [yolov3.weight](https://pjreddie.com/media/files/yolov3.weights) to Tensorflow style
- [ ] **Backbone**:  A simple task, but I have no time.
- [ ] ...

---

Download yolov3.weight and yolov3.cfg from the [Homepage](https://pjreddie.com/darknet/yolo/).

Run our script(you need Tensorflow, Numpy and Python3 only).

```python
from yoloparser import YoloParser

weights_path = "./yolov3/yolov3.weights"
cfg_path = "./yolov3/yolov3.cfg"
out_path = "./yolov3/yolov3.ckpt"

parser = YoloParser(cfg_path, weights_path, out_path)
parser.run()
```

```shell
Reading .cfg file ...
Converting ...
From C:\Users\yuyang\Downloads\yolov3\yolov3.weights
To   C:\Users\yuyang\Downloads\yolov3\yolov3.ckpt
Encode weights...
Success!
Model Parameters:
<tf.Variable 'Conv_0/weights:0' shape=(3, 3, 3, 32) dtype=float32_ref>
<tf.Variable 'Conv_0/BatchNorm/gamma:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Conv_0/BatchNorm/beta:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Conv_0/BatchNorm/moving_mean:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Conv_0/BatchNorm/moving_variance:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'Conv_1/weights:0' shape=(3, 3, 32, 64) dtype=float32_ref>
<tf.Variable 'Conv_1/BatchNorm/gamma:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'Conv_1/BatchNorm/beta:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'Conv_1/BatchNorm/moving_mean:0' shape=(64,) dtype=float32_ref>
<tf.Variable 'Conv_1/BatchNorm/moving_variance:0' shape=(64,) dtype=float32_ref>
...
Finish !
```

You can check all the variable by Tensorflow API. Of course,  renaming all variable is possible by tf.train.Saver.

```python
import tensorflow as tf

reader = tf.train.NewCheckpointReader(out_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
  print( key)
```

```shell
Conv_15/weights
Conv_0/BatchNorm/beta
Conv_0/BatchNorm/gamma
Conv_27/BatchNorm/gamma
Conv_11/weights
Conv_57/BatchNorm/moving_mean
Conv_1/BatchNorm/gamma
Conv_56/BatchNorm/gamma
Conv_0/BatchNorm/moving_mean
Conv_7/BatchNorm/gamma
Conv_44/BatchNorm/moving_variance
Conv_0/weights
Conv_15/BatchNorm/beta
Conv_0/BatchNorm/moving_variance
...
```

