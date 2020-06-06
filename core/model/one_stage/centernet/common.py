# -*- coding: utf-8 -*-\
import tensorflow as tf

EPS = 1e-8


class PreprocessInput(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(PreprocessInput, self).__init__(**kwargs)
        self.RGB_mean = [0.47026116, 0.44719303, 0.40789655]
        self.RGB_std = [0.27809834, 0.27408165, 0.2886383]

    def build(self, input_shape):
        super(PreprocessInput, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        x = tf.divide(tf.divide(inputs, 255.) - self.RGB_mean, self.RGB_std)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class Header(tf.keras.layers.Layer):
    def __init__(self, max_outputs=100, peaks_nms_size=3, **kwargs):
        self.max_outputs = max_outputs
        self.peaks_nms_size = peaks_nms_size

        super(Header, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Header, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        cls, delta_xy, size_wh = inputs
        dtype = inputs[0].dtype

        cls_shape = tf.shape(cls)
        delta_xy_shape = tf.shape(delta_xy)
        size_wh_shape = tf.shape(size_wh)
        b, h, w, num_classes = cls_shape[0], cls_shape[1], cls_shape[2], cls_shape[3]

        # peaks NMS
        cls = tf.sigmoid(cls)
        cls_max = tf.nn.max_pool2d(cls, self.peaks_nms_size, 1, padding='SAME')
        keep = tf.cast(tf.equal(cls_max, inputs), dtype)
        cls = cls * keep

        cls_flat = tf.reshape(cls, (b, -1))
        delta_xy_flat = tf.reshape(delta_xy, (delta_xy_shape[0], -1, delta_xy_shape[-1]))
        size_wh_shape_flat = tf.reshape(size_wh_shape, (size_wh_shape[0], -1, size_wh_shape[-1]))

        def _decode(args):
            _cls, _delta_xy, _size_wh = args

            _confi, _inds = tf.math.top_k(_cls, k=self.max_outputs, sorted=True)
            _classes = tf.cast(_inds % num_classes, dtype)
            _inds = tf.cast(_inds / num_classes, tf.int64)
            _xs = tf.cast(_inds % w, dtype)
            _ys = tf.cast(tf.cast(_inds / w, tf.int64), dtype)
            _delta_xy = tf.gather(_delta_xy, _inds)
            _size_wh = tf.gather(_size_wh, _inds)

            _xs = _xs + _delta_xy[..., 0]
            _ys = _ys + _delta_xy[..., 1]

            _x1 = _xs - _size_wh[..., 0] / 2.
            _y1 = _ys - _size_wh[..., 1] / 2.
            _x2 = _xs + _size_wh[..., 0] / 2.
            _y2 = _ys + _size_wh[..., 1] / 2.

            # rescale to image coordinates
            _x1 = _x1 / tf.maximum(w, EPS)
            _y1 = _y1 / tf.maximum(h, EPS)
            _x2 = _x2 / tf.maximum(w, EPS)
            _y2 = _y2 / tf.maximum(h, EPS)

            return tf.stack([_x1, _y1, _x2, _y2], -1), _confi, _classes

        detections = tf.map_fn(_decode, [cls_flat, delta_xy_flat, size_wh_shape_flat], dtype=dtype)
        return detections

    def compute_output_shape(self, input_shape):
        return ([input_shape[0][0], self.max_outputs, 4],
                [input_shape[0][0], self.max_outputs],
                [input_shape[0][0], self.max_outputs])
