# -*- coding: utf-8 -*-
import tensorflow as tf

from core.model.one_stage.centernet.common import PreprocessInput, Header

WEIGHT_DECAY = 5e-4


def HourglassConv2D(*args, **kwargs):
    hourglass_conv_kwargs = {"kernel_regularizer": tf.keras.regularizers.l2(WEIGHT_DECAY)}
    hourglass_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **hourglass_conv_kwargs)


def HourglassConv2D_BN_ReLU(*args, **kwargs):
    without_bias_kwargs = {"use_bias": False}
    without_bias_kwargs.update(kwargs)

    kernel_size = without_bias_kwargs.get("kernel_size")

    def wrapper(x):
        x = tf.keras.layers.ZeroPadding2D((kernel_size - 1) // 2)(x)
        x = HourglassConv2D(*args, **without_bias_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    return wrapper


def HourglassBlock(filters, stride=1):
    def wrapper(x):
        shortcut = x
        num_filters = tf.keras.backend.int_shape(shortcut)[-1]

        x = HourglassConv2D_BN_ReLU(filters, kernel_size=3, strides=stride)(x)

        x = HourglassConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if num_filters != filters or stride != 1:
            shortcut = HourglassConv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    return wrapper


def HourglassModule(cnv_dim, dims):
    def wrapper(x):

        features = [x]
        for kk, nh in enumerate(dims):
            x = HourglassBlock(nh, stride=2)(features[-1])
            x = HourglassBlock(nh)(x)
            features.append(x)

        y, dim = features[-1], dims[-1]
        y = HourglassBlock(dim)(y)
        y = HourglassBlock(dim)(y)
        y = HourglassBlock(dim)(y)
        y = HourglassBlock(dim)(y)

        for kk in reversed(range(len(dims))):
            x = features[kk]
            num_channels = dims[kk]
            num_channels_next = dims[max(kk - 1, 0)]
            x = HourglassBlock(num_channels_next)(x)
            x = HourglassBlock(num_channels_next)(x)

            # up: 2 times residual & nearest neighbour
            y = HourglassBlock(num_channels)(y)
            y = HourglassBlock(num_channels_next)(y)
            y = tf.keras.layers.UpSampling2D()(y)

            y = tf.keras.layers.Add()([x, y])

        x = y
        x = HourglassConv2D(cnv_dim, 3)(x)

        # headers
        # cls = HourglassConv2D(256, 3, padding='same')(x)
        # cls = tf.keras.layers.ReLU()(cls)
        # cls = HourglassConv2D(num_classes, 1)(cls)
        #
        # xy = HourglassConv2D(256, 3, padding='same')(x)
        # xy = tf.keras.layers.ReLU()(xy)
        # xy = HourglassConv2D(2, 1)(xy)
        #
        # wh = HourglassConv2D(256, 3, padding='same')(x)
        # wh = tf.keras.layers.ReLU()(wh)
        # wh = HourglassConv2D(2, 1)(wh)

        return x

    return wrapper


def HourglassNetwork(cfg,
                     input_size=None,
                     name=None):

    dims = [256, 384, 384, 384, 512]
    cnv_dim = 256
    num_stacks = 2

    if input_size is None:
        x = inputs = tf.keras.Input([None, None, 3])
    else:
        x = inputs = tf.keras.Input([input_size, input_size, 3])

    x = PreprocessInput()(x)

    x = HourglassConv2D(128, 7, 2, padding='same')(x)
    x = HourglassBlock(cnv_dim, 2)(x)

    outputs = []
    for i in range(num_stacks):
        prev_x = x
        x = HourglassModule(cnv_dim, dims)(x)
        outputs.append(x)
        if i < num_stacks - 1:
            x = HourglassConv2D(cnv_dim, 1, use_bias=False)(prev_x)
            x = tf.keras.layers.BatchNormalization()(x)

            y = HourglassConv2D(cnv_dim, 1, use_bias=False)(x)
            y = tf.keras.layers.BatchNormalization()(y)

            x = tf.keras.layers.Add()([x, y])
            x = tf.keras.layers.ReLU()(x)
            x = HourglassBlock(cnv_dim)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



    m = HourglassNetwork('1', 512)
    m.summary()

