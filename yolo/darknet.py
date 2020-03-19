from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add
from tensorflow.keras.regularizers import l2

WEIGHT_DECAY = 5e-4
LEAKY_ALPHA = 0.1


def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(WEIGHT_DECAY), 'padding': 'valid' if kwargs.get(
        'strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)

    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    without_bias_kwargs = {'use_bias': False}
    without_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = DarknetConv2D(*args, **without_bias_kwargs)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=LEAKY_ALPHA)(x)
        return x

    return wrapper


def DarknetBlock(filters, niter):
    def wrapper(x):
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        x = DarknetConv2D_BN_Leaky(filters, (3, 3), strides=(2, 2))(x)
        for _ in range(niter):
            y = DarknetConv2D_BN_Leaky(filters // 2, (1, 1))(x)
            y = DarknetConv2D_BN_Leaky(filters, (3, 3))(y)
            x = Add()([x, y])
        return x

    return wrapper
