# -*- coding: utf-8 -*-
import os
import time

from absl import app, flags
from tensorflow.keras import optimizers

from core.utils import decode_cfg, load_weights
from core.dataset import Dataset
from core.callbacks import COCOEvalCheckpoint, CosineAnnealingScheduler, WarmUpScheduler

flags.DEFINE_string('config', '', 'path to config file')
FLAGS = flags.FLAGS


def main(_argv):
    print('Config File From:', FLAGS.config)
    cfg = decode_cfg(FLAGS.config)

    model_type = cfg['yolo']['type']
    if model_type == 'yolov3':
        from core.model.one_stage.yolov3 import YOLOv3 as Model
        from core.model.one_stage.yolov3 import YOLOLoss as Loss
        num = 186
        epochs = 180
    elif model_type == 'yolov3_tiny':
        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model
        from core.model.one_stage.yolov3 import YOLOLoss as Loss
        num = 29
        epochs = 90
    elif model_type == 'yolov4':
        from core.model.one_stage.yolov4 import YOLOv4 as Model
        from core.model.one_stage.yolov4 import YOLOLoss as Loss
        num = 251
        epochs = 180
    elif model_type == 'yolov4_tiny':
        from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model
        from core.model.one_stage.yolov4 import YOLOLoss as Loss
        num = 29
        epochs = 90
    else:
        raise NotImplementedError()

    model, eval_model = Model(cfg)
    model.summary()
    train_dataset = Dataset(cfg)

    init_weight = cfg["train"]["init_weight_path"]
    anchors = cfg['yolo']['anchors']
    mask = cfg['yolo']['mask']
    strides = cfg['yolo']['strides']
    ignore_threshold = cfg['train']['ignore_threshold']
    loss_type = cfg['train']['loss_type']

    if init_weight:
        load_weights(model, init_weight)
    else:
        print("Training from scratch")
        num = 0

    loss = [Loss(anchors[mask[i]],
                 strides[i],
                 train_dataset.num_classes,
                 ignore_threshold,
                 loss_type) for i in range(len(mask))]

    ckpt_path = os.path.join(cfg["train"]["save_weight_path"], 'tmp', cfg["train"]["label"],
                             time.strftime("%Y%m%d%H%M", time.localtime()))
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    #     os.makedirs(os.path.join(ckpt_path, 'log', 'train', 'plugins', 'profile'))

    warmup_epochs = 10
    warmup_callback = [WarmUpScheduler(learning_rate=1e-3, warmup_step=warmup_epochs * len(train_dataset), verbose=1)]

    eval_callback = [COCOEvalCheckpoint(save_path=os.path.join(ckpt_path, "mAP-{mAP:.4f}.h5"),
                                        eval_model=eval_model,
                                        model_cfg=cfg,
                                        sample_rate=10,
                                        verbose=1)
                     ]
    lr_callback = [CosineAnnealingScheduler(learning_rate=1e-3,
                                            eta_min=1e-6,
                                            T_max=epochs * len(train_dataset),
                                            verbose=1)]

    for i in range(num): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))

    # warm-up
    model.compile(loss=loss, optimizer=optimizers.Adam(lr=0.), run_eagerly=False)
    model.fit(train_dataset,
              steps_per_epoch=len(train_dataset),
              epochs=warmup_epochs,
              callbacks=warmup_callback
              )
    model.save_weights('./weights.h5')

    for i in range(len(model.layers)): model.layers[i].trainable = True
    print('Unfreeze all layers.')

    model.compile(loss=loss, optimizer=optimizers.Adam(lr=0.), run_eagerly=False)
    model.fit(train_dataset,
              steps_per_epoch=len(train_dataset),
              epochs=epochs,
              callbacks=eval_callback + lr_callback
              )

    model.compile(loss=loss, optimizer=optimizers.Adam(lr=1e-6), run_eagerly=False)
    model.fit(train_dataset,
              steps_per_epoch=len(train_dataset),
              epochs=10,
              callbacks=eval_callback
              )


if __name__ == "__main__":
    app.run(main)
