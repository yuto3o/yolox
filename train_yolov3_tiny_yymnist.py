# -*- coding: utf-8 -*-
import os
import time

from absl import flags, app
from core import config
from core.yolov3_tiny import YoloV3_Tiny
from core.yolov3_commom import Yolo_Loss
from core.data import SimpleGenerator
from tensorflow.keras import optimizers, callbacks


def warmup(epoch, lr, after_epoch, from_lr, to_lr):
    if epoch > after_epoch:
        return lr
    else:
        return from_lr + (to_lr - from_lr) * epoch / after_epoch


def main(_argv):
    cfg = config.load("cfg/yymnist_yolov3-tiny.yaml")

    model = YoloV3_Tiny(cfg, is_training=True)
    if cfg["train"]["init_weight_path"]:
        model.load_weights(cfg["train"]["init_weight_path"])

    loss = [Yolo_Loss(cfg['yolo']['num_classes'],
                      cfg['yolo']['anchors'][cfg['yolo']['mask'][i]], cfg['yolo']['strides'][i],
                      cfg['train']['ignore_threshold']) for i in range(2)]
    optimizer = optimizers.Adam(lr=0.0)
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=False)

    ckpt_path = os.path.join(cfg["train"]["save_weight_path"], 'tmp', 'yymnist_yolov3_tiny', time.strftime("%Y%m%d%H%M", time.localtime()))
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(os.path.join(ckpt_path, 'log', 'train', 'plugins', 'profile'))

    callback = [callbacks.LearningRateScheduler(lambda epoch, lr: warmup(epoch, lr,
                                                                         cfg["train"]["warmup_epoch"],
                                                                         cfg["train"]["lr_end"],
                                                                         cfg["train"]["lr_init"]),
                                                verbose=1),
                callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, min_lr=cfg["train"]["lr_end"], factor=0.5),
                callbacks.ModelCheckpoint(os.path.join(ckpt_path, "loss-{val_loss:.4f}.h5"),
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True),
                callbacks.TensorBoard(os.path.join(ckpt_path, 'log'))
                ]
    train_dataset = SimpleGenerator("train", cfg)
    test_dataset = SimpleGenerator("test", cfg)

    model.fit(train_dataset(),
              steps_per_epoch=train_dataset.num_batch,
              epochs=cfg["train"]["num_epoch"],
              callbacks=callback,
              validation_data=test_dataset(),
              validation_steps=test_dataset.num_batch
              )


if __name__ == "__main__":
    app.run(main)
