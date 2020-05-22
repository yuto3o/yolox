# -*- coding: utf-8 -*-
import os
import time

from absl import app
from tensorflow.keras import optimizers

from core.utils import decode_cfg, load_weights
from core.dataset import Dataset
from core.callbacks import COCOEvalCheckpoint
from core.model.one_stage.yolov3 import YoloV3_Tiny, YoloLoss

import copy


def main(_argv):
    cfg = decode_cfg("cfgs/voc_yolov3-tiny.yaml")
    model, eval_model = YoloV3_Tiny(cfg)
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

    loss = [YoloLoss(anchors[mask[i]],
                     strides[i],
                     train_dataset.num_classes,
                     ignore_threshold,
                     loss_type) for i in range(len(mask))]

    ckpt_path = os.path.join(cfg["train"]["save_weight_path"], 'tmp', cfg["train"]["label"],
                             time.strftime("%Y%m%d%H%M", time.localtime()))
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(os.path.join(ckpt_path, 'log', 'train', 'plugins', 'profile'))

    _cfg = copy.deepcopy(cfg)
    _cfg['test']['anno_path'] = "./data/pascal_voc/voc2007_val.txt"
    callback = [
        COCOEvalCheckpoint(save_path=os.path.join(ckpt_path, "mAP-{mAP:.4f}.h5"),
                           eval_model=eval_model,
                           model_cfg=cfg,
                           sample_rate=cfg["train"]["sample_rate"],
                           verbose=1),
        COCOEvalCheckpoint(save_path=None,
                           eval_model=eval_model,
                           model_cfg=_cfg,
                           sample_rate=cfg["train"]["sample_rate"],
                           verbose=1)
    ]


    num = 29
    for i in range(num): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))

    model.compile(loss=loss, optimizer=optimizers.Adam(lr=1e-4), run_eagerly=False)
    model.fit(train_dataset,
              steps_per_epoch=len(train_dataset),
              epochs=30,
              callbacks=callback
              )

    for i in range(len(model.layers)): model.layers[i].trainable = True

    model.compile(loss=loss, optimizer=optimizers.Adam(lr=1e-5), run_eagerly=False)
    model.fit(train_dataset,
              steps_per_epoch=len(train_dataset),
              epochs=51,
              callbacks=callback
              )


if __name__ == "__main__":
    app.run(main)
