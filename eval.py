#!/usr/bin/env python
# -*- coding: utf-8 -*-
from absl import app, flags

from core.utils import decode_cfg, load_weights
from core.callbacks import VOCEvalCheckpoint

flags.DEFINE_string('config', '', 'path to config file')
FLAGS = flags.FLAGS


def main(_argv):
    print('Config File From:', FLAGS.config)
    cfg = decode_cfg(FLAGS.config)

    model_type = cfg['yolo']['type']
    if model_type == 'yolov3':
        from core.model.one_stage.yolov3 import YOLOv3 as Model

    elif model_type == 'yolov3_tiny':
        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model

    elif model_type == 'yolov4':
        from core.model.one_stage.yolov4 import YOLOv4 as Model

    elif model_type == 'yolov4_tiny':
        from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model

    elif model_type == 'yolox':
        from core.model.one_stage.custom import YOLOX as Model

    else:
        raise NotImplementedError()

    model, eval_model = Model(cfg)
    model.summary()

    init_weight = cfg["test"]["init_weight_path"]
    load_weights(model, init_weight)

    eval_callback = VOCEvalCheckpoint(save_path=None,
                                      eval_model=eval_model,
                                      model_cfg=cfg,
                                      verbose=1)
    eval_callback.on_epoch_end(0)


if __name__ == '__main__':
    app.run(main)
