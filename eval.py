#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.utils import decode_cfg, load_weights
from core.callbacks import COCOEvalCheckpoint, VOCEvalCheckpoint

if __name__ == '__main__':

    cfg = decode_cfg('cfgs/coco_yolov3_tiny.yaml')

    model_type = cfg['yolo']['type']
    if model_type == 'yolov3':
        from core.model.one_stage.yolov3 import YOLOv3 as Model

    elif model_type == 'yolov3_tiny':
        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model

    elif model_type == 'yolov4':
        from core.model.one_stage.yolov4 import YOLOv4 as Model

    elif model_type == 'yolov4_tiny':
        from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model

    else:
        raise NotImplementedError()

    model, eval_model = Model(cfg)
    model.summary()

    init_weight = cfg["train"]["init_weight_path"]
    load_weights(model, init_weight)

    eval_callback = VOCEvalCheckpoint(save_path=None,
                                      eval_model=eval_model,
                                      model_cfg=cfg,
                                      sample_rate=1,
                                      verbose=1)
    eval_callback.on_epoch_end(0)
