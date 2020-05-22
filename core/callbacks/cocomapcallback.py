import tensorflow as tf

from core.metrics import COCOeval
from core.callbacks.utils import local_eval


class COCOEvalCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self,
                 save_path,
                 eval_model,
                 model_cfg,
                 sample_rate,
                 verbose=0):
        super(COCOEvalCheckpoint, self).__init__()
        self.save_path = save_path
        self.eval_model = eval_model
        self.model_cfg = model_cfg

        self.verbose = verbose
        self.sample_rate = sample_rate

        self._image_size = self.model_cfg['test']['image_size'][0]
        self._best_mAP = -float('inf')

        self.name_path = self.model_cfg['train']['name_path']
        self.test_path = self.model_cfg['test']['anno_path']

        self.train_path = self.model_cfg['train']['anno_path']

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.sample_rate != self.sample_rate - 1:
            return

        print('\nTest')
        mAP = local_eval(COCOeval, self.eval_model, self._image_size, self.test_path, self.name_path, self.verbose)

        if mAP > self._best_mAP:
            if self.save_path is None:
                if self.verbose > 0:
                    print("mAP improved from {:.2%} to {:.2%}".format(self._best_mAP, mAP))
            else:
                save_path = self.save_path.format(mAP=mAP)
                if self.verbose > 0:
                    print(
                        "mAP improved from {:.2%} to {:.2%}, saving model to {}".format(self._best_mAP, mAP, save_path))

                self._best_mAP = mAP
                self.eval_model.save_weights(save_path)
        else:
            if self.verbose > 0:
                print("mAP not improved from {:.2%}".format(self._best_mAP))
