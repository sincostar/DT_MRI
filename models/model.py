import numpy as np
import tensorflow as tf
from abc import ABCMeta,abstractmethod

from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util as U
from utils.process_methods import one_hot

class Model(metaclass=ABCMeta):
    
    def __init__(self, net):
        self.net = net

    @abstractmethod
    def get_grads(self, data_dict):
        """
        """

    @abstractmethod
    def eval(self, data_dict, **kwargs):
        """
        """

    @abstractmethod
    def predict(self, data_dict):
        """
        """



class SimpleTFModel(Model):
    def __init__(self, net, x_suffix, y_suffix, m_suffix=None, dropout=0, loss_function={'balanced_dice': 1.}, weight_function=None):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._m_suffix = m_suffix
        self._loss_function = loss_function
        self._weight_function = weight_function

        self._loss_dict = {'cross-entropy': LF.cross_entropy,
                           'dice': LF.dice_coefficient,
                           'balanced_dice': LF.balanced_dice_coefficient}
        self._weight_dict = {'balance': LF.balance_weight_map,
                             'feedback': LF.feedback_weight_map}

        self.dropout = dropout

    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]

        with tf.GradientTape() as tape:
            logits = self.net(xs, self.dropout, True)
            loss = self._get_loss(logits, data_dict)                       
        grads = tape.gradient(loss, self.net.trainable_variables)
        return grads

    def eval(self, data_dict, **kwargs):
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        
        logits = self.net(xs, 0, False)
        prob = tf.nn.softmax(logits, -1)
        pred = one_hot(np.argmax(prob, -1), ys.shape[-1])

        loss = [self._get_loss(logits, data_dict)]
        acc = EM.accuracy(pred, ys)
        dice = EM.dice_coefficient(pred, ys)
        iou = EM.iou(pred, ys)
        # auc = EM.auc(pred, ys)

        precision = EM.precision(pred, ys)
        recall = EM.recall(pred, ys)
        sensitivity = EM.sensitivity(pred, ys)
        specificity = EM.specificity(pred, ys)

        eval_str = {'loss': loss,
                   'acc': acc,
                   'dice': dice,
                   'iou': iou,
                #    'auc': auc,
                   'precision': precision,
                   'recall': recall,
                   'sensitivity': sensitivity,
                   'specificity': specificity}

        need_imgs = kwargs.get('need_imgs', False)
        eval_img = None
        if need_imgs:
            eval_img = self._get_imgs_eval(xs, ys, prob)

        need_logits = kwargs.get('need_logits', False)
        if need_logits:
            return eval_str, eval_img, logits
        return eval_str, eval_img

    def predict(self, data_dict):
        return self.net(data_dict[self._x_suffix])

    def _get_loss(self, logits, data_dict):
        loss_data_dict = {'orgs': data_dict[self._x_suffix],
                          'labels': data_dict[self._y_suffix],
                          'masks': None,
                          'logits': logits}
        total_loss_map = None
        assert self._loss_function is not None and type(self._loss_function) is dict, 'loss_function should be a dict ({name: weight})'
        for lf in self._loss_function:
            weight = self._loss_function[lf]
            loss_function = self._loss_dict[lf]
            sub_loss_map = loss_function(loss_data_dict)
            total_loss_map = sub_loss_map * weight if total_loss_map is None else total_loss_map + sub_loss_map * weight
   
        if self._weight_function is not None:
            for wf in self._weight_function:
                weight_function = self._weight_dict[wf]
                if type(self._weight_function) is dict and self._weight_function[wf] is not None:
                    w_args = self._weight_function[wf]
                else:
                    w_args = {}
                sub_weight_map = weight_function(loss_data_dict, **w_args)
                total_loss_map *= sub_weight_map

        total_loss = tf.reduce_mean(total_loss_map)
        return total_loss

    def _get_imgs_eval(self, xs, ys, prob):
        img_dict = {}
        n_class = ys.shape[-1]
        for i in range(n_class):
            img = U.combine_2d_imgs_from_tensor([xs, ys[..., i], prob[..., i]])
            img_dict.update({'class %d'%i: img})

        argmax_ys = np.argmax(ys, -1)
        argmax_prob = np.argmax(prob, -1)
        img = U.combine_2d_imgs_from_tensor([xs, argmax_ys, argmax_prob])
        img_dict.update({'argmax': img})

        return img_dict

