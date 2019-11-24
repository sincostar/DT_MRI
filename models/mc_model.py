import numpy as np
import tensorflow as tf
from models.model import SimpleTFModel

class MonteCarloModel(SimpleTFModel):
    """Class for process images.

        Parameters
        ----------
        unc_dict: None or a dict of parameters, optional
            The uncertainty dictionary should contain dropout, alpha and t_stochastic values
            parameter dropout: the value of drop out to calculate the monte carlo integration
            parameter alpha: the minimum value of the weight if the uncertainty equals to 0
            parameter t_stochastic: number of forward passes to calculate the monte carlo integration
            EX. unc_dict={'dropout':0.20, 't_stochastic':20, 'alpha':0.1}

        """
    def __init__(self, net, x_suffix, y_suffix, m_suffix=None, dropout=0, loss_function={'cross-entropy': 1.}, weight_function=None, unc_dict=None):
        super().__init__(net, x_suffix, y_suffix, m_suffix, dropout, loss_function, weight_function)
        self._unc_dict = unc_dict

    def get_grads(self, data_dict):
        xs = data_dict[self._x_suffix]
        #get uncertainty and calculate the uncertainty weight map
        if self._unc_dict is not None:
            uncertainty = self._uncertainty_mc_integration(xs, self._unc_dict['dropout'], self._unc_dict['t_stochastic'])
            uncertainty_weight = self._uncertainty_weight_map(uncertainty, self._unc_dict['alpha'])
        else:
            uncertainty_weight = 1

        with tf.GradientTape() as tape:
            logits = self.net(xs, self.dropout, True)
            loss = self._get_loss(logits, data_dict, uncertainty_weight)
        grads = tape.gradient(loss, self.net.trainable_variables)
        return grads

    def eval(self, data_dict, **kwargs):
        eval_str, eval_img, logits = super().eval(data_dict, **kwargs, need_logits=True)
        cal_unc = kwargs.get('cal_unc', False)
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        prob = tf.nn.softmax(logits, -1)
        if cal_unc:
            uc_map = self._uncertainty_mc_integration(xs, 0.2, 20)
            eval_str.update({'uc_map': uc_map})
            eval_str.update({'prob_map': prob})
            eval_str.update({'org_map': xs})
            eval_str.update({'gt_map': ys})

        return eval_str, eval_img

    def _get_loss(self, logits, data_dict, uncertainty_weight=1):
        total_loss_map = super()._get_loss(logits, data_dict)
        #to apply the uncertainty weight map to total loss
        total_loss_map = uncertainty_weight * total_loss_map

        return tf.reduce_mean(total_loss_map)

    def _uncertainty_mc_integration(self, data, dropout, t_stochastic):
        # T stochastic Forward passes to calculate uncertainty after each iteration
        segmentation_score = None
        for _ in range(t_stochastic):
            logit = self.net(data, dropout, False)
            prob = tf.nn.softmax(logit, -1)
            segmentation_score = prob if segmentation_score is None else segmentation_score + prob

        # The average score contains the probability score of each class
        segmentation_score = segmentation_score / t_stochastic
        # The uncertainty will be calculated using the entropy of the segmentation score
        uncertainty = -np.sum(segmentation_score * np.log2(segmentation_score), axis=-1)
        return uncertainty

    def _uncertainty_weight_map(self, uncertainty, alpha):
        #a simple linear weight map
        uncertainty_weight= np.power(alpha, (1-uncertainty))
        return uncertainty_weight