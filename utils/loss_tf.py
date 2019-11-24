import numpy as np
import tensorflow as tf

# loss functions -------------------------------------------
def cross_entropy(data_dict):
    logits = data_dict['logits']
    labels = data_dict['labels']
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return loss_map

def mse(logits, labels):
    loss_map = tf.reduce_mean(tf.square(logits-labels))
    return loss_map


def dice_coefficient(data_dict, epsilon=1e-9):
    logits = tf.cast(data_dict['logits'], tf.float32)
    labels = tf.cast(data_dict['labels'], tf.float32)
    axis = tuple(range(len(labels.shape) - 1)) if len(labels.shape) > 1 else -1
    pred = tf.nn.softmax(logits)
    # pred = tf.one_hot(tf.argmax(logits, -1), labels.shape[-1])
    
    intersection = tf.reduce_sum(pred * labels, axis)
    sum_ = tf.reduce_sum(pred + labels, axis)
    dice = 1 - 2 * intersection / (sum_ + epsilon)
    return dice

def balanced_dice_coefficient(data_dict, epsilon=1e-9):
    labels = tf.cast(data_dict['labels'], tf.float32)
    dice_loss = dice_coefficient(data_dict, epsilon)
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 1 else -1
    c = 1/(np.sum(1/(epsilon + np.sum(labels, axis=axis))))
    balanced_weight = c/(epsilon + np.sum(labels, axis=axis))
    dice = dice_loss * balanced_weight
    return dice
# ----------------------------------------------------------

# weight maps ----------------------------------------------

def balance_weight_map(data_dict, epsilon=1e-9):
    labels = data_dict['labels']
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 1 else -1
    c = 1/(np.sum(1/(epsilon + np.sum(labels, axis=axis))))
    weight_map = np.sum(labels * np.tile(c/(epsilon + np.sum(labels, axis=axis, keepdims=True)), list(labels.shape[0:-1]) + [1]), axis=-1)
    return weight_map

def feedback_weight_map(data_dict, alpha=3, beta=100):
    logits = data_dict['logits']
    labels = data_dict['labels']
    probs = tf.nn.softmax(logits, -1)
    p = np.sum(probs * labels, axis=-1)
    weight_map = np.exp(-np.power(p, beta)*np.log(alpha))
    return weight_map 

# ----------------------------------------------------------