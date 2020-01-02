import os
import numpy as np
import math

from PIL import Image
from utils import util as U


def save_str(data_dict, filename, idx):
    with open(filename, 'a+') as f:
        f.write('{}: {}\n'.format(idx, U.dict_to_str(data_dict)))


def save_img(data_dict, filepath, idx):
    print(filepath)
    print(type(data_dict))
    if data_dict is None or not data_dict:
        print("Y")
        return
    if not os.path.exists(filepath):
        print("X")
        os.makedirs(filepath)
    for k in data_dict:
        imgs = data_dict[k]
        imgs = np.concatenate(imgs, 0)
        print('image shape: ' + str(imgs.shape))
        # for the 3D images we choose 5 slices from same direction to show the result
        if len(imgs.shape) == 4 or len(imgs.shape) == 3 and not(imgs.shape[-1] in [1, 3]):
            height = imgs.shape[2]
            height1 = math.floor(height / 6)
            Image.fromarray(imgs[..., height1]).save('{}/idx_{}_{}_{}.png'.format(filepath, idx, k, 'h1'))
            height2 = math.floor(height / 3)
            Image.fromarray(imgs[..., height2]).save('{}/idx_{}_{}_{}.png'.format(filepath, idx, k, 'h2'))
            height3 = math.floor(height / 2)
            Image.fromarray(imgs[..., height3]).save('{}/idx_{}_{}_{}.png'.format(filepath, idx, k, 'h3'))
            height4 = math.ceil(2 * height / 3)
            Image.fromarray(imgs[..., height4]).save('{}/idx_{}_{}_{}.png'.format(filepath, idx, k, 'h4'))
            height5 = math.ceil(5 * height / 6)
            Image.fromarray(imgs[..., height5]).save('{}/idx_{}_{}_{}.png'.format(filepath, idx, k, 'h5'))

        elif len(imgs.shape) == 3:
            Image.fromarray(imgs).save('{}/idx_{}_{}.png'.format(filepath, idx, k))
