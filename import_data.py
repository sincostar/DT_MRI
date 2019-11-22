import import_orig_image
import import_brain_image
import re
import os
from matplotlib import pylab as plt
import cv2
import numpy as np


file_path1 = 'D:/DT/BrainMRI/BrainMRI'

'''
global IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128
'''


def import_data(file_path):
    file_index_list = []

    for file_name in os.listdir(file_path):
        # print(file_name)
        pattern = re.compile(r'\d\d\d\d\d')
        file_index = pattern.findall(file_name)[0]
        if file_index not in file_index_list:
            file_index_list.append(file_index)
    print(file_index_list)
    '''
    for file_name in os.listdir(file_path):
        if re.match(".*T1_orig.*", file_name) is not None and file_index == int(pattern.findall(file_name)[0]):
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)

        elif re.match(".*T1_brain.*", file_name) is not None:
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)
            file_index = int(pattern.findall(file_name)[0])
    '''
    """
    brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28677_T1_brain.nii.gz'
    orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28677_T1_orig.nii.gz'
    orig = import_orig_image.load_orig(orig_filename)[:, :, 50].astype(np.uint8)
    brain = import_brain_image.load_brain(brain_filename)[:, :, 50].astype(np.uint8)
    print(brain.size)
    # plt.imshow(orig, cmap='Reds')
    # plt.imshow(brain)
    # plt.show()
    b = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    g = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    r = np.zeros((128, 128), dtype=np.uint8)
    img = cv2.merge([orig, brain, r])

    enlarge_img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("merged 1", enlarge_img)
    cv2.waitKey(0)
    cv2.destroyWindow('test')
    """

import_data(file_path1)