import import_orig_image
import import_brain_image
import re
import os
import nibabel as nib
import scipy.misc
import numpy as np
from matplotlib import pylab as plt
import cv2
from numpy import asarray


file_path1 = 'D:/DT/BrainMRI/BrainMRI'
save_path = "data/"

'''
global IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128
'''

def import_data(file_path):
    file_index_list = []
    orig_image_name_list = []
    brain_image_name_list = []
    for file_name in os.listdir(file_path):
        # print(file_name)
        pattern = re.compile(r'\d\d\d\d\d')
        file_index = pattern.findall(file_name)[0]
        if file_index not in file_index_list:
            file_index_list.append(file_index)

        if re.match(".*T1_orig.*", file_name) is not None:
            orig_image_name_list.append(file_name)
        elif re.match(".*T1_brain.*", file_name) is not None:
            brain_image_name_list.append(file_name)

    file_data_index_list = []
    brain_image_data_list = []
    orig_image_data_list = []
    for file_index in file_index_list:
        orig_image_data = []
        brain_image_data = []
        for orig_image_name in orig_image_name_list:
            if re.match(".*" + file_index + ".*", orig_image_name) is not None:
                orig_image_data = import_orig_image.load_orig(file_path1 + '/' + orig_image_name)
                break
        for brain_image_name in brain_image_name_list:
            if re.match(".*" + file_index + ".*", brain_image_name) is not None:
                brain_image_data = import_brain_image.load_brain(file_path1 + '/' + brain_image_name)
                break
        if orig_image_data != [] and brain_image_data != []:
            file_data_index_list.append(file_index)
            brain_image_data_list.append(brain_image_data)
            orig_image_data_list.append(orig_image_data)

    train_orig_image_data = orig_image_data_list[:80]
    train_brain_image_data = brain_image_data_list[:80]
    validation_orig_image_data = orig_image_data_list[81:100]
    validation_brain_image_data = brain_image_data_list[81:100]
    test_orig_image_data = orig_image_data_list[101:]
    test_brain_image_data = brain_image_data_list[101:]

    for i in range(len(train_orig_image_data)):
        nib.save(train_orig_image_data[i], save_path + "train/" + file_data_index_list[i] + "_orig.nii.gz")
        nib.save(train_brain_image_data[i], save_path + "train/" + file_data_index_list[i] + "_brain.nii.gz")

    for i in range(len(validation_orig_image_data)):
        nib.save(validation_orig_image_data[i], save_path + "validation/" +
                 file_data_index_list[i + len(train_orig_image_data)] + "_orig.nii.gz")
        nib.save(validation_brain_image_data[i], save_path + "validation/" +
                 file_data_index_list[i + len(train_orig_image_data)] + "_brain.nii.gz")

    for i in range(len(test_orig_image_data)):
        nib.save(test_orig_image_data[i], save_path + "test/" +
                 file_data_index_list[i+len(train_orig_image_data)+len(validation_orig_image_data)] + "_orig.nii.gz")
        nib.save(test_brain_image_data[i], save_path + "test/" +
                 file_data_index_list[i+len(train_orig_image_data)+len(validation_orig_image_data)] + "_brain.nii.gz")
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
    # return


import_data(file_path1)
