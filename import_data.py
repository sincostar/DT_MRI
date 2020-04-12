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
file_path2 = 'D:/DT/BrainMRI/BrainMRI/Testing_BrainMRI'
save_path1 = "data/distortion_data/"

'''
global IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128
'''


def import_data(orig_image_data, brain_image_data, index_list, save_path, **kwargs):
    data_subpath = kwargs.get('data_subpath', "")
    for i in range(len(orig_image_data)):
        # nib.save(orig_image_data[i], save_path + data_subpath + index_list[i] + "_brain.nii.gz")
        # nib.save(brain_image_data[i], save_path + data_subpath + index_list[i] + "_brain_restore.nii.gz")
        print(data_subpath + str(index_list[i]) + "_brain.nii.gz\tis saved.\n" +
              data_subpath + str(index_list[i]) + "_brain_restore.nii.gz\tis saved.")
    # return


def load_init_data(file_path, save_path):
    file_index_list = []
    orig_image_name_list = []
    brain_image_name_list = []
    for file_name in os.listdir(file_path):
        pattern = re.compile(r'\d{5}')
        if pattern.findall(file_name):
            file_index = pattern.findall(file_name)[0]
            if file_index not in file_index_list:
                file_index_list.append(file_index)

        if re.match(r".*T1_brain.nii.gz", file_name) is not None:
            orig_image_name_list.append(file_name)
        elif re.match(r".*T1_brain_restore.nii.gz", file_name) is not None:
            brain_image_name_list.append(file_name)

    file_data_index_list = []
    brain_image_data_list = []
    orig_image_data_list = []
    for file_index in file_index_list:
        orig_image_name = "sub-" + file_index + "_T1_brain.nii.gz"
        brain_image_name = "sub-" + file_index + "_T1_brain_restore.nii.gz"
        if orig_image_name in orig_image_name_list and brain_image_name in brain_image_name_list:
            orig_image_data = import_brain_image.load_img(file_path + '/' + orig_image_name, zoom_rate=0.25)
            orig_image_name_list.remove(orig_image_name)
            brain_image_data = import_brain_image.load_img(file_path + '/' + brain_image_name, zoom_rate=0.25)
            brain_image_name_list.remove(brain_image_name)
            file_data_index_list.append(file_index)
            brain_image_data_list.append(brain_image_data)
            orig_image_data_list.append(orig_image_data)

    train_orig_image_data = orig_image_data_list[:80]
    train_brain_image_data = brain_image_data_list[:80]
    train_index_list = file_data_index_list[:80]
    import_data(train_orig_image_data, train_brain_image_data, train_index_list, save_path, data_subpath="train/")

    validation_orig_image_data = orig_image_data_list[80:100]
    validation_brain_image_data = brain_image_data_list[80:100]
    validation_index_list = file_data_index_list[80:100]
    import_data(validation_orig_image_data, validation_brain_image_data, validation_index_list,
                save_path, data_subpath="validation/")

    test_orig_image_data = orig_image_data_list[100:]
    test_brain_image_data = brain_image_data_list[100:]
    test_index_list = file_data_index_list[100:]
    import_data(test_orig_image_data, test_brain_image_data, test_index_list, save_path, data_subpath="test/")


def load_addition_testing_data(file_path, save_path):
    file_index_list = []
    orig_image_name_list = []
    brain_image_name_list = []
    for file_name in os.listdir(file_path):
        pattern = re.compile(r'\d{5}')
        file_index = pattern.findall(file_name)[0]
        if file_index not in file_index_list:
            file_index_list.append(file_index)

        if re.match(r".*T1_orig.nii.gz", file_name) is not None:
            orig_image_name_list.append(file_name)
        elif re.match(r".*T1_orig_brain.nii.gz", file_name) is not None:
            brain_image_name_list.append(file_name)

    file_data_index_list = []
    brain_image_data_list = []
    orig_image_data_list = []
    for file_index in file_index_list:
        orig_image_data = []
        brain_image_data = []
        for orig_image_name in orig_image_name_list:
            if re.match(".*" + file_index + ".*", orig_image_name) is not None:
                orig_image_data = import_orig_image.load_orig(file_path + '/' + orig_image_name)
                break
        for brain_image_name in brain_image_name_list:
            if re.match(".*" + file_index + ".*", brain_image_name) is not None:
                brain_image_data = import_brain_image.load_brain(file_path + '/' + brain_image_name)
                break
        if orig_image_data != [] and brain_image_data != []:
            file_data_index_list.append(file_index)
            brain_image_data_list.append(brain_image_data)
            orig_image_data_list.append(orig_image_data)

    import_data(orig_image_data_list, brain_image_data_list, file_data_index_list, save_path, data_subpath="new_test/")


load_init_data(file_path1, save_path1)
# load_addition_testing_data(file_path2, save_path1)
