import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')

from nibabel.viewers import OrthoSlicer3D
import numpy as np
import math
import scipy.ndimage as ndi
import os
import re
import import_brain_image
import random
import glob
import cv2


brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28741_T1_orig_brain.nii.gz'
orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_orig.nii.gz'
pve_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28684_T1_brain_pve_0.nii.gz'
file_path1 = 'D:/DT/BrainMRI/BrainMRI'
file_path2 = 'D:/DT/BrainMRI/BrainMRI/Testing_BrainMRI'
save_path1 = "F:/DT_Data/256_size/multi_classification/"

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128


def three_fold_division(file_path):
    file_index_list = glob.glob(file_path + '*_orig.nii.gz')
    random.shuffle(file_index_list)
    fold1_file = file_index_list[0:100].copy()
    np.save(file_path+'fold1_file.npy', np.array(fold1_file))
    fold2_file = file_index_list[100:200].copy()
    np.save(file_path+'fold2_file.npy', np.array(fold2_file))
    fold3_file = file_index_list[200:300].copy()
    np.save(file_path+'fold3_file.npy', np.array(fold3_file))
    validation_file = file_index_list[300:].copy()
    np.save(file_path+'validation_file.npy', np.array(validation_file))


def import_data(orig_image_data, brain_image_data, index_list, save_path, **kwargs):
    data_subpath = kwargs.get('data_subpath', "")
    nib.save(orig_image_data, save_path + data_subpath + index_list + "_orig.nii.gz")
    nib.save(brain_image_data, save_path + data_subpath + index_list + "_multiclass_pve.nii.gz")
    print(data_subpath + str(index_list) + "_orig.nii.gz\tis saved.\n" +
          data_subpath + str(index_list) + "_multiclass_pve.nii.gz\tis saved.")


def get_skull(orig_image_data, brain_image_data):
    out_image_data = np.where(brain_image_data == 0, 1, 0).copy()
    skull_image_data = np.where(orig_image_data * out_image_data > 0.10, 1, 0).copy()
    whole_brain = (skull_image_data + np.where(brain_image_data == 0, 0, 1)).astype('uint8').copy()
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel2 = np.ones((5, 5), np.uint8)
    for i in range(len(whole_brain)):
        slice_image = whole_brain[i].copy()
        slice_image = cv2.morphologyEx(slice_image, cv2.MORPH_CLOSE, kernel1)
        slice_image = cv2.morphologyEx(slice_image, cv2.MORPH_OPEN, kernel2)
        whole_brain[i] = slice_image.copy()
    for i in range(len(whole_brain[1])):
        slice_image = whole_brain[:, i].copy()
        slice_image = cv2.morphologyEx(slice_image, cv2.MORPH_CLOSE, kernel1)
        slice_image = cv2.morphologyEx(slice_image, cv2.MORPH_OPEN, kernel2)
        whole_brain[:, i] = slice_image.copy()
    # OrthoSlicer3D(whole_brain - np.where(brain_image_data == 0, 0, 1)).show()
    return whole_brain - np.where(brain_image_data == 0, 0, 1)


def load_img(filename, **kwargs):
    orig_image_data = kwargs.get('orig_image_data', None)
    zoom_rate = kwargs.get('zoom_rate', 1)

    out_heigth = int(IMAGE_HEIGHT * 2 * zoom_rate)
    out_length = int(IMAGE_LENGTH * 2 * zoom_rate)
    out_width = int(IMAGE_WIDTH * 2 * zoom_rate)

    img_mask = np.zeros((out_heigth, out_length, out_width))
    img = nib.load(filename)

    width, height, queue = img.dataobj.shape

    img_arr = img.dataobj[:, :, :].copy()

    if width < IMAGE_WIDTH * 2:
        img_arr = np.pad(img_arr, ((IMAGE_WIDTH - math.ceil(width / 2), IMAGE_WIDTH - math.floor(width / 2)),
                                   (0, 0), (0, 0)), 'constant')
    else:
        x_start = round(width / 2) - IMAGE_WIDTH
        x_end = round(width / 2) + IMAGE_WIDTH
        img_arr = img_arr[x_start:x_end, :, :]

    if height < IMAGE_LENGTH * 2:
        img_arr = np.pad(img_arr,
                         ((0, 0), (IMAGE_LENGTH - math.ceil(height / 2), IMAGE_LENGTH - math.floor(height / 2)),
                          (0, 0)), 'constant')
    else:
        y_start = round(height / 2) - IMAGE_LENGTH
        y_end = round(height / 2) + IMAGE_LENGTH
        img_arr = img_arr[:, y_start:y_end, :]

    if queue < IMAGE_HEIGHT * 2:
        img_arr = np.pad(img_arr, ((0, 0), (0, 0),
                                   (IMAGE_HEIGHT - math.ceil(queue / 2), IMAGE_HEIGHT - math.floor(queue / 2))),
                         'constant')
    else:
        z_start = round(queue / 2) - IMAGE_HEIGHT
        z_end = round(queue / 2) + IMAGE_HEIGHT
        img_arr = img_arr[:, :, z_start:z_end]
    skull_part = get_skull(orig_image_data, img_arr)
    img_mask = skull_part + img_arr + np.where(img_arr == 0, 0, 1)
    # OrthoSlicer3D(img_mask).show()
    img_mask = ndi.zoom(img_mask, zoom_rate, order=1)

    # img_mask = ndi.zoom(img_mask, zoom_rate, order=1).astype(dtype='int8')
    nib_img = nib.Nifti1Image(img_mask, img.affine)
    # OrthoSlicer3D(img_mask).show()
    return nib_img


def load_init_data(file_path, save_path, **kwargs):
    zoom_rate = kwargs.get('zoom_rate', 1)

    file_index_list = []
    orig_image_name_list = []
    brain_image_name_list = []
    for file_name in os.listdir(file_path):
        pattern = re.compile(r'\d{5}')
        if pattern.findall(file_name):
            file_index = pattern.findall(file_name)[0]
            if file_index not in file_index_list and int(file_index) > 28740:
                file_index_list.append(file_index)

        if re.match(r".*T1_orig.nii.gz", file_name) is not None:
            orig_image_name_list.append(file_name)
        elif re.match(r".*T1_orig_brain_pveseg.nii.gz", file_name) is not None:
            brain_image_name_list.append(file_name)

    for file_index in file_index_list:
        orig_image_name = "sub-" + file_index + "_T1_orig.nii.gz"
        brain_image_name = "sub-" + file_index + "_T1_orig_brain_pveseg.nii.gz"

        orig_image = []
        brain_image = []

        if orig_image_name in orig_image_name_list and brain_image_name in brain_image_name_list:
            orig_image_data = import_brain_image.load_img(file_path + '/' + orig_image_name, zoom_rate=1).dataobj
            orig_image = import_brain_image.load_img(file_path + '/' + orig_image_name, zoom_rate=zoom_rate)
            orig_image_name_list.remove(orig_image_name)
            brain_image = load_img(file_path + '/' + brain_image_name, orig_image_data=orig_image_data, zoom_rate=zoom_rate)
            brain_image_name_list.remove(brain_image_name)
        import_data(orig_image, brain_image, file_index, save_path, data_subpath="whole_data/")


def load_addition_testing_data(file_path, save_path, **kwargs):
    zoom_rate = kwargs.get('zoom_rate', 1)

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
        elif re.match(r".*T1_orig_brain_pveseg.nii.gz", file_name) is not None:
            brain_image_name_list.append(file_name)

    for file_index in file_index_list:
        orig_image_data = []
        orig_image = []
        brain_image = []
        for orig_image_name in orig_image_name_list:
            if re.match(".*" + file_index + ".*", orig_image_name) is not None:
                orig_image_data = import_brain_image.load_img(file_path + '/' + orig_image_name, zoom_rate=1).dataobj.copy()
                orig_image = import_brain_image.load_img(file_path + '/' + orig_image_name, zoom_rate=zoom_rate)
                break
        for brain_image_name in brain_image_name_list:
            if re.match(".*" + file_index + ".*", brain_image_name) is not None:
                brain_image = load_img(file_path + '/' + brain_image_name, orig_image_data=orig_image_data,
                                       zoom_rate=zoom_rate)
                break
        if orig_image != [] and brain_image != []:
            import_data(orig_image, brain_image, file_index, save_path, data_subpath="whole_data/")


# load_img(pve_filename, zoom_rate=0.25)
# load_init_data(file_path1, save_path1, zoom_rate=1)
load_addition_testing_data(file_path2, save_path1, zoom_rate=1)
three_fold_division(save_path1 + "whole_data/")

