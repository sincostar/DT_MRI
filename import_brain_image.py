from matplotlib import pylab as plt
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')

# from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import math
import scipy.ndimage as ndi


brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28741_T1_orig_brain.nii.gz'
orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_orig.nii.gz'

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128


def load_brain(filename):
    img = nib.load(filename)
    # print(img)
    # print(img.header['db_name'])  # 输出头信息

    width, height, queue = img.dataobj.shape

    # OrthoSlicer3D(img.dataobj).show()

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

    for i in range(IMAGE_WIDTH * 2):
        for j in range(IMAGE_LENGTH * 2):
            for k in range(IMAGE_HEIGHT * 2):
                if img_arr[i, j, k]:
                    img_arr[i, j, k] = 255

    img_arr = ndi.zoom(img_arr, 0.25)
    nib_img = nib.Nifti1Image(img_arr, img.affine)

    # OrthoSlicer3D(img_arr).show()
    return nib_img


def export_mask(mask_arr, out_size):
    out_width, out_length, out_height = out_size
    img_arr = ndi.zoom(mask_arr, 4, order=4).copy()
    # OrthoSlicer3D(img_arr).show()
    width, height, queue = img_arr.shape

    if width < out_width:
        img_arr = np.pad(img_arr, ((math.ceil(out_width / 2) - math.ceil(width / 2),
                                    math.floor(out_width / 2) - math.floor(width / 2)),
                                   (0, 0), (0, 0)), 'constant')
    else:
        x_start = IMAGE_WIDTH - math.ceil(out_width/2)
        x_end = IMAGE_WIDTH + math.floor(out_width/2)
        img_arr = img_arr[x_start:x_end, :, :]

    if height < out_length:
        img_arr = np.pad(img_arr, ((0, 0), (math.ceil(out_length / 2) - math.ceil(height / 2),
                                            math.floor(out_length / 2) - math.floor(height / 2)),
                                   (0, 0)), 'constant')
    else:
        y_start = IMAGE_LENGTH - math.ceil(out_length/2)
        y_end = IMAGE_LENGTH + math.floor(out_length/2)
        img_arr = img_arr[:, y_start:y_end, :]

    if queue < out_height:
        img_arr = np.pad(img_arr, ((0, 0), (0, 0), (math.ceil(out_height / 2) - math.ceil(queue / 2),
                                                    math.floor(out_height / 2) - math.floor(queue / 2))),
                         'constant')
    else:
        z_start = IMAGE_HEIGHT - math.ceil(out_height/2)
        z_end = IMAGE_HEIGHT + math.floor(out_height/2)
        img_arr = img_arr[:, :, z_start:z_end]
    # OrthoSlicer3D(img_arr).show()
    return img_arr


# load_brain(brain_filename)
