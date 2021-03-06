import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import scipy.ndimage as ndi
import math

from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np

brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_brain.nii.gz'
orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_orig.nii.gz'
standard_filename = "D:/DT/BrainMRI/BrainMRI/sub-28797_T1_orig.nii.gz"

IMAGE_HEIGHT = 128
IMAGE_LENGTH = 128
IMAGE_WIDTH = 128


def load_orig(filename):
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
        img_arr = np.pad(img_arr, ((0, 0), (IMAGE_LENGTH - math.ceil(height / 2), IMAGE_LENGTH - math.floor(height / 2)),
                                   (0, 0)), 'constant')
    else:
        y_start = round(height / 2) - IMAGE_LENGTH
        y_end = round(height / 2) + IMAGE_LENGTH
        img_arr = img_arr[:, y_start:y_end, :]

    if queue < IMAGE_HEIGHT * 2:
        img_arr = np.pad(img_arr, ((0, 0), (0, 0),
                                   (IMAGE_HEIGHT - math.ceil(queue / 2), IMAGE_HEIGHT - math.floor(queue / 2))), 'constant')
    else:
        z_start = round(queue / 2) - IMAGE_HEIGHT
        z_end = round(queue / 2) + IMAGE_HEIGHT
        img_arr = img_arr[:, :, z_start:z_end]
    img_arr = ndi.zoom(img_arr, 0.25)
    nib_img = nib.Nifti1Image(img_arr, img.affine)

    # OrthoSlicer3D(img_arr).show()
    return nib_img


# load_orig(standard_filename)
