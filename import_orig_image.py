# Plot the mean image because we have no anatomic data

from matplotlib import pylab as plt
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')

from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np

brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28677_T1_brain.nii.gz'
orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_orig.nii.gz'

IMAGE_HEIGHT = 100
IMAGE_LENGTH = 125
IMAGE_WIDTH = 100


def load_orig(filename):
    img = nib.load(filename)
    print(img)
    print(img.header['db_name'])  # 输出头信息

    width, height, queue = img.dataobj.shape

    print("width: " + str(width))
    print("height: " + str(height))
    print("queue: " + str(queue))

    # OrthoSlicer3D(img.dataobj).show()

    x_start = round(width / 2) - IMAGE_WIDTH
    x_end = round(width / 2) + IMAGE_WIDTH
    y_start = round(height / 2) - IMAGE_LENGTH
    y_end = round(height / 2) + IMAGE_LENGTH
    z_start = round(queue / 2) - IMAGE_HEIGHT
    z_end = round(queue / 2) + IMAGE_HEIGHT

    img_arr = img.dataobj[x_start:x_end, y_start:y_end, round(queue / 2)].copy()
    img_arr1 = img.dataobj[x_start:x_end, y_start:y_end, z_start:z_end].copy()
    # OrthoSlicer3D(img_arr1).show()
    print(type(img_arr1))
    print(img_arr.shape)
    plt.imshow(img_arr, cmap='gray')
    plt.show()


load_orig(orig_filename)
