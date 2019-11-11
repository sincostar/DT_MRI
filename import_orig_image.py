# Plot the mean image because we have no anatomic data
"""
func_filename = "D:\\DT\\BrainMRI\\BrainMRI\\sub-28675_T1_orig.nii.gz"
mean_img = image.mean_img(func_filename)
weight_img = nib.load('sub-28675_T1_orig.nii.gz')
plot_stat_map(weight_img, mean_img, title='SVM weights')
show()


plotting.plot_glass_brain("D:\\DT\\BrainMRI\\BrainMRI\\sub-28675_T1_orig.nii.gz")
"""

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import numpy as np

brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_brain.nii.gz'
orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28675_T1_orig.nii.gz'

img = nib.load(brain_filename)
print(img)
print(img.header['db_name'])  # 输出头信息

width, height, queue = img.dataobj.shape

print("width: " + str(width))
print("height: " + str(height))
print("queue: " + str(queue))

# OrthoSlicer3D(img.dataobj).show()

img_arr = img.dataobj[:, :, 105].copy()


for i in range(width):
    for j in range(height):
        if img_arr[i, j]:
            img_arr[i, j] = 255

print(img_arr.shape)
plt.imshow(img_arr, cmap='gray')
plt.show()