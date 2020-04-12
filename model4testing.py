from core.trainer_tf import Trainer
from core.data_provider import DataProvider
from models.model import SimpleTFModel
from nets_tf.unet3d import UNet3D
from core.data_processor import SimpleImageProcessor
import glob
import tensorflow as tf
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import matplotlib
matplotlib.use('TkAgg')
from import_brain_image import export_mask
from import_brain_image import export_brain
import scipy.ndimage as ndi
import re
import matplotlib.pyplot as plt
import numpy as np
from models.model_reg import RegressionModel


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

org_suffix = '_brain.nii.gz'
lab_suffix = '_brain_restore.nii.gz'
train_set = glob.glob('data/distortion_data/train/*_brain.nii.gz')
valid_set = glob.glob('data/distortion_data/validation/*_brain.nii.gz')
test_set1 = glob.glob('data/distortion_data/test/*_brain.nii.gz')
test_set2 = glob.glob('data/distortion_data/new_test/*_brain.nii.gz')
test_result_set2 = glob.glob('data/distortion_data/new_test/*_brain_restore.nii.gz')

img_save_path = 'data/out_image/'

u_net = UNet3D(n_class=1, n_layer=3, root_filters=16, use_bn=True)

model = RegressionModel(u_net, org_suffix, lab_suffix, dropout=0)
trainer = Trainer(model)
trainer.restore('results/test3/ckpt/final')

pre = {org_suffix: [('channelcheck', 1)],
       lab_suffix: [('channelcheck', 1)]}
processor = SimpleImageProcessor(pre=pre)

train_provider = DataProvider(train_set, [org_suffix, lab_suffix],
                              is_pre_load=False,
                              processor=processor)
validation_provider = DataProvider(valid_set, [org_suffix, lab_suffix],
                                   is_pre_load=False,
                                   processor=processor)
test_provider = DataProvider(test_set1, [org_suffix, lab_suffix],
                             is_pre_load=False,
                             processor=processor)
new_test_provider = DataProvider(test_set2, [org_suffix, lab_suffix],
                                 is_pre_load=False,
                                 processor=processor)
'''
eval_dict = trainer.eval(test_provider ,
                         batch_size=10)
'''
eval_dict, eval_img_list = trainer.eval(new_test_provider, batch_size=1, need_imgs=True)
for i in range(len(eval_dict['loss'])):
    if eval_dict['loss'][i] > 0.002:
        x = eval_img_list['class 0'][i][:, 128: 192, :]
        x = (x - x.min()) / (x.max() - x.min())
        print(test_result_set2[i])
        print(eval_dict['loss'][i])
        img = nib.load(test_result_set2[i])
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(OrthoSlicer3D(x))
        # plt.subplot(1, 2, 2)
        # plt.imshow(OrthoSlicer3D(img.dataobj))
        # plt.show()
        # OrthoSlicer3D(x).link_to(OrthoSlicer3D(img.dataobj))
        # OrthoSlicer3D(x)
        img_arr = img.dataobj[:, :, :].copy()
        y = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        OrthoSlicer3D(y - x).show()

print(np.mean(eval_dict['loss']))
print(np.std(eval_dict['loss']))
'''
mask_list = []
# to get the test image from the eval image eval dict and store into the mask_list
for img in eval_image_dict['argmax']:
    stand_size = img.shape[-1]
    for i in range(int(img.shape[0]/stand_size)):
        mask = img[i*stand_size:(1+i)*stand_size, 2*stand_size:3*stand_size, :]
        mask_list.append(mask)

mask_index = 0
pattern = re.compile(r'\d{5}')
for data_path in train_provider._file_list:
    mask = mask_list[mask_index]
    mask_index += 1
    img = nib.load(data_path)
    brain_index = pattern.findall(data_path)[0]
    img_data = img.dataobj[:, :, :].copy()
    brain_arr = export_brain(mask, img_data)
    nib_img = nib.Nifti1Image(brain_arr, img.affine)
    nib.save(nib_img, "data/distortion_data/" + brain_index + ".nii.gz")'''
"""
# mask = ndi.zoom(mask_list[0], 4)
mask = mask_list[0]
img = nib.load('D:/DT/BrainMRI/BrainMRI/sub-28783_T1_orig.nii.gz')
# img = nib.load('data/test/28783_orig.nii.gz')
img_data = img.dataobj[:, :, :].copy()
img_shape = img_data.shape
img_mask = export_mask(mask, img_shap e)

for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        for k in range(img_shape[2]):
            if img_mask[i, j, k] < 100:
                img_data[i, j, k] = 0
nib_img = nib.Nifti1Image(img_data, img.affine)
OrthoSlicer3D(img_data).show()
nib.save(nib_img, img_save_path + "28783.nii.gz")
"""