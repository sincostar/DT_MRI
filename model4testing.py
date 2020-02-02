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
import scipy.ndimage as ndi


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

org_suffix = '_orig.nii.gz'
lab_suffix = '_brain.nii.gz'
test_set1 = glob.glob('data/test/*_orig.nii.gz')
test_set2 = glob.glob('data/new_test/*_orig.nii.gz')

img_save_path = 'data/out_image/'

u_net = UNet3D(n_class=2, n_layer=3, root_filters=16, use_bn=True)

model = SimpleTFModel(u_net, org_suffix, lab_suffix, dropout=0, loss_function={'cross-entropy': 1.},
                      weight_function=None)
trainer = Trainer(model)
trainer.restore('results/best_ckpt/final')

pre = {org_suffix: [('channelcheck', 1)],
       lab_suffix: [('one-hot', 2), ('channelcheck', 2)]}
processor = SimpleImageProcessor(pre=pre)

test_provider = DataProvider(test_set1, [org_suffix, lab_suffix],
                             is_pre_load=False,
                             processor=processor)
'''
eval_dict = trainer.eval(test_provider,
                         batch_size=10)
'''
eval_dict, eval_image_dict = trainer.eval(test_provider,
                                          batch_size=10,
                                          print_str=False,
                                          need_imgs=True,
                                          print_all_results=True)

mask_list = []
for img in eval_image_dict['argmax']:
    stand_size = img.shape[-1]
    for i in range(int(img.shape[0]/stand_size)):
        mask = img[i*stand_size:(1+i)*stand_size, 2*stand_size:3*stand_size, :]
        mask_list.append(mask)

# mask = ndi.zoom(mask_list[0], 4)
mask = mask_list[0]
img = nib.load('D:/DT/BrainMRI/BrainMRI/sub-28783_T1_orig.nii.gz')
# img = nib.load('data/test/28783_orig.nii.gz')
img_data = img.dataobj[:, :, :].copy()
img_shape = img_data.shape
img_mask = export_mask(mask, img_shape)
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        for k in range(img_shape[2]):
            if img_mask[i, j, k] < 100:
                img_data[i, j, k] = 0
nib_img = nib.Nifti1Image(img_data, img.affine)
OrthoSlicer3D(img_data).show()
nib.save(nib_img, img_save_path + "28783.nii.gz")
