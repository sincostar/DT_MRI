import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
from nibabel.viewers import OrthoSlicer3D
import import_orig_image
import import_brain_image
import numpy as np
import cv2

target_brain_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28677_T1_brain.nii.gz'
target_orig_filename = 'D:/DT/BrainMRI/BrainMRI/sub-28677_T1_orig.nii.gz'


def show_orig_and_brain_contact_ratio(orig_filename, brain_filename, height):
    orig = import_orig_image.load_orig(orig_filename)[:, :, height].astype(np.uint8)
    brain = import_brain_image.load_brain(brain_filename)[:, :, height].astype(np.uint8)
    print(brain.size)
    # plt.imshow(orig, cmap='Reds')
    # plt.imshow(brain)
    # plt.show()
    b = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    g = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    r = np.zeros((128, 128), dtype=np.uint8)
    img = cv2.merge([orig, brain, r])

    enlarge_img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("merged_1", enlarge_img)
    cv2.waitKey(0)
    cv2.destroyWindow('test')


show_orig_and_brain_contact_ratio(target_orig_filename, target_orig_filename, 50)
