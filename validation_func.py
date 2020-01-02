import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
from nibabel.viewers import OrthoSlicer3D


filename = 'data/validation/28779_brain.nii.gz'

img = nib.load(filename)

OrthoSlicer3D(img.dataobj).show()