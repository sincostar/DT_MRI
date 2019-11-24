import import_data
import numpy as np


file_path = 'D:/DT/BrainMRI/BrainMRI'


# import_data.import_data(file_path)

orig_image_data = np.load(file='orig_image_data.npy')
brain_image_data = np.load(file='brain_image_data.npy')

train_orig_image_data = orig_image_data[:100, :, :, :]
train_brain_image_data = orig_image_data[:100, :, :, :]

test_orig_image_data = orig_image_data[101:, :, :, :]
test_brain_image_data = orig_image_data[101:, :, :, :]

