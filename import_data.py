# import import_orig_image
import re
import os


file_path1 = 'D:/DT/BrainMRI/BrainMRI'

global IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT

IMAGE_HEIGHT = 100
IMAGE_LENGTH = 125
IMAGE_WIDTH = 100


def import_data(file_path):
    for file_name in os.listdir(file_path):
        # print(file_name)
        if re.match(".*T1_orig.*", file_name) is not None:
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)
        elif re.match(".*T1_brain.*", file_name) is not None:
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)

import_data(file_path1)