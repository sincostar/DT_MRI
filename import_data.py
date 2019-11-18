# import import_orig_image
import re
import os


file_path1 = 'D:/DT/BrainMRI/BrainMRI'

global IMAGE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT

IMAGE_HEIGHT = 100
IMAGE_LENGTH = 125
IMAGE_WIDTH = 100


def import_data(file_path):
    file_index = 0
    for file_name in os.listdir(file_path):
        # print(file_name)
        pattern = re.compile(r'\d\d\d\d\d')
        if re.match(".*T1_orig.*", file_name) is not None and file_index == int(pattern.findall(file_name)[0]):
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)

        elif re.match(".*T1_brain.*", file_name) is not None:
            # import_orig_image.load_orig(file_path + '/' + file_name)
            print(file_name)
            file_index = int(pattern.findall(file_name)[0])

import_data(file_path1)