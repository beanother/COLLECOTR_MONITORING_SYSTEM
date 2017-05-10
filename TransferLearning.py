# -*- coding:utf-8 -*-
import glob
import os.path

import numpy as np

# Inception-v3 model bottleneck tensor size
BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# Input Image Name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# path of Inception-v3 model
MODEL_DIR = '/path/to/model'

# Download Inception-v3 fileName
MODEL_FILE = 'classify_image_graph_def.pb'

# Cache file path
CACHE_DIR = '/tmp/bottleneck'

INPUT_DATA = '/path/to/brush_data'

# Validation percent
VALIDATION_PERCENTAGE = 0

# Test percent
TEST_PERCENTAGE = 0

# parameters of neural network
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# read data from INPUT_DATA, separate training validation and testing
def create_image_lists(testing_percentage, validation_percentage):
    # result is a dictionary, its keys are 'warning' 'alert' and 'allowed', its values are name of images
    result = {}  # get all directory and sub_directorys
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # first one of sub_dirs is root
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # get all images
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    for extension in extensions:
        file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

    # get classify name
    label_name = dir_name.lower()
    # initial current TrainingData TestingData and ValidationData
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
        base_name = os.path.basename(file_name)
        # divide into TrainingData TestingData or ValidationData by random
        chance = np.random.randint(100)
        if chance < validation_percentage:
            validation_images.append(base_name)
        elif chance < (testing_percentage + validation_percentage):
            testing_images.append(base_name)
        else:
            training_images.append(base_name)

    # save result in result
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }

    return result
