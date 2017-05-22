# -*- coding:utf-8 -*-
import glob
import os.path
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Inception-v3 model bottleneck tensor size
BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# Input Image Name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# path of Inception-v3 model
MODEL_DIR = ''

# Download Inception-v3 fileName
MODEL_FILE = 'tensorflow_inception_graph.pb'

# Cache file path
CACHE_DIR = 'tmp/bottleneck'

INPUT_DATA = 'flower_photos'

# Validation percent
VALIDATION_PERCENTAGE = 10

# Test percent
TEST_PERCENTAGE = 10

# parameters of neural network
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# read data from INPUT_DATA, separate training validation and testing
def create_image_lists(testing_percentage, validation_percentage):
    # result is a dictionary, its keys are 'warning' 'alert' and 'allowed', its values are name of images
    # get all directory and sub_directorys
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # first one of sub_dirs is root and pass it
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # get all images like jpg jpeg JPG JPEG and later there still needs png and PNG and so on
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # focus on the images and get their full path
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # put path into file_list
            file_list.extend(glob.glob(file_glob))
            # if null pass it
            if not file_list:
                continue

        # get classify name
        # why lower?
        label_name = dir_name.lower()
        # initial current TrainingData TestingData and ValidationData
        training_images = []
        testing_images = []
        validation_images = []
        # operate with every image file
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # divide into TrainingData TestingData or ValidationData by random
            # if wanna change the percentage of each part, change the parameter above
            chance = np.random.randint(100)
            # print('chance: ', chance)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage) and chance >= validation_percentage:
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # save result in result, result is dictionary
        # training_images testing_images and validation_images are lists
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    # print('result:', result)
    return result


# image_lists tells all image information
# image_dir tells root path
# label_name tells class
# index tells image number
# category tells where it belongs trainingData testingData or validationData
def get_image_path(image_lists, image_dir, label_name, index, category):
    # print('category: ', category)
    # get all image information
    # print(label_name)
    label_lists = image_lists[label_name]
    # print('label_lists:', label_lists)
    category_list = label_lists[category]
    # print('category_list:', category_list)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    # print('full_path: ', full_path)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    # print(image_lists, CACHE_DIR, label_name, index, category + 'txt')
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, feed_dict={image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    # print('bottleneck_values: ', bottleneck_values)
    return bottleneck_values


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(
        image_lists, label_name, index, category
    )
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(
            image_lists, INPUT_DATA, label_name, index, category
        )
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor
        )
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # print('bottleneck_values', bottleneck_values)
    return bottleneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor
        )
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    print('random_cached_bottlenecks: ', len(bottlenecks))
    # print('ground_truths: ', ground_truths)
    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    print('label_name_list', label_name_list)
    for label_index, label_name, in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
                image_lists[label_name][category]
        ):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    print('test_bottlenecks: ', len(bottlenecks))
    # print('ground_truths: ', ground_truths)
    return bottlenecks, ground_truths


def main(_):
    # read image_lists by function create_image_lists
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    # n_classes is count of candidates
    n_classes = len(image_lists.keys())
    # print('there are %d candidate', %n_classes)

    # import train model
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        # create tf.GraphDef class
        graph_def = tf.GraphDef()
        # read train model
        graph_def.ParseFromString(f.read())

    # very important function: read defined graph
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]
    )
    # input tensorflow
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder'
    )

    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes], name='GroundTruthInput'
    )

    with tf.name_scope('final_training_ops'):
        # weights
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        # biases
        biases = tf.Variable(tf.zeros([n_classes]))
        # logits: bottleneck_input * weights + biases
        logits = tf.matmul(bottleneck_input, weights) + biases

        final_tensor = tf.nn.softmax(logits)

    # cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('logs/', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH,
                                                                                  'training', jpeg_data_tensor,
                                                                                  bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes,
                                                                                                image_lists, BATCH,
                                                                                                'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                                           ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled ''%d examples = %.1f%%' % (
                    i, BATCH, validation_accuracy * 100))

            test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                                                                       bottleneck_tensor)
            print(len(test_ground_truth))
            test_accuracy = sess.run(evaluation_step,
                                     feed_dict={bottleneck_input: test_bottlenecks,
                                                ground_truth_input: test_ground_truth})
            print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
            # writer.add_summary(test_accuracy, i)


if __name__ == '__main__':
    # set_trace()
    tf.app.run()
