# -*- coding:utf-8 -*-

import os

import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到


def training_data_maker(file_name=None, path=None):
    # make training data
    if path is None:
        cwd = os.curdir
        print("当前文件路径： ", cwd)
    else:
        cwd = path

    classes = {'allowed', 'warning', 'alert'}  # 设定 3 类图片

    if file_name is None:
        file_name = 'COLLECTOR_TRAIN'
    else:
        file_name = file_name

    writer = tf.python_io.TFRecordWriter(file_name + ".tfrecords")  # 要生成的文件

    for index, name in enumerate(classes):
        class_path = cwd + os.sep + name + os.sep
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的路径
            # print(img_path)
            img = Image.open(img_path)
            img = img.resize((255, 255))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def training_data_reader(file_name=None, path=None):
    # read learning file
    if file_name is None:
        file_name = "COLLECTOR_TRAIN.tfrecords"
    else:
        file_name = file_name

    if path is None:
        cwd = os.curdir
    else:
        cwd = path

    # print(cwd + os.sep + file_name)
    fileNameQue = tf.train.string_input_producer([cwd + os.sep + file_name])
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    # 返回features
    features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                        'img_raw': tf.FixedLenFeature([], tf.string), })

    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img = tf.reshape(img, [255,255,3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features["label"], tf.int32)

    return img, label

training_data_maker()

img, label = training_data_reader()
img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=30,
                                                capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l = sess.run([img_batch, label_batch])
        print(val.shape, l)
