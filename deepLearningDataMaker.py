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

            img = Image.open(img_path)
            img = img.resize((255, 255))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()

training_data_maker(file_name="COLLECTOR_BRUSH")


def training_data_reader(file_name=None, path=None):
    # read learning file
    if file_name is None:
        file_name = "COLLECTOR_BRUSH.tfrecords"
    else:
        file_name = file_name

    if path is None:
        cwd = os.curdir
    else:
        cwd = path

    print(cwd + os.sep + file_name)
    fileNameQue = tf.train.string_input_producer(cwd + os.sep + file_name)
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    # 返回features
    features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                        'img': tf.FixedLenFeature([], tf.string), })

    img = tf.decode_raw(features["img"], tf.uint8)
    label = tf.cast(features["label"], tf.int32)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

training_data_reader()
