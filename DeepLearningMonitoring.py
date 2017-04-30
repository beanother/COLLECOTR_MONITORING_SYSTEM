# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image

# 读取训练集数据
fileName = [""]

# 创建tf输入对象
fileNameQue = tf.train.string_input_producer(fileName)
reader = tf.TFRecordReader()
key, value = reader.read(fileNameQue)
print("key:", key)
print("value: ", value)

# 返回features
features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                    'img': tf.FixedLenFeature([], tf.string), })

img = tf.decode_raw(features["img"], tf.uint8)
label = tf.cast(features["label"], tf.int32)
print("img: ", img)
print("label: ", label)
course_2_tf_nn.py
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)