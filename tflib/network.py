import tflib
import tflib.ops
import tensorflow as tf
import numpy as np


def im2latex_cnn(X, num_feats, bn, train_mode=True):
    KERNEL_SIZE = 3

    X = X-128.
    X = X/128.

    #padding (2,2) in first step?

    #X = tf.nn.relu(tflib.ops.conv2d('conv1', X, 3, 1, 1, num_feats, pad = 'SAME', bias=False)) 
    #X = tflib.ops.max_pool('pool1', X, k=2, s=2)

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X,pool_size=2,strides=2)
    


    #X = tf.nn.relu(tflib.ops.conv2d('conv2', X, 3, 1, num_feats, num_feats*2, pad = 'SAME', bias=False))
    #X = tflib.ops.max_pool('pool2', X, k=2, s=2)

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats*2,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X,pool_size=2,strides=2)



    #X = tf.nn.relu(tflib.ops.conv2d('conv3', X, 3, 1, num_feats*2, num_feats*4,  batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    if bn:
        X = tf.layers.batch_normalization(X, training=train_mode)
    X = tf.nn.relu(X)



    #X = tf.nn.relu(tflib.ops.conv2d('conv4', X, 3, 1, num_feats*4, num_feats*4, pad = 'SAME', bias=False))
    #X = tflib.ops.max_pool('pool4', X, k=(1,2), s=(1,2))

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats*4,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X,pool_size=(1,2),strides=(1,2))



    #X = tf.nn.relu(tflib.ops.conv2d('conv5', X, 3, 1, num_feats*4, num_feats*8, batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))
    #X = tflib.ops.max_pool('pool5', X, k=(2,1), s=(2,1))

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats*4,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    if bn:
        X = tf.layers.batch_normalization(X, training=train_mode)
    X = tf.nn.relu(X)
    X = tf.layers.max_pooling2d(X,pool_size=(2,1),strides=(2,1))



    #X = tf.nn.relu(tflib.ops.conv2d('conv6', X, 3, 1, num_feats*8, num_feats*8, batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))

    filter_values = tf.contrib.layers.xavier_initializer()
    X = tf.layers.conv2d(X,num_feats,KERNEL_SIZE,kernel_initializer=filter_values,padding='SAME',use_bias=False)
    if bn:
        X = tf.layers.batch_normalization(X, training=train_mode)
    X = tf.nn.relu(X)

    return X
