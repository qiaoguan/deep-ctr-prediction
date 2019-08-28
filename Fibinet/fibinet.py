# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import dice
import collections
import random
'''
A tensorflow implementation of Fibinet

Tongwen Huang et all. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction,"  In RecSys 19.
@author: Qiao
'''
def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
    return net

def build_Bilinear_Interaction_layers(net, params):
    # Build Bilinear-Interaction Layer

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)(b,f,k)

    element_wise_product_list = []
    count = 0
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            with tf.variable_scope('weight_', reuse=tf.AUTO_REUSE):
                weight = tf.get_variable(name='weight_' + str(count), shape=[dimension, dimension],
                                         initializer = tf.glorot_normal_initializer(seed = random.randint(0,1024)),
                                         dtype=tf.float32)
            element_wise_product_list.append(
                tf.multiply(tf.matmul(feature_embeddings[:, i, :], weight), feature_embeddings[:, j, :]))
                #tf.multiply(feature_embeddings[:, i, :], feature_embeddings[:, j, :]))
            count += 1
    element_wise_product = tf.stack(element_wise_product_list)  # (f*(f-1)/2,b,k)(把它们组合成一个tensor)
    element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2],
                                        name="element_wise_product")  # (b, f*(f-1)/2, k)

    bilinear_output = tf.layers.flatten(element_wise_product)  #(b, f*(f-1)/2*k)
    return bilinear_output

def build_SENET_layers(net, params):
    # Build SENET Layer

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    reduction_ratio = params['reduction_ratio']
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)(b,f,k)
    original_feature = feature_embeddings
    if params['pooling'] == "max":
        feature_embeddings = tf.reduce_max(feature_embeddings, axis=2)   # (b,f) max pooling
    else:
        feature_embeddings = tf.reduce_mean(feature_embeddings, axis=2)  #(b,f) mean pooling

    reduction_num = max(column_num/reduction_ratio, 1)     # f/r
    """
    weight1 = tf.get_variable(name='weight1', shape=[column_num, reduction_num],
                             initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)),
                             dtype=tf.float32)
    weight2 = tf.get_variable(name='weight2', shape=[reduction_num, column_num],
                              initializer=tf.glorot_normal_initializer(seed=random.randint(0, 1024)),
                              dtype=tf.float32)
    """
    att_layer = tf.layers.dense(feature_embeddings, units=reduction_num, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())     # (b, f/r)
    att_layer = tf.layers.dense(att_layer, units=column_num, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer())  # (b, f)
    senet_layer = original_feature * tf.expand_dims(att_layer, axis=-1)    # (b, f, k)
    senet_output = tf.layers.flatten(senet_layer)  # (b, f*k)

    return senet_output

def _check_fm_columns(feature_columns):
  if isinstance(feature_columns, collections.Iterator):
    feature_columns = list(feature_columns)
  column_num = len(feature_columns)
  if column_num < 2:
    raise ValueError('feature_columns must have as least two elements.')
  dimension = -1
  for column in feature_columns:
    if dimension != -1 and column.dimension != dimension:
      raise ValueError('fm_feature_columns must have the same dimension.')
    dimension = column.dimension
  return column_num, dimension

def fibinet_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    senet_layer = build_SENET_layers(net, params)
    combination_layer = tf.concat([build_Bilinear_Interaction_layers(net, params),
                                   build_Bilinear_Interaction_layers(senet_layer, params)], axis=1)

    last_layer = build_deep_layers(combination_layer, params)
    # head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(last_layer, units=head.logits_dimension,
                             kernel_initializer=tf.glorot_uniform_initializer())
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    preds = tf.sigmoid(logits)
    user_id = features['user_id']
    label = features['label']

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': preds,
            'user_id': user_id,
            'label': label
        }
        export_outputs = {
            'regression': tf.estimator.export.RegressionOutput(predictions['probabilities'])
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
    )

