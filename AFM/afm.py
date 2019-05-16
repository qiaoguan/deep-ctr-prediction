# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import dice
import collections
'''
A tensorflow implementation of AFM

Jun Xiao et all.  "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks,"  In IJCAI,2017.
@author: Qiao
'''

def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        # net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
        #net = dice(net, name='dice_' + str(layer_id))
    return net

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

def _initialize_weights(params):
    all_weights = dict()
    glorot = np.sqrt(2.0 / (int(params['hidden_factor'][0]) + int(params['hidden_factor'][1])))
    all_weights['attention_b'] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(1, int(params['hidden_factor'][0]))), dtype=np.float32,
        name="attention_b")  # 1 * AK
    all_weights['attention_p'] = tf.Variable(
        np.random.normal(loc=0, scale=1, size=(int(params['hidden_factor'][0]))), dtype=np.float32, name="attention_p")  # AK

    return all_weights

def build_afm_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)(b,m,k)
    """
    # sum_square part
    summed_feature_embeddings = tf.reduce_sum(feature_embeddings, 1)  # (batch_size,embedding_size)
    summed_square_feature_embeddings = tf.square(summed_feature_embeddings)

    # squre-sum part
    squared_feature_embeddings = tf.square(feature_embeddings)
    squared_sum_feature_embeddings = tf.reduce_sum(squared_feature_embeddings, 1)

    fm_embeddings = 0.5 * tf.subtract(summed_square_feature_embeddings, squared_sum_feature_embeddings)
    """
    weights = _initialize_weights(params)
    element_wise_product_list = []
    count = 0
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            element_wise_product_list.append(
                tf.multiply(feature_embeddings[:, i, :], feature_embeddings[:, j, :]))
            count += 1
    element_wise_product = tf.stack(element_wise_product_list)  # (m*(m-1)/2,b,k)(把它们组合成一个tensor)
    element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2],
                                             name="element_wise_product")  # (b, m*(m-1)/2, k)
    interactions = tf.reduce_sum(element_wise_product, 2,
                                      name="interactions")
    # _________ MLP Layer / attention part _____________
    num_interactions = column_num*(column_num-1)/2

    #attention_mul = tf.reshape(tf.matmul(tf.reshape(element_wise_product, shape=[-1, self.hidden_factor[1]]), \
    #                weights['attention_W']), shape=[-1, num_interactions, hidden_factor[0]])  # (None, M*(M-1)/2, hidden_factor[0])
    attention_mul = tf.layers.dense(element_wise_product, units=int(params['hidden_factor'][0]),
                                    kernel_initializer=tf.glorot_uniform_initializer())   #(b, m*(m-1)/2, hidden_factor[0])
    attention_relu = tf.reduce_sum(tf.multiply(weights['attention_p'],
                        tf.nn.relu(attention_mul + weights['attention_b'])), 2, keep_dims=True)  # None * (M'*(M'-1)) * 1     weights['attention_p']:size(hidden_factor[0])
    attention_out = tf.nn.softmax(attention_relu)  # (None, M*(M-1)/2,1)

    AFM = tf.reduce_sum(tf.multiply(attention_out, element_wise_product), 1, name="afm")  # [b, k]

    """
    AFM_FM = tf.reduce_sum(element_wise_product, 1, name="afm_fm")  # None * K
    AFM_FM = AFM_FM / num_interactions
    #AFM = tf.nn.dropout(AFM, self.dropout_keep[1])  # dropout
    """
    return AFM


def afm_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    last_deep_layer = build_deep_layers(net, params)

    last_layer = build_afm_layers(net, params)
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

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
    )

