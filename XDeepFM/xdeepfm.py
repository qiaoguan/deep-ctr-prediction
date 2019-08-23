# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import dice
import collections
'''
A tensorflow implementation of xDeepFM

Jianxun Lian et all.  "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems,"  In KDD,2018.

@author: Qiao
'''

def _build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
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

def _build_fm_layers(net, params):
    # Build the FM layers, sized according to the 'hidden_units' param.

    # cal FM part
    column_num, dimension = _check_fm_columns(params['feature_columns'])
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  # (batch_size,column_num, embedding_size)

    # sum_square part
    summed_feature_embeddings = tf.reduce_sum(feature_embeddings, 1)  # (batch_size,embedding_size)
    summed_square_feature_embeddings = tf.square(summed_feature_embeddings)

    # squre-sum part
    squared_feature_embeddings = tf.square(feature_embeddings)
    squared_sum_feature_embeddings = tf.reduce_sum(squared_feature_embeddings, 1)

    fm_second_order = 0.5 * tf.subtract(summed_square_feature_embeddings, squared_sum_feature_embeddings)

    return fm_second_order

def _build_xdeepfm_layers(net, params):
    # Build xdeepFM layers, sized according to the 'hidden_units' param.
    field_num, dim = _check_fm_columns(params['feature_columns'])

    nn_input = tf.reshape(net, (-1, field_num, dim))  #(batch_size,field_num, dim)

    final_len = 0
    field_nums = []
    field_nums.append(int(field_num))
    hidden_nn_layers = []
    hidden_nn_layers.append(nn_input)
    final_result = []
    split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)  # 把每个field分离成dim个tensor

    for idx, layer_size in enumerate(params['cross_layer_sizes']):
        split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)  # (dim, batch_size, field_nums[-1], 1)
        dot_result_m = tf.matmul(split_tensor0, split_tensor,
                                 transpose_b=True)  # (dim,batch_size,field_num,field_nums[-1])
        dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
        dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # (instance_cnt,dim,field_nums[0]*field_nums[-1])

        filters = tf.get_variable(name="f_" + str(idx),
                                  shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                  dtype=tf.float32)

        curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')  # [batch_size,dim,layer_size]

        curr_out = tf.nn.relu(curr_out)

        curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # (batch_size, layer_size, dim)

        direct_connect = curr_out
        next_hidden = curr_out
        final_len += int(layer_size)
        field_nums.append(int(layer_size))

        final_result.append(direct_connect)
        hidden_nn_layers.append(next_hidden)

    result = tf.concat(final_result, axis=1)   #(batch_size, final_len, dim)
    result = tf.reduce_sum(result, -1)         #(batch_size, final_len)

    '''
    w_nn_output = tf.get_variable(name='w_nn_output',
                                  shape=[final_len, 1],
                                  dtype=tf.float32)
    b_nn_output = tf.get_variable(name='b_nn_output',
                                  shape=[1],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())

    exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

    return exFM_out
    '''
    return result


def xdeepfm_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    last_deep_layer = _build_deep_layers(net, params)
    last_xdeepfm_layer = _build_xdeepfm_layers(net, params)

    if params['use_xdeepfm']:
        print('--use xdeepfm layer--')
        last_layer = tf.concat([last_deep_layer, last_xdeepfm_layer], 1)
    else:
        last_layer = last_deep_layer

    # head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(last_layer, units=head.logits_dimension,
                             kernel_initializer=tf.glorot_uniform_initializer())
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    # optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
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
    # loss = focal_loss(logits=logits, labels=labels, alpha=0.5, gamma=6, beta=1)
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