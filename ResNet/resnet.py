# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import dice
import collections


def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
    return net

def build_residual_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.
    net = tf.layers.batch_normalization(net)
    shortcut = net
    residual = tf.layers.dense(net, units=256, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())
    net = tf.concat([shortcut, residual], 1)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dense(net, units=256, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer())

    net = tf.layers.batch_normalization(net)
    shortcut = net
    residual = tf.layers.dense(net, units=128, activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_uniform_initializer())
    net = tf.concat([shortcut, residual], 1)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dense(net, units=128, activation=tf.nn.relu,
                          kernel_initializer=tf.glorot_uniform_initializer())

    return net


def resnet_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    last_layer = build_residual_layers(net, params)
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

