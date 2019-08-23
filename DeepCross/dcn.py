#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses

def build_deep_layers(net, params):
  # Build the hidden layers, sized according to the 'hidden_units' param.
  
  for num_hidden_units in params['hidden_units']:
    net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu,
                          kernel_initializer=tf.glorot_uniform_initializer())
  return net


def build_cross_layers(x0, params):
  num_layers = params['num_cross_layers']
  x = x0
  for i in range(num_layers):
    x = cross_layer(x0, x, 'cross_{}'.format(i))
  return x

def cross_layer(x0, x, name):
  with tf.variable_scope(name):
    input_dim = x0.get_shape().as_list()[1]
    w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
    return x0 * xb + b + x

def dcn_model_fn(features, labels, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  last_deep_layer = build_deep_layers(net, params)
  last_cross_layer = build_cross_layers(net, params)

  if params['use_cross']:
    print('--use cross layer--')
    last_layer = tf.concat([last_deep_layer, last_cross_layer], 1)
  else:
    last_layer = last_deep_layer

  #head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
  head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                  n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
  logits = tf.layers.dense(last_layer, units=head.logits_dimension, kernel_initializer=tf.glorot_uniform_initializer())
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  #optimizer = tf.contrib.opt.GGTOptimizer(learning_rate=params['learning_rate'])
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

  if mode == tf.estimator.ModeKeys.TRAIN:

  return head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
  )

