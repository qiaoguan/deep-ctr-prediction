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

def esmm_model_fn(features, labels, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  last_ctr_layer = build_deep_layers(net, params)
  last_cvr_layer = build_deep_layers(net, params)

  #head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
  head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                  n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
  ctr_logits = tf.layers.dense(last_ctr_layer, units=head.logits_dimension,
                               kernel_initializer=tf.glorot_uniform_initializer())
  cvr_logits = tf.layers.dense(last_cvr_layer, units=head.logits_dimension,
                               kernel_initializer=tf.glorot_uniform_initializer())
  ctr_preds = tf.sigmoid(ctr_logits)
  cvr_preds = tf.sigmoid(cvr_logits)
  ctcvr_preds = tf.multiply(ctr_preds, cvr_preds)

  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  ctr_label = labels['ctr_label']
  cvr_label = labels['cvr_label']

  user_id = features['user_id']
  click_label = features['label']
  conversion_label = features['is_conversion']


  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'ctr_preds': ctr_preds,
      'cvr_preds': cvr_preds,
      'ctcvr_preds': ctcvr_preds,
      'user_id': user_id,
      'click_label': click_label,
      'conversion_label': conversion_label
    }
    export_outputs = {
      'regression': tf.estimator.export.RegressionOutput(predictions['cvr_preds'])  #线上预测需要的
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  else:
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits))
    ctcvr_loss = tf.reduce_sum(tf.losses.log_loss(labels=cvr_label, predictions=ctcvr_preds))
    loss = ctr_loss + ctcvr_loss  # loss这儿可以加一个参数，参考multi-task损失的方法

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  """
  return head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
  )
  """

