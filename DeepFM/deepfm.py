# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
import collections
from tensorflow.python.ops import metrics as metrics_lib
from metric import cal_group_auc

def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for num_hidden_units in params['hidden_units']:
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

def dfm_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns']) # shape(batch_size, column_num * embedding_size)
    last_deep_layer = build_deep_layers(net, params)

    column_num, dimension = _check_fm_columns(params['feature_columns'])
    feature_embeddings = tf.reshape(net, (-1, column_num, dimension))  #(batch_size,column_num, embedding_size)

    # sum_square part
    summed_feature_embeddings = tf.reduce_sum(feature_embeddings, 1)  # (batch_size,embedding_size)
    summed_square_feature_embeddings = tf.square(summed_feature_embeddings)

    # squre-sum part
    squared_feature_embeddings = tf.square(feature_embeddings)
    squared_sum_feature_embeddings = tf.reduce_sum(squared_feature_embeddings, 1)

    fm_second_order = 0.5 * tf.subtract(summed_square_feature_embeddings, squared_sum_feature_embeddings)
    #print(tf.shape(fm_second_order))
    #print(fm_second_order.get_shape())

    if params['use_fm']:
        print('--use fm--')
        last_layer = tf.concat([fm_second_order, last_deep_layer], 1)
    else:
        last_layer = last_deep_layer
    #head = tf.contrib.estimator.binary_classification_head(loss_reduction=losses.Reduction.SUM)
    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                    n_classes=2, weight_column=None, label_vocabulary=None, loss_reduction=losses.Reduction.SUM)
    logits = tf.layers.dense(last_layer, units=head.logits_dimension,
                             kernel_initializer=tf.glorot_uniform_initializer())
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

    preds = tf.sigmoid(logits)
    #print(tf.shape(preds))
    #print(preds.get_shape())
    user_id = features['user_id']
    label = features['label']
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels['class'],
                                       predictions=tf.to_float(tf.greater_equal(preds, 0.5)))
        auc = tf.metrics.auc(labels['class'], preds)
        label_mean = metrics_lib.mean(labels['class'])
        prediction_mean = metrics_lib.mean(preds)

        prediction_squared_difference = tf.math.squared_difference(preds, prediction_mean[0])
        prediction_squared_sum = tf.reduce_sum(prediction_squared_difference)
        num_predictions = tf.to_float(tf.size(preds))
        s_deviation = tf.sqrt(prediction_squared_sum/num_predictions), accuracy[0]     #标准差

        c_variation = tf.to_float(s_deviation[0]/prediction_mean[0]), accuracy[0]    #变异系数

        #group_auc = tf.to_float(cal_group_auc(labels['class'], preds, labels['user_id'])), accuracy[0] # group auc

    
        metrics = {'accuracy': accuracy, 'auc': auc, 'label/mean': label_mean, 'prediction/mean': prediction_mean,
                   'standard deviation': s_deviation, 'coefficient of variation': c_variation}
         #          'group auc': group_auc}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        tf.summary.scalar('label/mean', label_mean[1])
        tf.summary.scalar('prediction/mean', prediction_mean[1])
        tf.summary.scalar('s_deviation', s_deviation[1])
        tf.summary.scalar('c_variation', c_variation[1])
        #tf.summary.scalar('group_auc', group_auc[1])
        
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['class'], logits=logits))
        #print(tf.shape(loss))
        #print(loss.get_shape())
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': preds,
            'user_id': user_id,
            'label': label
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
    )

