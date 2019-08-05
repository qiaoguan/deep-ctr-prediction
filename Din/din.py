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

def attention_layer(querys, keys, keys_id):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of clicked creativeid for each sample)
        keys_id:     [Batchsize, max_seq_len]
    """

    keys_length = tf.shape(keys)[1] # padded_dim
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
    querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])

    net = tf.concat([keys, keys - querys, querys, keys*querys], axis=-1)
    for units in [32,16]:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        # shape(batch_size, max_seq_len, 1)
    outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  #shape(batch_size, 1, max_seq_len)
    
    scores = outputs
    #key_masks = tf.expand_dims(tf.cast(keys_id > 0, tf.bool), axis=1)  # shape(batch_size, 1, max_seq_len) we add 0 as padding
    # tf.not_equal(keys_id, '0')  如果改成str
    key_masks = tf.expand_dims(tf.not_equal(keys_id, '0'), axis=1)
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)
    scores = scores / (embedding_size ** 0.5)       # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)    #(batch_size, 1, embedding_size)
    outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")   #(batch_size, embedding_size)
    
    return outputs


def din_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    last_click_creativeid = tf.string_to_hash_bucket_fast(features["user_click_creatives_att"], 200000)
    creativeid_embeddings = tf.get_variable(name="attention_creativeid_embeddings", dtype=tf.float32,
                                            shape=[200000, 20])
    last_click_creativeid_emb = tf.nn.embedding_lookup(creativeid_embeddings, last_click_creativeid)
    att_creativeid = tf.string_to_hash_bucket_fast(features["creative_id_att"], 200000)
    creativeid_emb = tf.nn.embedding_lookup(creativeid_embeddings, att_creativeid)

    creative_click_attention = attention_layer(creativeid_emb, last_click_creativeid_emb,
                                                  features["user_click_creatives_att"])

    last_click_productid = tf.string_to_hash_bucket_fast(features["user_click_products_att"], 40000)
    productid_embeddings = tf.get_variable(name="attention_productid_embeddings", dtype=tf.float32,
                                           shape=[40000, 20])
    last_click_productid_emb = tf.nn.embedding_lookup(productid_embeddings, last_click_productid)
    att_productid = tf.string_to_hash_bucket_fast(features["product_id_att"], 40000)
    productid_emb = tf.nn.embedding_lookup(productid_embeddings, att_productid)
    product_click_attention = attention_layer(productid_emb,
                                                 last_click_productid_emb, features["user_click_products_att"])

    last_deep_layer = build_deep_layers(net, params)
    last_cross_layer = build_cross_layers(net, params)

    last_layer = tf.concat([last_deep_layer, last_cross_layer, creative_click_attention, product_click_attention], 1)


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

