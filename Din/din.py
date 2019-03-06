# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import dice
import collections


def build_deep_layers(net, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.

    for layer_id, num_hidden_units in enumerate(params['hidden_units']):
        # net = tf.layers.dense(net, units=num_hidden_units, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
        net = tf.layers.dense(net, units=num_hidden_units, activation=None,
                              kernel_initializer=tf.glorot_uniform_initializer())
        net = dice(net, name='dice_' + str(layer_id))
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

def attention_layer(querys, keys):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, N, embedding_size]  N is the number of keys(e.g. number of keyword for each sample)
    """

    keys_length = tf.shape(keys)[1] # padded_dim
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
    querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])
    #print(querys)
    #print(keys)
    net = tf.concat([keys, keys - querys, querys, keys*querys], axis=-1)
    for units in [32,16]:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        # shape(batch_size, N, 1)
    outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  #shape(batch_size, 1, N)
    outputs = outputs / (embedding_size ** 0.5)
    outputs = tf.nn.softmax(outputs)
    outputs = tf.matmul(outputs, keys)  # (batch_size, 1, embedding_size)
    outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")   #(batch_size, embedding_size)
    return outputs

def varlen_attention_layer(seq_ids, tid, id_type):
    with tf.variable_scope("attention_" + id_type):
      """
      embedding_size = self._params["embedding_size"][id_type]
      embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                   shape=[self._params["vocab_size"][id_type], embedding_size])
      """
      embedding_size = 15
      embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                   shape=[500000, embedding_size])
      seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids.indices)  # shape(batch_size, max_seq_len, embedding_size)
      #seq_emb = tf.nn.embedding_lookup_sparse(embeddings, seq_ids, sp_weights=None)
      #tid_emb = tf.nn.embedding_lookup(embeddings, tid)  # shape(batch_size, embedding_size)

      embeddings1 = tf.get_variable(name="embeddings1", dtype=tf.float32,
                                   shape=[200000, embedding_size])
      tid_emb = tf.nn.embedding_lookup(embeddings1, tid)      # shape(batch_size, embedding_size)

      max_seq_len = tf.shape(seq_ids)[1] # padded_dim


      print('=============================')
      print(seq_ids)
      print(tid)
      print(seq_emb)
      print(tid_emb)
      print(max_seq_len)
      print("=============================")
      
      u_emb = tf.reshape(seq_emb, shape=[-1, max_seq_len, embedding_size])
      a_emb = tf.reshape(tf.tile(tid_emb, [1, 1, max_seq_len]), shape=[-1, max_seq_len, embedding_size])
      net = tf.concat([u_emb, u_emb - a_emb, a_emb], axis=1)
      for units in [32,16]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
      att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        # shape(batch_size, max_seq_len, 1)
      att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")
      wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
      #masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
      masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)   # shape(batch_size, max_seq_len, 1)
      att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")#shape(batch_size,embedding_size)
      return att_emb, tf.reshape(tid_emb, shape=[-1, embedding_size])


def din_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    attention_keyword = tf.string_to_hash_bucket_fast(features["keyword_attention"], 500000)
    attention_keyword_embeddings = tf.get_variable(name="attention_keyword_embeddings", dtype=tf.float32,
                                 shape=[500000, 20])
    # shape(batch_size, len, embedding_size)
    attention_keyword_emb =  tf.nn.embedding_lookup(attention_keyword_embeddings, attention_keyword)


    attention_creativeid = tf.string_to_hash_bucket_fast(tf.as_string(features["creative_id"]), 200000)
    attention_creativeid_embeddings = tf.get_variable(name="attention_creativeid_embeddings", dtype=tf.float32,
                                 shape=[200000, 20])
    # shape(batch_size, 1, embedding_size)
    attention_creativeid_emb = tf.nn.embedding_lookup(attention_creativeid_embeddings, attention_creativeid)

    keyword_creativeid_attention = attention_layer(attention_creativeid_emb,
                                                   attention_keyword_emb)  # (batchsize,embedding_size)


    last_deep_layer = build_deep_layers(net, params)
    last_cross_layer = build_cross_layers(net, params)

    last_layer = tf.concat([last_deep_layer, last_cross_layer, keyword_creativeid_attention], 1)


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

