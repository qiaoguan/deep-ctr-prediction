# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from utils import feedforward, multihead_attention, layer_normalization
import collections

'''
A tensorflow implementation of Transformer

Ashish Vasmani et all.  "Attention is All You Need,"  In NIPS,2017.

'''

class TransformerNetwork(object):
    def __init__(self, num_units, num_blocks, num_heads, max_len, dropout_rate, pos_fixed=True, l2_reg=0.0):
        self.num_units = num_units        # embedding_size
        self.num_blocks = num_blocks      # the number of multi-head attention we use
        self.num_heads = num_heads
        self.max_len = max_len              # the max length of the sequence
        self.dropout_keep_prob = 1. - dropout_rate
        self.position_encoding_matrix = None
        self.pos_fixed = pos_fixed
        self.l2_reg = l2_reg

    def  get_position_encoding(self, inputs, scope="pos_embedding/", reuse=None, dtype=tf.float32):
        '''
        Args:
            inputs: sequence embeddings, shape: (batch_size , max_len, embedding_size)
        Return:
            Output sequences which has the same shape with inputs
        '''
        #E = inputs.get_shape().as_list()[-1]  # static
        #N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
        with tf.variable_scope(scope, reuse=reuse):
            if self.position_encoding_matrix is None:
                encoded_vec = np.array(
                    [pos / np.power(10000, 2 * i / self.num_units) for pos in range(self.max_len) for i in
                     range(self.num_units)])
                encoded_vec[::2] = np.sin(encoded_vec[::2])     #从index为0开始，下一个元素index+2
                encoded_vec[1::2] = np.cos(encoded_vec[1::2])
                encoded_vec = tf.convert_to_tensor(encoded_vec.reshape([self.max_len, self.num_units]), dtype=dtype)
                self.position_encoding_matrix = encoded_vec  # (max_len, num_units)

            N = tf.shape(inputs)[0]     # batch_size
            T = tf.shape(inputs)[1]     # max_len
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (batch_size , max_len)
            position_encoding = tf.nn.embedding_lookup(self.position_encoding_matrix,
                                                       position_ind)  # (batch_size, len, num_units)
        return position_encoding


    def __call__(self, inputs, mask):
        '''
        Args:
            inputs: sequence embeddings (item_embeddings +  pos_embeddings) shape: (batch_size , max_len, embedding_size)
            mask:  deal with mask shape: (batch_size, max_len, 1)
        Return:
            Output sequences which has the same shape with inputs
        '''
        if self.pos_fixed:  # use sin /cos positional embedding
            position_encoding = self.get_position_encoding(inputs)  # (batch_size, len, num_units)

        inputs += position_encoding
        inputs *= mask
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                inputs = multihead_attention(queries=layer_normalization(inputs),
                                             keys=inputs,
                                             num_units=self.num_units,
                                             num_heads=self.num_heads,
                                             dropout_keep_prob=self.dropout_keep_prob,
                                             causality=False,
                                             scope="self_attention")

                # Feed forward
                inputs = feedforward(layer_normalization(inputs), num_units=[self.num_units, self.num_units],
                                     dropout_keep_prob=self.dropout_keep_prob)

                inputs *= mask
        outputs = layer_normalization(inputs)  # (batch_size, max_len, num_units)
        return outputs

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


def transformer_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    last_click_creativeid = tf.string_to_hash_bucket_fast(features["user_click_creatives_att"], 200000)
    creativeid_embeddings = tf.get_variable(name="attention_creativeid_embeddings", dtype=tf.float32,
                                            shape=[200000, 20])
    last_click_creativeid_emb = tf.nn.embedding_lookup(creativeid_embeddings, last_click_creativeid)

    last_click_productid = tf.string_to_hash_bucket_fast(features["user_click_products_att"], 40000)
    productid_embeddings = tf.get_variable(name="attention_productid_embeddings", dtype=tf.float32,
                                        shape=[40000, 20])
    last_click_productid_emb = tf.nn.embedding_lookup(productid_embeddings, last_click_productid)

    his_click_emb = tf.concat([last_click_creativeid_emb, last_click_productid_emb], 2)  # (batch_size,10,emb_size*2)

    transformerNetwork_click = TransformerNetwork(params['transformer_num_units'], params['num_blocks'],
                                                  params['num_heads'],
                                                  max_len=10, dropout_rate=params['dropout_rate'], pos_fixed=True)
    mask_click = tf.expand_dims(tf.to_float(tf.cast(tf.not_equal(features["user_click_creatives_att"], "0"),
                                                    tf.float32)), -1)  # (batch_size, 10, 1)

    transformer_click_outputs = transformerNetwork_click(his_click_emb, mask_click)  # (batch_size, max_len, num_units)
    transformer_click_outputs = tf.reshape(tf.reduce_sum(transformer_click_outputs, 1),
                                           shape=[-1, params['transformer_num_units']])


    last_deep_layer = build_deep_layers(net, params)
    last_cross_layer = build_cross_layers(net, params)

    last_layer = tf.concat([last_deep_layer, last_cross_layer, transformer_click_outputs], 1)


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

