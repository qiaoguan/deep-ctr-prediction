#-*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


FixedLenFeatureColumns=["label", "user_id", "creative_id", "has_target", "terminal",
        "hour", "weekday","template_category",
        "day_user_show", "day_user_click", "city_code","network_type"]
StringVarLenFeatureColumns = ["keyword"]  #特征长度不固定
FloatFixedLenFeatureColumns = ['creative_history_ctr']
StringFixedLenFeatureColumns = ["keyword_attention"]
StringFeatureColumns = ["device_type", "device_model", "manufacturer"]

DayShowSegs = [1, 5, 8, 12, 18, 26, 54, 120, 250, 432, 823]
DayClickSegs = [1, 2, 3, 6, 23]


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    # hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    creative_id = tf.feature_column.categorical_column_with_hash_bucket(
        'creative_id', hash_bucket_size=200000, dtype=tf.int64)
    # To show an example of hashing:
    has_target = tf.feature_column.categorical_column_with_identity(
        'has_target', num_buckets=3)
    terminal = tf.feature_column.categorical_column_with_identity(
        'terminal', num_buckets=10)
    hour = tf.feature_column.categorical_column_with_identity(
        'hour', num_buckets=25)
    weekday = tf.feature_column.categorical_column_with_identity(
        'weekday', num_buckets=10)
    day_user_show = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('day_user_show', dtype=tf.int32), boundaries=DayShowSegs)
    day_user_click = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column('day_user_click', dtype=tf.int32), boundaries=DayClickSegs)

    city_code = tf.feature_column.categorical_column_with_hash_bucket(
        'city_code', hash_bucket_size=2000, dtype=tf.int64)

    network_type = tf.feature_column.categorical_column_with_identity(
        'network_type', num_buckets=20, default_value=19)

    device_type = tf.feature_column.categorical_column_with_hash_bucket(   #androidphone这些
        'device_type', hash_bucket_size=500000, dtype=tf.string
    )
    device_model = tf.feature_column.categorical_column_with_hash_bucket(  #型号如iPhone10  vivo X9
        'device_model', hash_bucket_size=200000, dtype=tf.string
    )
    manufacturer = tf.feature_column.categorical_column_with_hash_bucket(  #手机品牌 vivo iphone等
        'manufacturer', hash_bucket_size=50000, dtype=tf.string
    )


    deep_columns = [
        tf.feature_column.embedding_column(creative_id, dimension=15,combiner='sum'),
        tf.feature_column.embedding_column(has_target, dimension=15,combiner='sum'),
        tf.feature_column.embedding_column(terminal, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(hour, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(weekday, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(day_user_show, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(day_user_click, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(city_code, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(network_type, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(device_type, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(device_model, dimension=15, combiner='sum'),
        tf.feature_column.embedding_column(manufacturer, dimension=15, combiner='sum'),

    ]
    # base_columns = [user_id, ad_id, creative_id,  product_id, brush_num, terminal,terminal_brand]
    '''
    crossed_columns = [tf.feature_column.crossed_column(
                            ['userId', 'adId'], hash_bucket_size = 50000000),
                      、、、
                      ]
    '''
    return deep_columns

def feature_input_fn(data_file, num_epochs, shuffle, batch_size, labels=True):
  """Generate an input function for the Estimator."""

  def parse_tfrecord(value):
    tf.logging.info('Parsing {}'.format(data_file[:10]))
    FixedLenFeatures = {
        key: tf.FixedLenFeature(shape=[1], dtype=tf.int64) for key in FixedLenFeatureColumns
    }

    StringVarLenFeatures = {
        key: tf.VarLenFeature(dtype=tf.string) for key in StringVarLenFeatureColumns
    }
    FloatFixedLenFeatures = {
        key: tf.FixedLenFeature(shape=[1], dtype=tf.float32) for key in FloatFixedLenFeatureColumns
    }
    StringFixedLenFeatures = {
        key: tf.FixedLenFeature(shape=[20], dtype=tf.string) for key in StringFixedLenFeatureColumns
    }
    StringFeatures = {
        key: tf.FixedLenFeature(shape=[1], dtype=tf.string) for key in StringFeatureColumns
    }
    features={}
    features.update(FixedLenFeatures)
    features.update(StringVarLenFeatures)
    features.update(FloatFixedLenFeatures)
    features.update(StringFixedLenFeatures)
    features.update(StringFeatures)
    
    fea = tf.parse_example(value, features)
    feature = {
        key: fea[key] for key in features
    }
    classes = tf.to_float(feature['label'])
    return feature, classes

  # Extract lines from input files using the Dataset API.
  filenames = tf.data.Dataset.list_files(data_file)
  dataset = filenames.apply(tf.contrib.data.parallel_interleave(
      lambda filename: tf.data.TFRecordDataset(filename),
      cycle_length=32))

  if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size*64)

  dataset = dataset.repeat(num_epochs).batch(batch_size).prefetch(buffer_size=batch_size*8)
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=32)

  return dataset

