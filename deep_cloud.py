"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep", "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

# used when loading the csv file that acts as the source data for training the model
COLUMNS = [ "uid_type", "uid_principalid", "uid_arn", "uid_accountid", "uid_accesskeyid", "uid_sess_ctx_mfa",
            "uid_sess_ctx_creationdate", "uid_issuer_type", "uid_issuer_principalid", "uid_issuer_arn",
            "uid_issuer_accountid", "uid_issuer_username", "event_time", "event_source", "event_name",
            "aws_region", "source_ip_address", "user_agent", "event_type", "recipient_accountid" ]

# this is the field in the training data where the classifier looks for the target value
LABEL_COLUMN = "label"

# these labels identify features that are represented by a grab-bag of strings like "cook" or "argentina"
CATEGORICAL_COLUMNS = [ "uid_type", "uid_principalid", "uid_arn", "uid_accountid", "uid_accesskeyid", "uid_sess_ctx_mfa",
            "uid_sess_ctx_creationdate", "uid_issuer_type", "uid_issuer_principalid", "uid_issuer_arn",
            "uid_issuer_accountid", "uid_issuer_username", "event_source", "event_name", "aws_region",
            "source_ip_address", "user_agent", "event_type", "recipient_accountid" ]

# these labels identify features that are represented by real numbers like 3 and 1.12
CONTINUOUS_COLUMNS = [ "event_time" ]

def build_estimator(model_dir):
  """Build an estimator."""
  # Sparse base columns.

  # returns a scaffold to train models
  # load in the column information
  # return the object

  # build the column objects to

  # aws region has a known list of possible column values
  # so we pass the known keys, the region names
  aws_region = tf.contrib.layers.sparse_column_with_keys(column_name="aws_region",
                                                     keys=["ap_south_1", "ap_northeast_1", "ap_northeast_2", "ap_southeast_1", "ap_southeast_2", "us_west_1", "us_west_2", "us_east_1", "sa_east_1", "eu_west_1", "eu_central_1" ])

  # these are columns where the values are all over the place
  # field = tf.contrib.layers.sparse_column_with_hash_bucket("field", hash_bucket_size=1000)


  # Continuous base columns.
  event_time = tf.contrib.layers.real_valued_column("event_time")

  # Deep columns
  deep_columns = [
      tf.contrib.layers.embedding_column(aws_region, dimension=8),
      event_time
  ]

  m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])

  return m

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_and_eval():
  """Train and evaluate the model."""

  train_file_name = './persons.data'
  test_file_name = './persons.test'

  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True)
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1)

  # Iterate over the training data and build the column for the actual classifier
  # df_train and df_test will have a new column called 'label' that contains
  # either a '0' or a '1'.
  # Everything that isn't a in the column 'label' is a feature to use in the training

  df_train[LABEL_COLUMN] = (df_train["malicious"].apply(lambda x: "yes" in x)).astype(int)

  # print(df_train[LABEL_COLUMN])

  # Do the same for the separate test data
  df_test[LABEL_COLUMN] = (df_test["malicious"].apply(lambda x: "yes" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  #model_dir = './model/'
  print("model directory = %s" % model_dir)

  model = build_estimator(model_dir)
  model.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)
  results = model.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()

if __name__ == "__main__":
  tf.app.run()
