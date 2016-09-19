"""Example of DNNClassifier for Iris plant dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import cross_validation
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn

import json
from pprint import pprint
import numpy as np



def main(unused_argv):
    # Load dataset.
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

    with open('values.json') as data_file:
        data = json.load(data_file)

    print(len(data[0]))

    w = 42034
    h = 257

    vals = np.zeros((w,h), dtype=np.int32)

    y_vals = np.zeros(w, dtype=np.int32)
    # fill the y_vals with random data changed very
    for x in range(w):
        v = 0
        # if np.random.rand(1)[0] >= 0.5:
        #     v = 1
        if x < 1000:
            v = 1
        y_vals.itemset(x,v)

    print(y_vals[990:1010])

    print(vals.shape)
    print(y_vals.shape)
    print(x_train.shape)
    print(y_train.shape)

    for i, v in enumerate(data):
        vals.itemset((v[0], v[1]), v[2])

    #print(range(10))
    # n = 10000
    # for x in range(n,n+10):
    #     print(vals[n])

    x_train = vals[:40000]
    y_train = y_vals[:40000]
    x_test = vals[40000:]
    y_test = y_vals[40000:]

    # print(vals.shape)
    # print(vals[1:3])

    # print(y_train)
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    # print(x_train.shape)
    # feature_columns = learn.infer_real_valued_columns_from_input(x_train)

    feature_columns = learn.infer_real_valued_columns_from_input(vals)

    classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=2)

    # Fit and predict.
    classifier.fit(x_train, y_train, steps=200)
    predictions = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, predictions)
    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()
