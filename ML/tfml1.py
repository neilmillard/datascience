# import TensorFlow and numpy
import tensorflow as tf
import numpy as np

#next load training and test sets
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# This is equivalent to above
# classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
# classifier.fit(x=training_set.data, y=training_set.target, steps=1000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))