import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
feature_columns = skflow.infer_real_valued_columns_from_input(iris.data)

# only 150 entries in dataset, so using linear Classifier
#classifier = skflow.LinearClassifier(n_classes=3,feature_columns=feature_columns)
# for more data, we could use a Deep Neural Net Classifier
classifier = skflow.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)

# Training
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)

score = metrics.accuracy_score(iris.target, list(classifier.predict(iris.data, as_iterable=True)))

print("Accuracy: %f" % score)

