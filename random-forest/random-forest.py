from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from difflib import SequenceMatcher
import subprocess

# load dataset
dataset_x, dataset_y = datasets.load_iris(return_X_y=True)
feature_names = datasets.load_iris().feature_names

# another random sampling way compares to the one in knn example
random_selector = np.random.randint(0, 2, len(dataset_x))
train_x, test_x, train_y, test_y = dataset_x[random_selector==1], dataset_x[random_selector==0], dataset_y[random_selector==1], dataset_y[random_selector==0]

model = RandomForestClassifier(n_estimators=5)

# train the random forest model and keep track of the time
train_start = time.time()
model.fit(train_x, train_y)
train_end = time.time()
result = model.predict(test_x)
test_end = time.time()

# the predicted result test set
print "predicted result : " 
print result
# the ground truth of the test set
print "ground truth : " 
print test_y
# the correctness represents in a ratio form
compare = SequenceMatcher()
compare.set_seqs(result, test_y)
print "the correctness ratio : %f" % compare.ratio()
# the time spent on each part
print "\ntime spent on training : %f" % (train_end - train_start)
print "time spent on predictin : %f" % (test_end - train_end)