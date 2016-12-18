from sklearn import datasets
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

# set numpy random seed
np.random.seed(28)

# load data of iris dataset and separate it into feature(x) and label(y)
data_source = datasets.load_iris()
data_x = data_source.data
data_y = data_source.target

# randomly separate data into train and test
random_separator = np.random.permutation(len(data_x))
train_x = data_x[random_separator[:-20]]
train_y = data_y[random_separator[:-20]]
test_x = data_x[random_separator[-20:]]
test_ground_truth = data_y[random_separator[-20:]]

# create and train model
# model parameter : 
# 	3 nearest neighbors
# 	weight function based on distance
# 	auto algorithm to find the nearest neighbors
# see more : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')
model.fit(train_x, train_y)

# test the model and compare the result with the ground truth
print model.predict(test_x)
print test_ground_truth