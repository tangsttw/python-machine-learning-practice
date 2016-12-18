from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# load iris dataset from sklearn datasets
iris = dataset.load_iris()

# create Gaussian Naive Bayes model 
model = GaussianNB

# train model
model.fit(iris.data, iris.target)
