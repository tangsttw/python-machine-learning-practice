from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
import time

# load dataset and feature name
iris_x, iris_y = datasets.load_iris(return_X_y=True)
feature_names = list(datasets.load_iris().feature_names)

# create model and fit
model = DecisionTreeClassifier()

train_start = time.time()
model.fit(iris_x, iris_y)
train_duration = time.time() - train_start

print "the training takes %f seconds." % train_duration

# the visualization function
def tree_visualize(model, feature):
    with open("tree.dot", "w") as f:
        export_graphviz(model, out_file=f, feature_names=feature)
    
    output_command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
    try:
    	# to produce the png file
        subprocess.check_call(output_command)
    except:
        print "fail to produce visualization result"

tree_visualize(model, feature_names)
