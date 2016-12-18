from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess

iris_x, iris_y = datasets.load_iris(return_X_y=True)
feature_names = list(datasets.load_iris().feature_names)

model = DecisionTreeClassifier()
model.fit(iris_x, iris_y)

def tree_visualize(model, feature):
    with open("tree.dot", "w") as f:
        export_graphviz(model, out_file=f, feature_names=feature)
    
    output_command = ["dot", "-Tpng", "tree.dot", "-o", "tree.png"]
    try:
        subprocess.check_call(output_command)
    except:
        print "fail to produce visualization"

tree_visualize(model, feature_names)
