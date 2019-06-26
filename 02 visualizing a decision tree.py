# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/6/26 2:37 PM
@file: 02 visualizing a decision tree.py
@desc:
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
test_idx = [0, 50, 100]

train_target = np.delete(iris.target, test_idx, axis=None)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")


print(test_data[1],test_target[1])
print(iris.feature_names,iris.target_names)