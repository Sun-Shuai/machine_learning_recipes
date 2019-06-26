# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/6/26 3:29 PM
@file: 04 write a pipeline.py
@desc:
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# my_classifier = tree.DecisionTreeClassifier()
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
