#KNN Demo
from sklearn.neighbors import KNeighborsClassifier
X = [[0],[1],[2],[3]]
y = [ 0,  0,  1,  1]

#default K=5, let's change it to 3
neigh_clf = KNeighborsClassifier(n_neighbors=3)
neigh_clf.fit(X, y)
neigh_clf.predict([[1]])
neigh_clf.predict([[1.1], [2.5], [-10]])

#Decision Tree demo
from sklearn import tree
X = [[0,0], [1,1]]
y = [0, 1]

tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X, y)
tree_clf.predict([2, 2])

#Logistic Regression demo
from sklearn.linear_model import LogisticRegression
logit_clf = LogisticRegression()
y_pred_logit = logit_clf.fit(iris.data, iris.target).predict(iris.data)
incorrect = (iris.target != y_pred).sum()
print "number mis-classified:", incorrect

#cross validation demo
from sklearn import cross_validation
from sklearn import datasets
iris = datasets.load_iris()
print len(iris['data'])
X_train, X_test, y_train, y_test = 
              cross_validation.train_test_split(iris.data, iris.target, train_size=0.8)
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_clf.score(X_test, y_test)

import numpy as np
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
Y = np.array([0, 1, 0, 1])

kf = cross_validation.KFold(len(Y), n_folds=2)
for train,test in kf: print("%s %s" % (train, test))
X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
X = [[0., 0.],
     [1., 1.],
     [-1., -1.],
     [2., 2.],
     [3., 3.],
     [4., 4.],
     [0., 1.]]
Y = [0, 0, 0, 1, 1, 1, 0]
skf = cross_validation.StratifiedKFold(Y, 2)
for train, test in skf: print("%s %s" % (train, test))