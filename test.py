


from sklearn import tree
from sklearn.datasets import load_iris

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 1, 1, 1]
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)

from sklearn.externals.six import StringIO

with open("decisionTree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f, feature_names = ['a', 'b'])

print clf.tree_.value
print clf.tree_.children_left
print clf.tree_.children_right
print clf.tree_.feature
print clf.tree_.threshold
print clf.tree_.value[4][0][1]

# from inspect import getmembers
# print(getmembers(clf.tree_))

# with open("iris.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
