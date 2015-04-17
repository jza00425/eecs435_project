import pandas as pd
import numpy as np
from sklearn.externals.six import StringIO
from sklearn import tree

# dummy data:
# df = pd.DataFrame({'a':[0,0,1,1],'b':[0,1,0,1], 'c': [0,1,1,0], 'dv':[0,0,0,1]})
# df = pd.read_csv('dump.csv')
df = pd.read_csv('result.csv')
print df

# create decision tree
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(df.ix[:,:11], df.dv)
with open("decisionTree.dot", 'w') as f:
    f = tree.export_graphviz(dt, out_file = f, feature_names = ['G5', 'G6', 'G7', 'G0_0', 'G1_0', 'G2_0', 'G3_0', 'G0_1', 'G1_1', 'G2_1', 'G3_1'])

def get_lineage(dtree, feature_names):
    left = dtree.tree_.children_left
    right = dtree.tree_.children_right
    threshold = dtree.tree_.threshold
    features = [feature_names[i] for i in dtree.tree_.feature]

    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    tmp = ''
    for child in idx:
        for node in recurse(left, right, child):
            if isinstance(node, int): 
                tmpValue = 'out = 0' if dtree.tree_.value[node][0][0] > 0 else 'out = 1'
                tmp = tmp + tmpValue + ';'
                print tmp
                tmp = ''
            else:
                tmp = tmp + node[3] + ' = '
                a =  '0' if node[1] == 'l' else '1'
                tmp += a
                tmp += ' '

            # for line in node:
                # if type(line) == type(int()):
                    # print line
                    # result.append(line)
                # else:
                    # print line
                    # tmp = line[3] + '='
                    # tmp =+ '0' if line[1] == 'l' else '1'
            # print tmp





get_lineage(dt, df.columns)
