import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df_train=pd.read_csv("tamALL.csv")

#the number of the stress data 
# df_train.groupby('stress').size()

var_col=[c for c in df_train.columns if c not in ['time','stress']]
X=df_train.loc[:,var_col]
y=df_train.loc[:,'stress']

X_train, X_valid, y_train, y_valid=train_test_split(X, y, test_size=0.2, random_state=42)

""""
#Create the figure
#plt.figure(figsize=(20,10))

#Create the tree plot
plot_tree(model_tree,
           feature_names = var_col, #Feature names
           class_names = ["0","1"], #Class names
           rounded = True,
           filled = True)

plt.show()
"""

def tree_training(max_leaf_nodes, X_train, y_train, X_valid, y_valid):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, class_weight='balanced')
    model_tree.fit(X_train, y_train)
    
    y_train_pred = model_tree.predict(X_train)
    y_valid_pred = model_tree.predict(X_valid)
    
    auc_train = metrics.roc_auc_score(y_train, y_train_pred)
    auc_valid = metrics.roc_auc_score(y_valid, y_valid_pred)
    
    print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(max_leaf_nodes,auc_train,auc_valid,auc_train-auc_valid))
    print(f'Accuracy- : {model_tree.score(X,y):.3f}')

# Run few iterations to find which max_tree_nodes works best
for i in range(2, 20):
    tree_training(i, X_train, y_train, X_valid, y_valid)
    #this gives back result


# CV function requires a scorer of this form
def cv_roc_auc_scorer(model, X, y): 
    return metrics.roc_auc_score(y, model.predict(X))
"""
# Loop through multiple values of max_leaf_nodes to find best parameter
for num_leaf_node in range(2,16):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=num_leaf_node, class_weight='balanced')
    kfold_scores = cross_validate(model_tree,
                                  X,
                                  y,
                                  cv=5,
                                  scoring=cv_roc_auc_scorer,
                                  return_train_score=True)

    # Find average train and test score
    train_auc_avg = np.mean(kfold_scores['train_score'])
    test_auc_avg = np.mean(kfold_scores['test_score'])

    print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(num_leaf_node,
                                                                     train_auc_avg,
                                                                     test_auc_avg,
                                                                     train_auc_avg-test_auc_avg))
"""