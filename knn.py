import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics

df_train=pd.read_csv("trashALL.csv")

#the number of the stress data 
# df_train.groupby('stress').size()

var_col=[c for c in df_train.columns if c not in ['time','stress']]
X=df_train.loc[:,var_col]
y=df_train.loc[:,'stress']

X_train, X_valid, y_train, y_valid=train_test_split(X, y, test_size=0.2, random_state=42)

#scalar=StandardScaler()
#scalar.fit(X_train)

clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

def acc():
    y_train_pred = clf.predict(X_train)
    y_valid_pred = clf.predict(X_valid)

    auc_train = metrics.roc_auc_score(y_train, y_train_pred)
    auc_valid = metrics.roc_auc_score(y_valid, y_valid_pred)

    print("Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(auc_train,auc_valid,auc_train-auc_valid))

    print(f'Accuracy- : {clf.score(X,y):.3f}')

    print(classification_report(y_valid, y_valid_pred))
    print(confusion_matrix(y_valid, y_valid_pred))

acc()

