import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

'''
labels:
    0:BoCongNho(199)
    1:PhinhVi(217)
    2:ThanVi(660)
    3:HangVi(418)
    4:TamVi(154)
    5:MonVi(151)
'''
dataset = pd.read_csv('project01_03.csv')
point = ['0']
for i in range(43):
    point.append(str(i))
X = dataset[point].values
y = dataset['labels'].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# KNN
print("GIST+KNN")
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(1,30)}
knn_gscv = GridSearchCV(knn,param_grid,cv = 20)
knn_gscv.fit(X_train,y_train)
print(knn_gscv.best_params_['n_neighbors'])
knn1 = KNeighborsClassifier(n_neighbors = knn_gscv.best_params_['n_neighbors'])
knn1.fit(X_train,y_train)
y_pred = knn1.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#SVM
print("GIST+SVM")
clf = LinearSVC()
clf.fit(X_train,y_train)
y_pred1 = clf.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))