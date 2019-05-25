import os
import gist
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cnt = -1
features = []
labels = []

path = 'C:\\Users\\Admin\\Desktop\\train-20190508T034909Z-001\\train'
for c in os.listdir(path):
    cnt += 1
    for file_name in os.listdir(os.path.join(path,c)):
        file_path = os.path.join(path,c,file_name)
        img = cv2.imread(file_path)
        labels.append(cnt)
        features.append(gist.extract(img))

scaled_data = StandardScaler().fit_transform(features)
pca = PCA()
pca.fit(scaled_data)
print(pca.explained_variance_ratio_)

pca1 = PCA(n_components=43)
data1 = pca1.fit_transform(scaled_data)

X_pan = pd.DataFrame(data1)
X_pan1 = pd.DataFrame(features)
X_pan["labels"] = labels
X_pan1["labels"] = labels
"""
project01_02.csv: dataset without PCA
project01_03.csv: dataset with PCA
"""
X_pan.to_csv("project01_03.csv")
X_pan1.to_csv("project01_02.csv")
'''
labels:
    0:BoCongNho(199)
    1:PhinhVi(217)
    2:ThanVi(660)
    3:HangVi(418)
    4:TamVi(154)
    5:MonVi(151)
'''