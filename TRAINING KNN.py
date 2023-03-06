import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import joblib

data_bayam = pd.read_csv('DATA_BAYAM2.csv')

x = data_bayam.iloc[:,1:3].values
y = data_bayam.iloc[:,0].values

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=35,test_size=0.40)
print(y_test)
print(x_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred)*100)

joblib.dump(knn,'knn.sav')