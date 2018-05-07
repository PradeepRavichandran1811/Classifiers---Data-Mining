import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt

#Reading HandWrittenLetters.csv
d = pd.read_csv("HandWrittenLetters.csv",header=None)
a=np.asarray(d.ix[0,:])
B=np.asarray(d.ix[1:,:])
print (B.shape)
B = B.T

#Split Dataset into Training data and Testing Data
X,tX,y,ty=train_test_split(B,a,test_size=0.10,random_state=42)
print ("train")
print (X.shape)
c=X.T
print (c.shape)
d=c.T
print (d.shape)
print (ty.shape)
print (tX.shape)

#Array Creation
nb_predict=[]
nb1_predict=[]
nb2_predict=[]
centro_predict=[]
lin_predict=[]
svm_predict=[]

#Fitting Data
#K Nearest neighbors (K=1)
nb = neighbors.KNeighborsClassifier(1, weights='uniform')
nb.fit(X, y)
#K Nearest neighbors (K=3)
nb1 = neighbors.KNeighborsClassifier(3, weights='uniform')
nb1.fit(X, y)
#K Nearest neighbors (K=5)
nb2 = neighbors.KNeighborsClassifier(5, weights='uniform')
nb2.fit(X, y)
#Centroid
centro = NearestCentroid()
centro.fit(X, y)
#Support Vector Machine
s = svm.SVC(kernel='linear',C=1)
s.fit(X, y)
for i in range (0,102):
    t = np.asarray(tX[i,0:])
    t= np.reshape(t,(1,-1))

    print (i)
    #prediction
    print ("K Nearest Neighbors for k=1")
    nb_pred = nb.predict(t)
    nb_predict.insert(i, nb_pred[0])
    print (nb_pred)
    print ("K Nearest Neighbors for k=3")
    nb1_pred = nb1.predict(t)
    nb1_predict.insert(i, nb1_pred[0])
    print (nb1_pred)
    print ("K Nearest Neighbors for k=5")
    nb2_pred = nb2.predict(t)
    nb2_predict.insert(i, nb2_pred[0])
    print (nb2_pred)

    print ("Centroid ")
    centro_pred = centro.predict(t)
    centro_predict.insert(i, centro_pred[0])
    print (centro_pred)

    print ("Support Vector Machine")
    svm_pred = s.predict(t)
    svm_predict.insert(i, svm_pred[0])
    print (svm_pred)

#Accuracy for Classifiers
print (nb_pred)
print ("Accuracy for KNN (k=1)")
print (accuracy_score(ty,nb_predict))
print (nb1_pred)
print ("Accuracy for KNN (k=3)")
print (accuracy_score(ty,nb1_predict))
print (nb2_pred)
print ("Accuracy for KNN (k=5)")
print (accuracy_score(ty,nb2_predict))
print (centro_pred)
print ("Accuracy for Centroid")
print (accuracy_score(ty,centro_predict))
print (svm_pred)
print ("Accuracy for SVM")
print (accuracy_score(ty,svm_predict))

# 2-fold Cross Validation for Classifiers
sc=cross_val_score(nb,X,y,cv=2)
print ("Cross validation for KNN (k=1)")
print (sc)
print ("Accuracy under 2-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb1,X,y,cv=2)
print ("Cross validation for KNN (k=3)")
print (sc)
print ("Accuracy under 2-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb2,X,y,cv=2)
print ("Cross validation for KNN (k=5)")
print (sc)
print ("Accuracy under 2-fold cross validation")
print (sc.mean())

sc=cross_val_score(centro,X,y,cv=2)
print ("Cross Validation for Centroid")
print (sc)
print ("Accuracy under 2-fold cross validation")
print (sc.mean())

sc=cross_val_score(s,X,y,cv=2)
print ("Cross Validation for Support Vector Machine")
print (sc)
print ("Accuracy under 2-fold cross validation")
print (sc.mean())

# 3-fold Cross Validation for Classifiers
sc=cross_val_score(nb,X,y,cv=3)
print ("Cross validation for KNN (k=1)")
print (sc)
print ("Accuracy under 3-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb1,X,y,cv=3)
print ("Cross validation for KNN (k=3)")
print (sc)
print ("Accuracy under 3-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb2,X,y,cv=3)
print ("Cross validation for KNN (k=5)")
print (sc)
print ("Accuracy under 3-fold cross validation")
print (sc.mean())

sc=cross_val_score(centro,X,y,cv=3)
print ("Cross Validation for Centroid")
print (sc)
print ("Accuracy under 3-fold cross validation")
print (sc.mean())

sc=cross_val_score(s,X,y,cv=3)
print ("Cross Validation for Support Vector Machine")
print (sc)
print ("Accuracy under 3-fold cross validation")
print (sc.mean())

# 5-fold Cross Validation for Classifiers
sc=cross_val_score(nb,X,y,cv=5)
print ("Cross validation for KNN (k=1)")
print (sc)
print ("Accuracy under 5-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb1,X,y,cv=5)
print ("Cross validation for KNN (k=3)")
print (sc)
print ("Accuracy under 5-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb2,X,y,cv=5)
print ("Cross validation for KNN (k=5)")
print (sc)
print ("Accuracy under 5-fold cross validation")
print (sc.mean())

sc=cross_val_score(centro,X,y,cv=5)
print ("Cross Validation for Centroid")
print (sc)
print ("Accuracy under 5-fold cross validation")
print (sc.mean())

sc=cross_val_score(s,X,y,cv=5)
print ("Cross Validation for Support Vector Machine")
print (sc)
print ("Accuracy under 5-fold cross validation")
print (sc.mean())


# 10-fold Cross Validation for Classifiers
sc=cross_val_score(nb,X,y,cv=10)
print ("Cross validation for KNN (k=1)")
print (sc)
print ("Accuracy under 10-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb1,X,y,cv=10)
print ("Cross validation for KNN (k=3)")
print (sc)
print ("Accuracy under 10-fold cross validation")
print (sc.mean())

sc=cross_val_score(nb2,X,y,cv=10)
print ("Cross validation for KNN (k=5)")
print (sc)
print ("Accuracy under 10-fold cross validation")
print (sc.mean())

sc=cross_val_score(centro,X,y,cv=10)
print ("Cross Validation for Centroid")
print (sc)
print ("Accuracy under 10-fold cross validation")
print (sc.mean())

sc=cross_val_score(s,X,y,cv=10)
print ("Cross Validation for Support Vector Machine")
print (sc)
print ("Accuracy under 10-fold cross validation")
print (sc.mean())

plt.plot([1,2,3,4],[0.71,0.73,0.75,0.75],'bo-')
plt.plot([5,6,7,8],[0.70,0.70,0.72,0.71],'bo-')
plt.plot([9,10,11,12],[0.76,0.78,0.80,0.81],'bo-')
plt.axis([0,14,0,1])
plt.ylabel('Accuracy')
plt.xlabel('K Nearest Neighbors,Centroid,SVM (2fold,3fold,5fold,10fold)')
plt.show()