# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:01:55 2022

@author: James
"""

import tensorflow
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#Ridge regression 
clf = Ridge(alpha=0.01, fit_intercept=False)
clf.fit(x_train, y_train) #fit
x_guess = clf.predict(x_test) #predict
plt.plot(x_guess)
print(mean_squared_error(y_test, x_guess))


objective_value = []
target = x_train

#k clustering
k = 3
kmeanModel = KMeans(k)
km = kmeanModel.fit(target)
objective_value.append(kmeanModel.inertia_)

         
# Plot the clusters
fig, ax = plt.subplots()
sc = plt.scatter(target[:,0], target[:,1], c=kmeanModel.labels_.astype(float))
ax.legend(*sc.legend_elements())
plt.scatter(kmeanModel.cluster_centers_[:, 0], kmeanModel.cluster_centers_[:, 1],s=300, marker='+', c='red', label='Centroids')
plt.show()


#get the clusters
def getCluster(num, labels_array):
  return np.where(labels_array == num)[0]

# Kmeans model + 3 ridge regression models
T1 = RidgeCV()
T2 = RidgeCV()
T3 = RidgeCV()

ridge_models = [T1, T2, T3]

cluster_predict_train = kmeanModel.predict(x_train)
cluster_predict_test = kmeanModel.predict(x_test)

for i in range(3):
    x_train_new = x_train[(cluster_predict_train == i)]
    y_train_new = y_train[(cluster_predict_train == i)]
    ridge_models[i].fit(x_train_new, y_train_new)

y_predict = []
for i in range(len(x_test)):
    index = cluster_predict_test[i]
    predict = ridge_models[index].predict(x_test[i].reshape(1, -1))
    y_predict.append(predict[0])

score = mean_squared_error(y_test, y_predict)
print("Kmeans + Three ridge regression models' MSE is: \n", score)




#PCA
plt.figure()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)
# print(principalComponents)
coeff = (np.transpose(pca.components_))
plt.scatter(principalComponents[:,0], principalComponents[:,1])
plt.plot( [0,coeff[0,0]], [0,coeff[1,0]],'r')
plt.plot( [0,coeff[0,1]], [0,coeff[1,1]],'r')


# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(x_train)
# # print(principalComponents)
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(principalComponents[:,0], principalComponents[:,1],principalComponents[:,2])



#part 2
from sklearn.datasets import load_wine

x, y  = load_wine(return_X_y=(True))


length = 178//2


x_train = x[0 :length, :]
y_train = y[0 :length]
x_test = x[-length : , :]
y_test = y[-length :]

#ridge regression
clf = Ridge(alpha=0.01, fit_intercept=False)
clf.fit(x_train, y_train)
x_guess = clf.predict(x_test)
plt.plot(x_guess)
print(mean_squared_error(y_test, x_guess))


objective_value = []
target = x_train

#k clusters and find the best k value
for k in range(1,10):
    # Building and fitting the model
    kmeanModel = KMeans(k)
    kmeanModel.fit(x_train)
    objective_value.append(kmeanModel.inertia_)
    
    # Plot the clusters
    fig, ax = plt.subplots()
    sc = plt.scatter(x_train[:,0], x_train[:,1], c=kmeanModel.labels_.astype(float))
    ax.legend(*sc.legend_elements())
    plt.scatter(kmeanModel.cluster_centers_[:, 0], kmeanModel.cluster_centers_[:, 1],s=300, marker='+', c='red', label='Centroids')
    plt.show()
    
plt.plot(range(1,10),objective_value, 'bx-')
plt.xlabel('Values of k')
plt.ylabel('Objective value')
plt.show()

# use PCA to double check
plt.figure()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)
# print(principalComponents)
coeff = (np.transpose(pca.components_))
plt.scatter(principalComponents[:,0], principalComponents[:,1])
plt.plot( [0,coeff[0,0]], [0,coeff[1,0]],'r')
plt.plot( [0,coeff[0,1]], [0,coeff[1,1]],'r')


#k value = 2
k = 2
kmeanModel = KMeans(k)
km = kmeanModel.fit(target)
objective_value.append(kmeanModel.inertia_)

         
# Plot the clusters
fig, ax = plt.subplots()
sc = plt.scatter(target[:,0], target[:,1], c=kmeanModel.labels_.astype(float))
ax.legend(*sc.legend_elements())
plt.scatter(kmeanModel.cluster_centers_[:, 0], kmeanModel.cluster_centers_[:, 1],s=300, marker='+', c='red', label='Centroids')
plt.show()


def getCluster(num, labels_array):
  return np.where(labels_array == num)[0]

# Kmeans model + 2 ridge regression models
T1 = RidgeCV()
T2 = RidgeCV()

ridge_models = [T1, T2]

cluster_predict_train = kmeanModel.predict(x_train)
cluster_predict_test = kmeanModel.predict(x_train)

for i in range(2):
    x_train_new = x_train[(cluster_predict_train == i)]
    y_train_new = y_train[(cluster_predict_train == i)]
    ridge_models[i].fit(x_train_new, y_train_new)

y_predict = []
for i in range(len(x_train)):
    index = cluster_predict_test[i]
    predict = ridge_models[index].predict(x_test[i].reshape(1, -1))
    y_predict.append(predict[0])

score = mean_squared_error(y_test, y_predict)
print("Kmeans + two ridge regression models' MSE is: \n", score)

# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# X = x_train
# y = y_train
# clf = LinearDiscriminantAnalysis()
# clf.fit(X, y)
# LinearDiscriminantAnalysis()
# predict = clf.predict(x_train)
# print(mean_squared_error(y, predict))

from sklearn.linear_model import LinearRegression

model = LinearRegression()                                # build a linear regression model
model.fit(x_train, y_train) 
predict = model.predict(x_test)
score = mean_squared_error(y_test, predict) 
plt.plot(predict)
print(score)


