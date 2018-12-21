# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
from sklearn.neighbors import KNeighborsClassifier;
import time;

def getSklearnInstance(data, tailleTrain):
    X = list()
    Y = list()    
    
    instance = KNeighborsClassifier(n_neighbors=10, weights='uniform')
    nx, ny, npixel, nombre_d_elements = data['X'][:,:,:,:].shape
    #tampon = data['X'][:,:,:,:].reshape(nombre_d_elements,nx*ny*npixel)    

    if tailleTrain == -1:
        tailleTrain = nombre_d_elements
    
    for i in range(tailleTrain):
        Y.append(data['y'][i])
        X.append(data['X'][:,:,:,i].reshape(nx*ny*npixel))
    
    instance.fit(X, np.ravel(Y))
    
    return instance
    
def donneMoiLaClasse(X, SklearnInstance):
    nx, ny, npixel = X.shape
    return SklearnInstance.predict(X.reshape(1,nx*ny*npixel)) 
    
def test(train_data, test_data, tailleTest, tailleTrain):
    matrice = np.zeros((10,10))    
    start_time = time.time()
    bonneReponse = 0;
    SklearnInstance = getSklearnInstance(train_data, tailleTrain)

    start_time2 = time.time()
    
    if tailleTest == -1 or tailleTest > len(test_data['y']):
        tailleTest = len(test_data['y'])
    
    for i in range(tailleTest):
        classePredite = donneMoiLaClasse(test_data["X"][:, :, :, i], SklearnInstance)
        matrice[test_data['y'][i][0]%10][classePredite%10] += 1
        if test_data['y'][i]%10 == classePredite:
            bonneReponse += 1
    
    print('Résultat par Sklearn: '+str(bonneReponse)+'/'+str(tailleTest))
    print("--- temps creation instance = %s seconds ---" % (start_time2 - start_time))
    print("--- temps recherche voisin = %s seconds ---" % (time.time() - start_time2))
    print("--- temps total = %s seconds ---" % (time.time() - start_time))
    print("Matrice de confusion")    
    print(matrice)

