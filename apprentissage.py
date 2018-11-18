# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
from sklearn.neighbors import KNeighborsClassifier;

def getSklearnInstance(data):
    instance = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    nx, ny, npixel, nombre_d_elements = data['X'][:,:,:,:].shape
    tampon = data['X'][:,:,:,:].reshape(nombre_d_elements,nx*ny*npixel)  

    instance.fit(tampon, np.ravel(data['y']))
    
    return instance
    
def donneMoiLaClasse(X, SklearnInstance):
    nx, ny, npixel = X.shape
    return SklearnInstance.predict(X.reshape(1,nx*ny*npixel)) 
    
def test(data, taille):
    bonneReponse = 0;
    SklearnInstance = getSklearnInstance(data)
    
    if taille == -1:
        taille = len(data['y'])
    
    for i in range(taille):
        if data['y'][i]%10 == donneMoiLaClasse(data["X"][:, :, :, i], SklearnInstance):
            bonneReponse += 1
    
    print(bonneReponse)
