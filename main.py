# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import moyenne as moy;
import apprentissage as app;
import pretraitement as pre;
import acp as pca


def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])

if __name__ == "__main__":
    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')
    perfect_train_data = loadmat('../perfect_train_data.mat') #pre.traitementBinarisation(train_data, 'perfect_train_data.mat')    
    perfect_test_data = loadmat('../perfect_test_data.mat') #pre.traitementBinarisation(test_data, 'perfect_test_data.mat')     

    d2_data = pca.reductionDimension(perfect_train_data)
    moyennes2D = pca.getMoyennes2d(d2_data)    
    moyennes = moy.getMoyennes(perfect_train_data)
    
    moy.test(moyennes, perfect_test_data, -1)
    app.test(perfect_train_data, perfect_test_data, 100)
    pca.test(moyennes2D, d2_data, -1)    
    
    
    #print(perfect_train_data['X'][:,:,:,0]);
    #plt.imshow(train_data['X'][:,:,:,1]);
    #plt.show()