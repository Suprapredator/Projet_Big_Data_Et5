# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
import math as mt;
import moyenne as moy;
import apprentissage as app;
import pretraitement as pre;


def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])

if __name__ == "__main__":
    train_data = loadmat('../train_32x32.mat')
    test_data = loadmat('../test_32x32.mat')
    perfect_train_data = loadmat('../perfect_train_data.mat') #pre.traitementBinarisation(train_data, 'perfect_train_data.mat')    
    perfect_test_data = pre.traitementBinarisation(test_data, 'perfect_test_data.mat') #loadmat('../perfect_test_data.mat')     
    
    moyennes = moy.getMoyennes(perfect_train_data)
    
    moy.test(moyennes, perfect_train_data, -1)
    app.test(perfect_train_data, perfect_test_data, 100)
    
    
    #print(perfect_train_data['X'][:,:,:,0]);
    plt.imshow(moyennes[2]);
    plt.show()