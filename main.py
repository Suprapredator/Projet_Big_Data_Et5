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
    perfect_train_data = loadmat('../perfect_train_data.mat') #pre.traitementBinarisation(train_data)    
    
    moyennes = moy.getMoyennes(perfect_train_data)
    
    #plt.imshow(moyennes[0])
    #plt.show()    
    
    print('Résultat par moyenne: ')
    #moy.test(moyennes, perfect_train_data, -1)
    print('Résultat par Sklearn:')
    app.test(perfect_train_data, 100)
    