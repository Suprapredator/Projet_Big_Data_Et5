# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:37:53 2018

@author: qjardet
"""

from scipy.io import loadmat;
import apprentissage as app;
import pretraitement as pre;
import sys


def printPicture(data, index, RVB):
    print(data['X'][:,:,RVB,index])

if __name__ == "__main__":
    data = loadmat(str(sys.argv[1][2:]))
    
    print("Nous allons donc utiliser le fichier d'entrée comme base pour l'entrainement et le test.")
    print("Voulez-vous que les données soient pré-traitées ? (y/n)")
    inp = input()
    while(inp != 'y' and inp != 'n'):
        inp = input()
    
    if inp == 'y':
        data = pre.traitementBinarisation(data, 'perfect_data.mat')
    
    print("Combien de données coulez-vous pour le test ? (sachant que toutes les données du fichier sont utilisées pour l'entrainement)")
    inp = input()
    
    app.test(data, data, int(inp), -1)