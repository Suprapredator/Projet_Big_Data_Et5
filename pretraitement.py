import numpy as np;
import matplotlib.pyplot as plt;
from scipy.io import loadmat;
from scipy.io import savemat;
import math as mt;
import copy

### REGARDER NUMPY PLS "Ca fait le café" -Quentin 2018

def affinerContours(dataTraitees, index):
	# Détermination es valeurs moyennes
	moyennes = [0, 0, 0];
	for i in range(32):
		for j in range (32):
			for k in range (3):
				moyennes[k] += dataTraitees['X'][i, j, k, index]
		
	
	for k in range (0, 3):
		moyennes[k] /= 32*32
	
	# Création de la nouvelle image
	for i in range(32):
		for j in range (32):
			couleur = [0, 0, 0]
			for k in range (3):
				if dataTraitees['X'][i, j, k, index] > moyennes[k]:
					couleur[k] = 255
				else:
					couleur[k] = 0

			if sum(couleur) > 255 :
				dataTraitees['X'][i, j, :, index] = [0, 0, 0]
			else:
				dataTraitees['X'][i, j, :, index] = [255, 255, 255]

	return dataTraitees

def traitementBinarisation(data, name):
    dataTraitees = copy.deepcopy(data)
    
    for i in range(len(data['y'])):
        dataTraitees = affinerContours(dataTraitees, i)
    
    ecritureFichier(dataTraitees, name)
    return dataTraitees

def ecritureFichier(data, name):
    savemat('../'+name, data)
    

if __name__ == "__main__":

    train_data = loadmat('../train_32x32.mat')
    #test_data = loadmat('../test_32x32.mat')
    
    #ecritureFichier(train_data, 'perfect_train_data.mat')    
    
    #test_data = loadmat('../perfect_train_data.mat') 
        
    #image_idx = 2;
    #nouvelImage = traitementBinarisation(train_data)
    #plt.imshow(test_data['X'][:, :, :, image_idx])
    #plt.show()
    #plt.imshow(nouvelImage['X'][:, :, :, image_idx])
    #plt.show()
