import numpy as np;
from scipy.io import loadmat;
import math as mt;
import copy
from sklearn.decomposition import PCA
import time

def reductionDimension(data):    
	ligne, colonne, rgb, index = data['X'][:, :, :, :].shape
	images = data['X'][:, :, :, :].reshape(index, ligne * colonne * rgb)
	
	testPCA = PCA(n_components=10, copy=False)

	debut = time.time()
	reduc = testPCA.fit_transform(images)
	print("Création instance PCA %s sec" % (time.time() - debut))

	#print("copie des donnees")
	d2_data = copy.deepcopy(data)
	d2_data['X'] = reduc

	return d2_data

#def reductionDimension(data):
    #X = list()    
    
    #ligne, colonne, rgb, index = data['X'][:, :, :, :].shape
    
    #for i in range(index):
    #    Y.append(data['y'][i])
    #    X.append(data['X'][:,:,:,i].reshape(nx*ny*npixel))
    
    #testPCA = PCA(n_components=10, copy=False)
    
    #debut = time.time()
    #reduc = testPCA.fit_transform(X)
    #print("Création instance PCA %s sec" % (time.time() - debut))
    

def getMoyennes2d(trainData):
	moyenne = []
	
	for i in range(1,11):
		moyenne.append(calculerImage2dMoyenneClasseX(i,trainData))
	
	return moyenne

def calculerImage2dMoyenneClasseX(X,data):
	listeImage = []
	
	for i in range(len(data['y'])):
		if data['y'][i] == X:
			listeImage.append(data['X'][i, :])

	return np.mean(listeImage, axis=0).astype(int)

def donneMoiLaClasse(image, moyenne):
	distance = mt.inf 
	
	for i in range(len(moyenne)):
		newDistance = np.linalg.norm(image-moyenne[i])
		if distance > newDistance:
			bonneClasse = i+1
			distance = newDistance
	
	return bonneClasse

def test(moyenne, data, taille):
	start_time = time.time()
	bonneReponse = 0;
 
	if taille == -1 or taille > len(data['y']):
		taille = len(data['y'])
	
	for i in range(taille):
		if data['y'][i] == donneMoiLaClasse(data["X"][i, :], moyenne):
			bonneReponse += 1
	
	print('Résultat par PCA: '+str(bonneReponse)+'/'+str(taille)+' ('+str(bonneReponse*100/taille)+'%)')
	print("--- %s seconds ---" % (time.time() - start_time)) 

if __name__ == "__main__":

	print("initialisation matrices...")
	#train_data = loadmat('../train_32x32.mat')
	#test_data = loadmat('../test_32x32.mat')
	perf_data = loadmat('../perfect_train_data.mat')
	#ptest_data = loadmat('../perfect_test_data.mat')

	d2_data = reductionDimension(perf_data)
	moyennes = getMoyennes2d(d2_data)
	test(moyennes, d2_data, -1)
