import numpy as np

class Kfold:
    def __init__(self,K):
        """
		Initialiseur
		Args : 
			* k 		: nombre de fold 	(int)
		"""
        self.K = int(K)
    def split(X):
        """
        Cette fonction séparare un jeu de donnée X en K échantillon égale
        Args :
            * X : Jeu de données (np.matrix)
        Retours :
            * train_index   : liste des indexs des échantillons d'entrainement  (np.arrays)
            * test_index    : liste des indexs des échantillons de test         (np.arrays)
        """
        
        pass