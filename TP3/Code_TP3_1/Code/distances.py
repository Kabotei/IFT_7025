import numpy as np
from scipy.spatial import distance

def minkowski(p,x1,x2):
    """
    Cette fonction applique le calcul de la distance de minkowski
    entre deux vecteurs x1 et x2

    Args :
        x1 : vecteur 1 					(np.array)
        x2 : vecteur 2 					(np.array)
        p  : coefficient de minkowski 	(int)
    
    Retours :
        dist : La distance de mikowski entre les vecteur x1 et x2 (float)
    """

    # Pour Abalones :
    ### Si l'attribut Sexe est different
    ###     Alors dif = 1
    ### Sinon
    ###     dif = 0

    if np.size(x1) == 8:
        temp = 1
        if x1[0,0] == x2[0,0]:
            temp = 0
        
        dif = abs(np.array(x1[0,1:],dtype = float) - np.array(x2[0,1:],dtype = float))
        dif = np.insert(dif,0,temp)

    else:
        dif = abs(x1-x2)
    square = np.power(dif,p)
    som = np.sum(square)
    dist = som**(1/p)
    return dist

# Pour les fonctions suivantes scipy a été utilisé afin de tester differenté métrique pour le classifieur KNN

def cosine(x1,x2):
    """
    Cette fonction applique le calcul de la distance cosine
    entre deux vecteurs x1 et x2

    Args :
        x1 : vecteur 1 					(np.array)
        x2 : vecteur 2 					(np.array)
    
    Retours :
        dist : La distance de mikowski entre les vecteur x1 et x2 (float)
    """
    return distance.cosine(x1,x2)

def hamming(x1,x2):
    """
    Cette fonction applique le calcul de la distance de Hamming
    entre deux vecteurs x1 et x2

    Args :
        x1 : vecteur 1 					(np.array)
        x2 : vecteur 2 					(np.array)
    
    Retours :
        dist : La distance de mikowski entre les vecteur x1 et x2 (float)
    """
    return distance.hamming(x1,x2)

def jaccard(x1,x2):
    """
    Cette fonction applique le calcul de la distance de Jaccard
    entre deux vecteurs x1 et x2

    Args :
        x1 : vecteur 1 					(np.array)
        x2 : vecteur 2 					(np.array)
    
    Retours :
        dist : La distance de mikowski entre les vecteur x1 et x2 (float)
    """
    return distance.jaccard(x1,x2)