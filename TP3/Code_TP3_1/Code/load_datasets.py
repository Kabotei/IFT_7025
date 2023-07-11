import numpy as np
import random

def normalize(matrice,columnIndex):

  """Cette fonction a pour but de normiliser les données d'une colonne donnée

    Args:
        - matrice : la matrice a normaliser

        - columnIndex: index de la colonne a normaliser

    Retours:
        Cette fontion retourne la matrice d'entrer avec sa colonne normaliser
    """
  value = [float(val[columnIndex]) for val in matrice]
  maxVal = max(value)
  minVal = min(value)

  for i in range(len(value)):
    matrice[i][columnIndex] = (float(matrice[i][columnIndex])-minVal)/(maxVal-minVal)
  return matrice

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
    
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
    
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
    
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
  # remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.

    # Lecture du dataset
    iris = f.read()
    iris = [[m for m in n.split(',')]for n in iris.split('\n')]

    # Normalisation
    colonneANormaliser = [0,1,2,3]

    for col in colonneANormaliser:
      iris = normalize(iris,col)
    
    # Mélange
    random.shuffle(iris)

    #Splitting
    train = iris[:round(train_ratio*len(iris))]
    test = iris[round(train_ratio*len(iris)):]

    # Extraction des labels
    train_labels = [conversion_labels[individu.pop(-1)] for individu in train]
    test_labels = [conversion_labels[individu.pop(-1)] for individu in test]
    
    # Conversion en numpy
    train = np.matrix(train)
    test = np.matrix(test)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
  
  
  
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
    
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
    
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
    
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

  
    # TODO : le code ici pour lire le dataset
    # Lecture du dataset
    vinhoVerde = f.read()
    vinhoVerde = [[m for m in n.split(',')]for n in vinhoVerde.split('\n')]
    vinhoVerde.pop(-1) #Suppression du retour à la ligne

    # Normalisation
    colonneANormaliser = range(11)

    for col in colonneANormaliser:
      vinhoVerde = normalize(vinhoVerde,col)
    
    
    # Mélange
    random.shuffle(vinhoVerde)

    #Splitting
    train = vinhoVerde[:round(train_ratio*len(vinhoVerde))]
    test = vinhoVerde[round(train_ratio*len(vinhoVerde)):]

    # Extraction des labels
    train_labels = [individu.pop(-1) for individu in train]
    test_labels = [individu.pop(-1) for individu in test]
  
    # Conversion en numpy
    train = np.matrix(train)
    test = np.matrix(test)

    train_labels = np.array(train_labels, dtype = int)
    test_labels = np.array(test_labels, dtype = int)

  # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    
    # Lecture du dataset
    abalones = f.read()
    abalones = [[m for m in n.split(',')]for n in abalones.split('\n')]
    abalones.pop(-1) #Suppression du retour à la ligne

    # Normalisation
    colonneANormaliser = range(1,8)

    for col in colonneANormaliser:
      abalones = normalize(abalones,col)
    
    # Mélange
    random.shuffle(abalones)

    #Splitting
    train = abalones[:round(train_ratio*len(abalones))]
    test = abalones[round(train_ratio*len(abalones)):]

    # Extraction des labels
    train_labels = [individu.pop(-1) for individu in train]
    test_labels = [individu.pop(-1) for individu in test]
  
    # Conversion en numpy
    train = np.matrix(train)
    test = np.matrix(test)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    return (train, train_labels, test, test_labels)