def evaluation(confusionMatrix,classe):
    """
    Cette fonction evalue le classifieur d'une classe donnée
    par son exactitude, sa précision et son rappel
    lignes = prédictions
    colonnes = valeurs réelles

    Args:
        * confusionMatrix   : Matrice de confusion      (np.matrix)
        * classe            : Le numéro de la classe    (int)
    
    Retours
        * A : L'éxactitude de la classe en %    (float)
        * P : La précision de la classe en %    (float)
        * R : Le rappel de la classe en %       (float)
        * F : le F1-Score de la classe en %     (float)
    """
    TP = confusionMatrix[classe,classe]
    FP = int(confusionMatrix.sum(axis=1)[classe]) - TP
    FN = int(confusionMatrix.sum(axis=0)[classe]) - TP
    TN = confusionMatrix.sum() - TP - FN - FP


    # Calcul de la précision
    P = TP/(TP+FP)
    # Calcul du rappel
    R = TP/(TP+FN)

    # Calcul de l'accuracy
    A = (TP + TN)/(TP + TN + FP + FN)

    # Calcul du F1-Score
    F = 2*P*R/(P+R)

    return (A,P,R,F)

def affichage(matrice,A,P,R,F,classe):
    """Cette fonction affiche les évaluation du classifieur
    Args:
        * Matrice   : La matrice de confusion du classifieur    (np.matrix)
        * A         : L'exactitude de la classe en %            (float)
        * P         : La précision de la classe en %            (float)
        * R         : Le rappel de la classe en %               (float)
        * F         : le F1-Score de la classe en %             (float)
        * classe    : Le numéro de la classe                    (int)
    """
    ## Matrice de confusion :
    print(f"| Pour la classe {classe} |")
    print("--------------------")
    print("Matrice de confusion :")
    print("--------------------")


    for row in matrice:
        print("|",end=" ")
        for val in row:
            print(val,end =" ")
            print("|",end=" ")
        print()
        for val in row:
            print("_____",end=" ")
        print()
    

    ## Accuracy
    print(f"Acuracy : {A}")

    ## Precision
    print(f"Précision : {P}")

    ## Recall
    print(f"Recall : {R}")

    ## F1-Score
    print(f"F1-Score : {F}")

    print("\n")