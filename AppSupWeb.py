import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys

def knn_accident(accident_info, csv_file):
    # Chargement des données d'entraînement depuis le fichier CSV
    accidents = pd.read_csv(csv_file)
    X = accidents.drop('descr_grav', axis=1)
    y = accidents['descr_grav']
    
    # Initialisation du modèle KNN
    knn = KNeighborsClassifier()
    
    # Entraînement du modèle sur les données d'entraînement
    knn.fit(X, y)
    
    # Prédiction de la gravité de l'accident
    gravite = knn.predict([accident_info])[0]
    
    # Conversion du résultat en format JSON
    result = {'gravite': gravite}
    json_result = json.dumps(result)
    
    return json_result


json_result = knn_accident(json.loads(sys.argv[1]), sys.argv[2])
print(json_result)