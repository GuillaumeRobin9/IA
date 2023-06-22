import json
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_gravite(accident_info, csv_file):
    # Lecture du fichier CSV
    df = pd.read_csv(csv_file, sep=';')

    # Création du modèle KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    # Entrainement du modèle
    knn.fit(df.drop('descr_grav', axis=1), df['descr_grav'])
    
    # Extraction des valeurs du dictionnaire et conversion en nombres réels
    accident_info_values = [
        accident_info['date'],
        accident_info['latitude'],
        accident_info['longitude'],
        accident_info['descr_cat_veh'],
        accident_info['descr_agglo'],
        accident_info['descr_athmo'],
        accident_info['descr_lum'],
        accident_info['descr_etat_surf'],
        accident_info['description_intersection'],
        accident_info['age'],
        accident_info['place'],
        accident_info['descr_dispo_secu'],
        accident_info['descr_motif_traj'],
        accident_info['descr_type_col']
    ]
    
    # Prédiction de la gravité de l'accident
    gravite = knn.predict(np.array([accident_info_values]).reshape(1, -1))[0]

    # Conversion du résultat en format JSON
    result = {'gravite': int(gravite)}
    json_result = json.dumps(result)
    
    return json_result

accident_info = json.loads(sys.argv[1])[0]  # Accéder au premier élément de la liste
json_result = knn_gravite(accident_info, sys.argv[2])
print(json_result)

#export du resultat json_result dans un fichier json
with open('json/result_sup.json', 'w') as outfile:
    json.dump(json_result, outfile)
