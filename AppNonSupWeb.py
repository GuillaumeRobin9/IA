import json
from sklearn.cluster import KMeans
import sys

def k_means_accident(latitude, longitude, centroids):
    # Création des données d'entrée
    accident_data = [[latitude, longitude]]

    # Initialisation du modèle k-means
    kmeans = KMeans(n_clusters=len(centroids), init=centroids, n_init=1)
    
    # Entraînement du modèle sur les données d'entrée
    kmeans.fit(accident_data)
    
    # Prédiction du cluster d'appartenance de l'accident
    cluster = kmeans.predict(accident_data)[0]
    
    # Conversion du résultat en format JSON
    result = {'cluster': cluster}
    json_result = json.dumps(result)
    
    return json_result



json_result = k_means_accident(float(sys.argv[1]), float(sys.argv[2]), json.loads(sys.argv[3]))
print(json_result)

#export du resultat json_result dans un fichier json
with open('json/result_nonSup.json', 'w') as outfile:
    json.dump(json_result, outfile)