import json
from sklearn.cluster import KMeans
import sys

def k_means_accident(latitude, longitude, centroids):
    # Création des données d'entrée
    accident_data = [[latitude, longitude]]

    # Initialisation du modèle k-means
    cluster_results = []
    for centroid in centroids:
        kmeans = KMeans(n_clusters=1, init=[centroid], n_init=1)
    
        # Entraînement du modèle sur les données d'entrée
        kmeans.fit(accident_data)
    
        # Prédiction du cluster d'appartenance de l'accident
        cluster = kmeans.predict(accident_data)[0]
        cluster_results.append(int(cluster))
    
    # Conversion du résultat en format JSON
    results = {'clusters': cluster_results}
    json_result = json.dumps(results)
    
    return json_result

json_result = k_means_accident(float(sys.argv[1]), float(sys.argv[2]), json.loads(sys.argv[3]))
print(json_result)

# Export du résultat json_result dans un fichier json
with open('json/result_nonSup.json', 'w') as outfile:
    outfile.write(json_result)
