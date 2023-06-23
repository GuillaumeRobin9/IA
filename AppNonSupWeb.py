#Creation d'une fonction qui prend en entree latitude, longitude et les différents centroids des clusters et qui retourne le cluster auquel appartient l'accident
#Le script vérifie auprès de quel cluster l'accident est le plus proche et retourne le numéro du cluster
#Le script retourne un fichier json contenant le numéro du cluster
#Le script est appelé par le script AppNonSupWeb.py

import json
from sklearn.cluster import KMeans
import sys
import numpy as np

def cluster_accident(latitude, longitude, centroids):

    #Initialisation du modèle KMeans
    kmeans = KMeans(n_clusters=len(centroids), random_state=0, init=centroids, n_init=1)
    #Entrainement du modèle
    kmeans.fit(centroids)
    #Prédiction du cluster de l'accident
    cluster = kmeans.predict([[latitude, longitude]])
    #Conversion du résultat en format JSON
    result = {'cluster': int(cluster[0])}
    json_result = json.dumps(result)
    
    return json_result


#-----------------------------------------------------MISE EN PLACE DES ARGUMENTS-----------------------------------------------------#
#latitude au format float
#longitude au format float
#centroids au format [[latitude, longitude], [latitude, longitude], ...]
#-------------------------------------------------------------------------------------------------------------------------------------#

#chargement du json contenant les centroids json/centroids.json
with open(sys.argv[3]) as json_file:
    centroids = json.load(json_file)

# print(centroids)
json_result = cluster_accident(float(sys.argv[1]), float(sys.argv[2]), np.array(centroids))
print(json_result)


#booleen pour choisir si on veut exporter le resultat dans un fichier json ou non
export = False

if export == True:
    #export du resultat json_result dans un fichier json
    with open('json/result_non_sup.json', 'w') as outfile:
        json.dump(json_result, outfile)