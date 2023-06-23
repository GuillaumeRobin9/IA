import json
import joblib
import sys

def high_level_classification(accident_info, classification_method, trained_model):
    # Utilisation du modèle de classification spécifié
    if classification_method == 'SVM':
        model = joblib.load(trained_model)
        # Effectuer la prédiction avec le modèle 1
        
    elif classification_method == 'RF':
        model = joblib.load(trained_model)
        # Effectuer la prédiction avec le modèle 2
    
    elif classification_method == 'MLP':
        model = joblib.load(trained_model)
        
    # ... Ajouter d'autres conditions pour les autres modèles
    
    # Prédiction de la gravité de l'accident
    gravite = model.predict([accident_info])[0]
    
    # Conversion du résultat en format JSON
    result = {'gravite': int(gravite)}
    json_result = json.dumps(result)
    
    return json_result


#-----------------------------------------------------MISE EN PLACE DES ARGUMENTS-----------------------------------------------------#
#accident_info au format [{"premiere_cle": valeur, "deuxieme_cle": valeur, ...}"}] avec les memes infos qu'une ligne du csv sans descr_grav
#classification_method au format "SVM", "RF" ou "MLP"
#trained_model au format "models/nom_du_fichier.pkl"
#-------------------------------------------------------------------------------------------------------------------------------------#

accident_info = json.loads(sys.argv[1])[0]
json_result = high_level_classification(accident_info, sys.argv[2], sys.argv[3])
print(json_result)


#booleen pour choisir si on veut exporter le resultat dans un fichier json ou non
export = False

if export == True:
    #export du resultat json_result dans un fichier json
    with open('json/result_clf.json', 'w') as outfile:
        json.dump(json_result, outfile)