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
    result = {'gravite': gravite}
    json_result = json.dumps(result)
    
    return json_result

json_result = high_level_classification(json.loads(sys.argv[1]), sys.argv[2], sys.argv[3])
print(json_result)


#export du resultat json_result dans un fichier json
with open('json/result_clf.json', 'w') as outfile:
    json.dump(json_result, outfile)