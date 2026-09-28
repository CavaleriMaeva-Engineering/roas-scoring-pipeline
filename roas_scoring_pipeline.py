import pandas as pd

## Logs bruts de pixels issus du tracking (1 ligne = 1 clic)
raw_logs = [
    {
        "user_id": "usr_101",
        "event": "page_view",
        "payload": '{"page": "/home"}',
    },
    {
        "user_id": "usr_101",
        "event": "page_view",
        "payload": '{"page": "/pricing"}',
    },
    {
        "user_id": "usr_101",
        "event": "add_to_cart",
        "payload": '{"cart_value": 85}',
    },
    {
        "user_id": "usr_101",
        "event": "purchase",
        "payload": '{"amount": 85}',
    },
    {
        "user_id": "usr_102",
        "event": "page_view",
        "payload": '{"page": "/home"}',
    },
    {
        "user_id": "usr_102",
        "event": "add_to_cart",
        "payload": '{"cart_value": 120}',
    },  # usr_102 n'a PAS acheté !
    {
        "user_id": "usr_103",
        "event": "page_view",
        "payload": '{"page": "/home"}',
    },
]

df_raw = pd.DataFrame(raw_logs)

def build_features(df) :
    features_list=[]

    #on regroupe toutes les lignes par utilisateur (GROUP BY)
    for user_id, group in df.groupby('user_id') : 
        #on extrait la colonne event de cet utilisateur et on la transforme en list python
        events = list(group['event'])

        has_cart = 1 if 'add_to_cart' in events else 0
        has_purchase =1 if 'purchase' in events else 0 

        panier_abandonne = 1 if (has_cart==1 and has_purchase==0) else 0

        #on stocke les variable calculées dans un dictionnaire 
        user_dict = {
            'user_id' : user_id,
            'nb_events' : len(events), #nbr total de clics
            'panier_abandonne' : panier_abandonne,
            'a_achete' : has_purchase,
        }
        features_list.append(user_dict)

    df_features = pd.DataFrame(features_list)
    return df_features

df_features = build_features(df_raw)
print(df_features)

## Entraînement du modèle de ML avec Scikit-Learn
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#on fixe le hasard pour rendre le code reproductible sinon les résultats seraient différents à chaque fois
np.random.seed(42)
N=500

X_historique = pd.DataFrame({
    'nb_events' : np.random.randint(1,10,N),
    'panier_abandonne' : np.random.choice([0,1], N, p=[0.7,0.3]),
})

#y vaut 1 si achat sinon 0 
#on simule un comportement réel : un panier abandonné augmente les chances d'achat futur
proba_achat = 1 / (1+np.exp(-0.3*X_historique['nb_events'] +1.5* X_historique['panier_abandonne']-2) )

y_historique = np.random.binomial(1, proba_achat)

#on sépare les données (80% entraînement et 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_historique, y_historique, test_size=0.2, random_state=42)

#on isntancie et entraîne le modèle random forest 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f'nbr dexemples dentraînement : {len(X_train)}')
print(f'classes gérées par le modèle : {model.classes_} (0=Non-acheteur, 1=Acheteur)')

#scoring 

#profil d'un nv visiteur en direct 
nv_visteur = pd.DataFrame([{
    'nb_events' : 4,
    'panier_abandonne' : 1
}])

# le modèle calcule la proba d'achat (score entre 0 et 1)
score_achat = model.predict_proba(nv_visteur)[0][1] #renvoie la prba d'achat (1)

#règle décisionnel pour optimiser le ROAS 
def decision_enchere(score) :
    if score >=0.7 :
        return "enchère fort, retargeting prioritaire"
    elif score >=0.35 : 
        return "enchère modérée"
    else : 
        return "exclusion du ciblage"
    
print(f"score d'intention d'achat prédit : {score_achat:.2f} (soit {score_achat * 100:.1f}% de chance de conversion)")
print(f"décision Média / ROAS : {decision_enchere(score_achat)}")

