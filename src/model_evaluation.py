"""
Module d'évaluation du modèle Titanic.
Auteur: marias0156 (Miguel Hurtado)
"""

import pickle
import pandas as pd
from data_preprocessing import load_data, preprocess_data

def load_model(filename="titanic_model.pkl"):
    """Charge un modèle entraîné depuis un fichier."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def generate_predictions(model, X_test, test_data):
    """Génère les prédictions et sauvegarde le fichier submission.csv."""
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    print("Les prédictions ont été enregistrées dans submission.csv.")

if __name__ == "__main__":
    train_data, test_data = load_data("train.csv", "test.csv")
    X_train, X_test = preprocess_data(train_data, test_data)

    model = load_model()
    generate_predictions(model, X_test, test_data)
