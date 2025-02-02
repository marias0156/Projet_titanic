"""
Module de prétraitement des données du Titanic.
Auteur: marias0156 (Miguel Hurtado)
"""

import pandas as pd

def load_data(train_path, test_path):
    """Charge les données Titanic depuis les fichiers CSV."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """Nettoie et prépare les données pour l'entraînement et les tests."""
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    train_data = pd.get_dummies(train_data[features])
    test_data = pd.get_dummies(test_data[features])
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_data("train.csv", "test.csv")
    X_train, X_test = preprocess_data(train_data, test_data)
    print("Prétraitement terminé.")
