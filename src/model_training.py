"""
Module d'entraînement du modèle Titanic.
Auteur: marias0156 (Miguel Hurtado)
"""

import pickle
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data

def train_model(X_train, y_train):
    """Entraîne un modèle RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename="titanic_model.pkl"):
    """Sauvegarde le modèle entraîné."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_data, test_data = load_data("train.csv", "test.csv")
    X_train, X_test = preprocess_data(train_data, test_data)
    y_train = train_data["Survived"]

    model = train_model(X_train, y_train)
    save_model(model)
    print("Modèle entraîné et sauvegardé.")
