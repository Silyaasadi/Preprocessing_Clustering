
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# DETECTION DES ATTRIBUTS
# =========================
def detecter_types_attributs(df, attributs):

    numeriques = []
    categoriels = []

    for col in attributs:

        serie = df[col].dropna()

        # Attribut non numérique
        if not pd.api.types.is_numeric_dtype(serie):

            categoriels.append(col)

        # Attribut binaire
        elif serie.nunique() == 2:

            categoriels.append(col)

        # Attribut catégoriel encodé potentiel
        elif (
            serie.nunique() <= 10
            and (serie % 1 == 0).all()
        ):

            categoriels.append(col)

        # Attribut quantitatif
        else:

            numeriques.append(col)

    return numeriques, categoriels


# =========================
# CHOIX DE LA DISTANCE
# =========================
def choisir_distance(numeriques, categoriels):

    if numeriques and not categoriels:

        return "Euclidienne"

    elif categoriels and not numeriques:

        return "Hamming"

    elif numeriques and categoriels:

        return "Mixte"

    else:

        raise ValueError("Aucun attribut sélectionné.")


# =========================
# PREPARATION DE X ET Y
# =========================
def preparer_classification(df, attributs, cible):

    if not attributs:

        raise ValueError(
            "Sélectionnez au moins un attribut explicatif."
        )

    if cible in attributs:

        raise ValueError(
            "La variable cible doit être exclue de X."
        )

    X = df[attributs].copy()

    y = df[cible].copy()

    if X.isna().any().any() or y.isna().any():

        raise ValueError(
            "Des valeurs manquantes existent encore."
        )

    if y.nunique() < 2:

        raise ValueError(
            "La cible doit contenir au moins deux classes."
        )

    # Conserver les classes comme des étiquettes
    y = y.astype(str)

    return X, y


# =========================
# DIVISION TRAIN / TEST
# =========================
def diviser_donnees(X, y, test_size=0.2):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


# =========================
# ENCODAGE DES ATTRIBUTS
# CATEGORIELS
# =========================
def encoder_attributs(
    X_train,
    X_test,
    numeriques,
    categoriels
):

    train = X_train.copy()
    test = X_test.copy()

    if categoriels:

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        train_cat = encoder.fit_transform(
            train[categoriels].astype(str)
        )

        test_cat = encoder.transform(
            test[categoriels].astype(str)
        )

        for i, col in enumerate(categoriels):

            train[col] = train_cat[:, i]
            test[col] = test_cat[:, i]

    ordre = numeriques + categoriels

    train = train[ordre].astype(float)
    test = test[ordre].astype(float)

    if not np.isfinite(train.to_numpy()).all():

        raise ValueError(
            "Valeurs infinies dans les données d'entraînement."
        )

    if not np.isfinite(test.to_numpy()).all():

        raise ValueError(
            "Valeurs infinies dans les données de test."
        )

    return train, test


# =========================
# DISTANCE MIXTE
# =========================
def distance_mixte(a, b, nb_numeriques):

    # Partie quantitative
    a_num = a[:nb_numeriques]
    b_num = b[:nb_numeriques]

    # Partie qualitative
    a_cat = a[nb_numeriques:]
    b_cat = b[nb_numeriques:]

    # Distance euclidienne ajustée
    d_euclidienne = np.sqrt(
        np.mean((a_num - b_num) ** 2)
    )

    # Distance de Hamming
    d_hamming = np.mean(a_cat != b_cat)

    # Combinaison des deux distances
    return (d_euclidienne + d_hamming) / 2


# =========================
# APPLICATION DE KNN
# =========================
def appliquer_knn(
    X_train,
    X_test,
    y_train,
    k,
    numeriques,
    categoriels
):

    if k < 1 or k > len(X_train):

        raise ValueError(
            f"K doit être compris entre 1 et {len(X_train)}."
        )

    # Détection automatique de la distance
    distance = choisir_distance(
        numeriques,
        categoriels
    )

    # Encodage des attributs catégoriels
    train, test = encoder_attributs(
        X_train,
        X_test,
        numeriques,
        categoriels
    )

    # =========================
    # DISTANCE EUCLIDIENNE
    # =========================
    if distance == "Euclidienne":

        modele = KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean"
        )

    # =========================
    # DISTANCE DE HAMMING
    # =========================
    elif distance == "Hamming":

        modele = KNeighborsClassifier(
            n_neighbors=k,
            metric="hamming",
            algorithm="brute"
        )

    # =========================
    # DISTANCE MIXTE
    # =========================
    else:

        modele = KNeighborsClassifier(
            n_neighbors=k,
            metric=distance_mixte,
            metric_params={
                "nb_numeriques": len(numeriques)
            },
            algorithm="brute"
        )

    # Entraînement
    modele.fit(train, y_train)

    # Prédiction
    y_pred = modele.predict(test)

    return modele, y_pred, distance


# =========================
# EVALUATION
# =========================
def evaluer_classification(y_test, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "Precision": precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1-score": f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )
    }


# =========================
# MATRICE DE CONFUSION
# =========================
def get_matrice_confusion(y_test, y_pred, classes):

    matrice = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    return pd.DataFrame(
        matrice,
        index=classes,
        columns=classes
    )


# =========================
# RAPPORT DE CLASSIFICATION
# =========================
def get_rapport_classification(y_test, y_pred):

    rapport = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(rapport).transpose()


# =========================
# TABLEAU DES PREDICTIONS
# =========================
def get_predictions(X_test, y_test, y_pred):

    resultat = X_test.copy()

    resultat["Classe réelle"] = y_test.to_numpy()

    resultat["Classe prédite"] = y_pred

    resultat["Résultat"] = np.where(
        y_test.to_numpy() == y_pred,
        "Correct",
        "Incorrect"
    )

    return resultat
