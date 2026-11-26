import streamlit as st
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import ConfusionMatrixDisplay

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

from arbres import (
    construire_arbre,
    construire_cart,
    elaguer,
    elaguer_alpha,
    suite_alphas,
    validation_croisee,
    dessiner_arbre,
    predire,
    profondeur,
    compter_feuilles
)

import streamlit as st
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import ConfusionMatrixDisplay

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
    signature = (
        tuple(X.columns), tuple(map(str, X.dtypes)), y.name, test_size,
        tuple(pd.util.hash_pandas_object(X, index=True)),
        tuple(pd.util.hash_pandas_object(y, index=True))
    )
    if (st.session_state.get("signature_classification") != signature
            or "division_classification" not in st.session_state):
        st.session_state["division_classification"] = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        st.session_state["signature_classification"] = signature
    return st.session_state["division_classification"]


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




def interface_arbres(df, variable_y, variables_x, test_size):

    st.subheader("Arbres de décision")

    # Données issues du preprocessing
    X = df[variables_x]
    y = df[variable_y].astype(str)

    if not all(pd.api.types.is_numeric_dtype(X[c])
               for c in X.columns):
        st.error("Les attributs X doivent être numériques.")
        return

    if X.isna().any().any() or df[variable_y].isna().any():
        st.error("Traitez les valeurs manquantes.")
        return

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error("Les attributs contiennent des valeurs infinies.")
        return

    try:
        X_train, X_test, y_train, y_test = diviser_donnees(X, y, test_size)
    except ValueError as e:
        st.error(str(e))
        return

    signature = (
        st.session_state["signature_classification"],
        tuple(pd.util.hash_pandas_object(X_train, index=True)),
        tuple(pd.util.hash_pandas_object(y_train, index=True))
    )
    if (st.session_state.get("signature_arbres") != signature
            or "modeles_arbres" not in st.session_state):
        st.session_state["signature_arbres"] = signature
        st.session_state["donnees_arbres"] = (
            X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float),
            y_train.to_numpy(), y_test.to_numpy()
        )
        st.session_state["modeles_arbres"] = {}

    Xa, Xt, ya, yt = st.session_state["donnees_arbres"]

    noms = list(variables_x)
    classes = sorted(y.unique())

    # Deux boutons côte à côte
    col1, col2 = st.columns(2)

    with col1:
        if st.button("C4.5", use_container_width=True):
            st.session_state["algorithme_arbre"] = "C4.5"

    with col2:
        if st.button("CART", use_container_width=True):
            st.session_state["algorithme_arbre"] = "CART"

    algo = st.session_state.get("algorithme_arbre")

    if algo is None:
        return

    modeles = st.session_state["modeles_arbres"]

    # Construction une seule fois
    if algo not in modeles:

        with st.spinner("Construction de l'arbre..."):

            if algo == "C4.5":

                arbre_complet = construire_arbre(
                    Xa, ya, noms
                )

                arbre_elague = copy.deepcopy(arbre_complet)

                elaguer(arbre_elague)

                modeles[algo] = {
                    "avant": arbre_complet,
                    "apres": arbre_elague
                }

            elif algo == "CART":

                if len(ya) < 5:
                    st.error("Au moins 5 données d'apprentissage sont nécessaires.")
                    return

                arbre_complet = construire_cart(
                    Xa, ya, noms
                )

                alphas = suite_alphas(arbre_complet)

                scores = validation_croisee(
                    Xa, ya, noms, alphas
                )

                meilleur_score = scores.max()

                meilleur_alpha = alphas[
                    scores >= meilleur_score - 1e-12
                ].max()

                arbre_elague = copy.deepcopy(arbre_complet)

                elaguer_alpha(
                    arbre_elague,
                    meilleur_alpha
                )

                modeles[algo] = {
                    "avant": arbre_complet,
                    "apres": arbre_elague,
                    "alpha": meilleur_alpha,
                    "alphas": alphas,
                    "scores": scores
                }

    modele = modeles[algo]

    st.markdown(f"### Arbre {algo}")

    # Choisir avant ou après élagage
    choix = st.radio(
        "Élagage",
        ["Avant élagage", "Après élagage"],
        horizontal=True,
        key=f"elagage_{algo}"
    )

    if choix == "Avant élagage":
        arbre = modele["avant"]
    else:
        arbre = modele["apres"]

    # Affichage de l'arbre
    fig = dessiner_arbre(
        arbre,
        f"{algo} - {choix}",
        classes
    )

    st.pyplot(fig)
    plt.close(fig)

    # Prédictions
    y_pred_train = predire(arbre, Xa)
    y_pred_test = predire(arbre, Xt)

    # Indicateurs
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Profondeur",
        profondeur(arbre)
    )

    col2.metric(
        "Feuilles",
        compter_feuilles(arbre)
    )

    col3.metric(
        "Accuracy Train",
        f"{accuracy_score(ya, y_pred_train):.2%}"
    )

    col4.metric(
        "Accuracy Test",
        f"{accuracy_score(yt, y_pred_test):.2%}"
    )

    # Paramètre alpha pour CART
    if algo == "CART":

        st.write(
            f"Meilleur alpha : {modele['alpha']:.4f}"
        )

        with st.expander("Validation croisée"):

            fig, ax = plt.subplots()

            ax.plot(
                modele["alphas"],
                modele["scores"],
                marker="o"
            )

            ax.axvline(
                modele["alpha"],
                color="red",
                linestyle="--",
                label="Alpha choisi"
            )

            ax.set_xlabel("Alpha")
            ax.set_ylabel("Accuracy")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

    # Matrice de confusion
    st.subheader("Matrice de confusion")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        yt,
        y_pred_test,
        labels=classes,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    # Rapport de classification
    st.subheader("Rapport de classification")

    rapport = classification_report(
        yt,
        y_pred_test,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(rapport).T.round(4),
        use_container_width=True
    )

    # Comparaison réel / prédit
    st.subheader("Résultats des prédictions")

    resultats = pd.DataFrame({
        "Classe réelle": yt,
        "Classe prédite": y_pred_test
    })

    st.dataframe(
        resultats,
        use_container_width=True
    )