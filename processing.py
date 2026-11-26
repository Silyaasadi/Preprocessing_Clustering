
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =========================
# CHARGEMENT DES DONNEES
# =========================
def charger_fichier(file):

    df = pd.read_csv(file)

    # Amélioration des types de données
    df = df.infer_objects()

    # Conversion vers les types pandas modernes
    df = df.convert_dtypes()

    return df


# =========================
# TYPES DE DONNEES
# =========================
def get_types_donnees(df):

    return pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str).values
    })


# =========================
# STATISTIQUES DESCRIPTIVES
# =========================
def get_statistiques(df):

    colonnes_numeriques = df.select_dtypes(include=["number"])

    if colonnes_numeriques.empty:
        return pd.DataFrame({
            "Message": ["Aucune colonne numérique disponible."]
        })

    return colonnes_numeriques.describe()


# =========================
# VALEURS MANQUANTES
# =========================
def get_valeurs_manquantes(df):

    return pd.DataFrame({
        "Colonne": df.columns,
        "Valeurs manquantes": df.isnull().sum().values
    })


# =========================
# DETECTION DES ZEROS
# =========================
def detecter_zeros(df):

    colonnes_numeriques = df.select_dtypes(include=["number"])

    nombre_zeros = (colonnes_numeriques == 0).sum()

    pourcentage_zeros = (
        nombre_zeros / len(df) * 100
    ).round(2)

    resultat = pd.DataFrame({
        "Colonne": colonnes_numeriques.columns,
        "Nombre de zéros": nombre_zeros.values,
        "Pourcentage de zéros (%)": pourcentage_zeros.values
    })

    # Garder uniquement les colonnes contenant des zéros
    return resultat[resultat["Nombre de zéros"] > 0]


# =========================
# REMPLACEMENT DES ZEROS
# =========================
def remplacer_zeros(df, colonnes):

    df_clean = df.copy()

    colonnes_numeriques = get_colonnes_numeriques(df_clean)

    for col in colonnes:

        if col in colonnes_numeriques:

            # Conversion en float pour accepter NaN
            df_clean[col] = df_clean[col].astype(float)

            # Remplacer les zéros par NaN
            df_clean[col] = df_clean[col].replace(
                0, float("nan")
            )

    return df_clean


# =========================
# NETTOYAGE DES DONNEES
# =========================
def nettoyer_donnees(df, methode, colonnes_zeros=None):

    df_clean = df.copy()

    # Remplacer les zéros suspects par NaN
    if colonnes_zeros:
        df_clean = remplacer_zeros(df_clean, colonnes_zeros)

    # Colonnes numériques
    colonnes_numeriques = df_clean.select_dtypes(
        include=["number"]
    ).columns

    # Colonnes non numériques
    colonnes_non_numeriques = df_clean.select_dtypes(
        exclude=["number"]
    ).columns

    # =========================
    # MOYENNE
    # =========================
    if methode == "Remplir par la moyenne":

        for col in colonnes_numeriques:

            if df_clean[col].isna().any():

                df_clean[col] = df_clean[col].astype(float)

                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].mean()
                )

    # =========================
    # MEDIANE
    # =========================
    elif methode == "Remplir par la médiane":

        for col in colonnes_numeriques:

            if df_clean[col].isna().any():

                df_clean[col] = df_clean[col].astype(float)

                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].median()
                )

    # =========================
    # MODE AUTOMATIQUE
    # Colonnes non numériques
    # =========================
    for col in colonnes_non_numeriques:

        mode_col = df_clean[col].mode()

        if not mode_col.empty:

            df_clean[col] = df_clean[col].fillna(
                mode_col.iloc[0]
            )

    return df_clean


# =========================
# ANALYSE DES COLONNES
# AVANT NORMALISATION
# =========================
def analyser_colonnes_normalisation(df, colonne_cible=None):

    resultat = []

    for col in df.columns:

        serie = df[col].dropna()

        # Nombre de valeurs distinctes
        nb_uniques = serie.nunique()

        # Nom de la colonne en minuscules
        nom_col = str(col).lower().strip()

        # =========================
        # VARIABLE CIBLE
        # =========================
        if col == colonne_cible:

            type_col = "Variable cible"
            normaliser = False

        # =========================
        # COLONNE VIDE
        # =========================
        elif nb_uniques == 0:

            type_col = "Colonne vide"
            normaliser = False

        # =========================
        # IDENTIFIANT POTENTIEL
        # =========================
        elif (
            nom_col in ["id", "matricule", "identifiant"]
            or nom_col.endswith("_id")
            or nom_col.startswith("id_")
        ):

            type_col = "Identifiant potentiel"
            normaliser = False

        # =========================
        # DONNEES NON NUMERIQUES
        # =========================
        elif (
            not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):

            type_col = "Non numérique"
            normaliser = False

        # =========================
        # COLONNE CONSTANTE
        # =========================
        elif nb_uniques == 1:

            type_col = "Constante"
            normaliser = False

        # =========================
        # VARIABLE BINAIRE
        # =========================
        elif nb_uniques == 2:

            type_col = "Binaire"
            normaliser = False

        # =========================
        # VARIABLE CATEGORIELLE
        # OU DISCRETE POTENTIELLE
        # =========================
        elif nb_uniques <= 10 and (serie % 1 == 0).all():

            type_col = "Catégorielle ou discrète potentielle"
            normaliser = False

        # =========================
        # VARIABLE QUANTITATIVE
        # =========================
        else:

            type_col = "Quantitative probable"
            normaliser = True

        resultat.append({
            "Colonne": col,
            "Type détecté": type_col,
            "Valeurs distinctes": nb_uniques,
            "Normalisation conseillée": normaliser
        })

    return pd.DataFrame(resultat)


# =========================
# COLONNES RECOMMANDEES
# POUR LA NORMALISATION
# =========================
def get_colonnes_normalisables(df, colonne_cible=None):

    analyse = analyser_colonnes_normalisation(
        df, colonne_cible
    )

    if analyse.empty:
        return []

    return analyse.loc[
        analyse["Normalisation conseillée"],
        "Colonne"
    ].tolist()


# =========================
# NORMALISATION DES DONNEES
# COLONNE PAR COLONNE
# =========================
def normaliser_donnees(df, methodes, colonne_cible=None):

    df_normalized = df.copy()

    colonnes_numeriques = get_colonnes_numeriques(df)

    for col in colonnes_numeriques:

        # Ne pas normaliser la variable cible
        if col == colonne_cible:
            continue

        # Ignorer les colonnes vides ou constantes
        if df[col].nunique() <= 1:
            continue

        # Récupérer la méthode choisie
        methode = methodes.get(col, "Aucune")

        # =========================
        # MIN-MAX SCALING
        # =========================
        if methode == "Min-Max Scaling":

            scaler = MinMaxScaler()

            df_normalized[col] = scaler.fit_transform(
                df_normalized[[col]]
            ).ravel()

        # =========================
        # Z-SCORE
        # =========================
        elif methode == "Z-score":

            scaler = StandardScaler()

            df_normalized[col] = scaler.fit_transform(
                df_normalized[[col]]
            ).ravel()

    return df_normalized


# =========================
# RECUPERER LES COLONNES
# NUMERIQUES
# =========================
def get_colonnes_numeriques(df):

    return list(
        df.select_dtypes(include=["number"]).columns
    )


# =========================
# PREPARATION AU CLUSTERING
# =========================
def preparer_clustering(df):

    return df.select_dtypes(
        include=["number"]
    ).copy()
