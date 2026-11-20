import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def charger_fichier(file):
     # Charger le fichier CSV dans un DataFrame pandas
    df = pd.read_csv(file)
     # Améliore automatiquement les types des colonnes 
     # (ex: object → int/float si possible)
    df = df.infer_objects()
     # Convertit les types en types pandas modernes 
     # (plus précis : Int64, string, etc.)
    df = df.convert_dtypes()
    return df


def get_types_donnees(df):
    return pd.DataFrame({
        "Colonne": df.columns,  
        "Type": df.dtypes.astype(str).values
    })


def get_statistiques(df):
    # garder seulement les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=["number"])
    if colonnes_numeriques.empty:
        return pd.DataFrame({"Message": ["Aucune colonne numérique disponible."]})
    
    # statistiques : moyenne, min, max, etc.
    return colonnes_numeriques.describe()


def get_valeurs_manquantes(df):
    return pd.DataFrame({
        "Colonne": df.columns,
        "Valeurs manquantes": df.isnull().sum().values
    })


def nettoyer_donnees(df, methode):
    df_clean = df.copy()

    if methode == "Aucune":
        return df_clean

    # =========================
    # Colonnes numériques
    # =========================
    colonnes_numeriques = df_clean.select_dtypes(include=["number"]).columns

    # =========================
    # Colonnes non numériques
    # =========================
    colonnes_non_numeriques = df_clean.select_dtypes(exclude=["number"]).columns

    # =========================
    # SUPPRESSION
    # =========================
    if methode == "Supprimer les lignes avec valeurs manquantes":
        df_clean = df_clean.dropna()

    # =========================
    # MOYENNE
    # =========================
    elif methode == "Remplir par la moyenne":
        if len(colonnes_numeriques) > 0:
            df_clean[colonnes_numeriques] = df_clean[colonnes_numeriques].astype(float)
            df_clean[colonnes_numeriques] = df_clean[colonnes_numeriques].fillna(
                df_clean[colonnes_numeriques].mean()
            )

    # =========================
    # MEDIANE
    # =========================
    elif methode == "Remplir par la médiane":
        if len(colonnes_numeriques) > 0:
            df_clean[colonnes_numeriques] = df_clean[colonnes_numeriques].astype(float)
            df_clean[colonnes_numeriques] = df_clean[colonnes_numeriques].fillna(
                df_clean[colonnes_numeriques].median()
            )

    # =========================
    # MODE 
    # =========================
    elif methode == "Remplir par le mode":
        for col in df_clean.columns:
            mode_col = df_clean[col].mode()
            if not mode_col.empty:
                df_clean[col] = df_clean[col].fillna(mode_col.iloc[0])

    # =========================
    # 
    # Toujours remplir les non numériques par le mode
    # =========================
    for col in colonnes_non_numeriques:
        mode_col = df_clean[col].mode()
        if not mode_col.empty:
            df_clean[col] = df_clean[col].fillna(mode_col.iloc[0])

    return df_clean

def normaliser_donnees(df, methode):
    df_normalized = df.copy()
    colonnes_numeriques = df_normalized.select_dtypes(include=["number"]).columns

    if len(colonnes_numeriques) == 0:
        return df_normalized

    if methode == "Aucune":
        return df_normalized

    elif methode == "Min-Max Scaling":
        scaler = MinMaxScaler()
        df_normalized[colonnes_numeriques] = scaler.fit_transform(
            df_normalized[colonnes_numeriques]
        )

    elif methode == "Standardisation":
        scaler = StandardScaler()
        df_normalized[colonnes_numeriques] = scaler.fit_transform(
            df_normalized[colonnes_numeriques]
        )

    return df_normalized


def get_colonnes_numeriques(df):
    return list(df.select_dtypes(include=["number"]).columns)


def preparer_clustering(df):
    return df.select_dtypes(include=["number"]).copy()