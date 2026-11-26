import streamlit as st
import matplotlib.pyplot as plt
from clustering import appliquer_kmeans, appliquer_kmedoids, preparer_clustering
from processing import (
    charger_fichier,
    get_types_donnees,
    get_statistiques,
    get_valeurs_manquantes,
    nettoyer_donnees,
    normaliser_donnees,
    get_colonnes_numeriques
)
# from classification import (
#     appliquer_classification_knn,
#     appliquer_classification_decision_tree
# )
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.patches as patches
import streamlit as st
import streamlit.components.v1 as components

from clustering import appliquer_agnes, appliquer_dbscan, appliquer_diana

from classification import (
    detecter_types_attributs,
    choisir_distance,
    preparer_classification,
    diviser_donnees,
    appliquer_knn,
    evaluer_classification,
    get_matrice_confusion,
    get_rapport_classification,
    get_predictions
)

from classification import interface_arbres

# ==============================
# LOAD CSS
# ==============================
def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
load_css()



#  TOUJOURS EN PREMIER
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# ==============================
# HOME
# ==============================
if st.session_state["page"] == "home":
    components.html("""
    <div style="
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #0e1117;
    ">
        <div style="
            width: 60%;
            padding: 40px;
            border-radius: 20px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            text-align: center;
            color: white;
            font-family: Arial;
        ">
            <h1 style="font-size:40px;">Projet Fouille de Données</h1>

            <p> Prétraitement, Clustering et Classification</p>

            <p><b>Module :</b> Fouille de Données</p>
            <p><b>Réalisé par :</b> Silya Sadi & Manel ZANOUN </p>
        </div>
    </div>
    """, height=500)

    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        if st.button("🚀 Commencer le Prétraitement"):
            st.session_state["page"] = "volet1"
            st.rerun()

    st.stop()

# ==============================
# VOLET 1 : PRETRAITEMENT
# ==============================
if st.session_state["page"] == "volet1":

    if st.button("⬅️ Retour"):
        st.session_state["page"] = "home"
        st.rerun()

    st.title("Prétraitement")

    file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if file is not None:
        df = charger_fichier(file)
        st.session_state["df"] = df

      # RESET CLUSTERING quand dataset change
        for key in ["kmeans", "kmedoids", "agnes", "diana", "dbscan"]:
            if key in st.session_state:
                del st.session_state[key]


        # DERNIERE VERSION
        if "df_norm" in st.session_state:
            df_current = st.session_state["df_norm"]
        elif "df_clean" in st.session_state:
            df_current = st.session_state["df_clean"]
        else:
            df_current = df

        # st.dataframe(df_current)

        tab1, tab2, tab3, tab4 = st.tabs([ 
            "📊 Exploration",
            "🧹 Nettoyage",
            "⚙️ Normalisation",
            "📈 Visualisation"
        ])

        # ======================
        # EXPLORATION
        # ======================
        with tab1:
            st.subheader("Données originales")

            #  Tableau principal
            st.dataframe(df)

            #  Créer 2 colonnes
            col1, col2 = st.columns(2)

            with col1:
                st.write("Types de données")
                st.dataframe(get_types_donnees(df))

            with col2:
                st.write("Valeurs manquantes")
                st.dataframe(get_valeurs_manquantes(df))

            #  Statistiques en dessous
            st.write("Statistiques")
            st.dataframe(get_statistiques(df))
       
        # ======================
        # NETTOYAGE
        # ======================
        with tab2:
            st.subheader("Données avant nettoyage")
            st.dataframe(df)

            choix = st.selectbox("Méthode de nettoyage des données num", [
                "Aucune",
                "Remplir par la moyenne",
                "Remplir par la médiane"
            ])

            if st.button("Appliquer nettoyage"):
               df_clean = nettoyer_donnees(df, choix)
               st.session_state["df_clean"] = df_clean

               st.subheader("Données après nettoyage")
               st.dataframe(df_clean)

        # ======================
        # NORMALISATION
        # ======================
               # ======================
        # NORMALISATION
        # ======================
        with tab3:

            if "df_clean" in st.session_state:

                df_base = st.session_state["df_clean"]

                st.subheader("Données avant normalisation")
                st.dataframe(df_base)

                # Récupérer les colonnes numériques
                colonnes = get_colonnes_numeriques(df_base)

                st.subheader("Normalisation par colonne")

                st.info(
                    "Choisissez une méthode de normalisation "
                    "pour chaque colonne numérique."
                )

                # Dictionnaire des méthodes
                methodes = {}

                # Choix pour chaque colonne
                for col in colonnes:

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(col)

                    with col2:
                        methodes[col] = st.selectbox(
                            "Méthode",
                            [
                                "Aucune",
                                "Min-Max Scaling",
                                "Z-score"
                            ],
                            key=f"norm_{col}",
                            label_visibility="collapsed"
                        )

                # Appliquer la normalisation
                if st.button("Appliquer normalisation"):

                    df_norm = normaliser_donnees(
                        df_base, methodes
                    )

                    st.session_state["df_norm"] = df_norm

                # Afficher les résultats
                if "df_norm" in st.session_state:

                    st.subheader("Données après normalisation")

                    st.dataframe(
                        st.session_state["df_norm"]
                    )

                st.markdown("---")

                # Boutons de navigation
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("➡️ Clustering"):
                        st.session_state["page"] = "volet2"
                        st.rerun()

                with col2:
                    if st.button("➡️ Classification"):
                        st.session_state["page"] = "volet3"
                        st.rerun()

            else:
                st.warning("⚠️ Données non nettoyées")
        # =========================
        # VISUALISATION
        # =========================

        with tab4:  #  Onglet 4 : partie visualisation des données

            #  Vérifier si les données normalisées existent
            if "df_norm" in st.session_state:
                df_visu = st.session_state["df_norm"]  # on utilise les données normalisées
                st.dataframe(df_visu)  # affichage

            #  Sinon, utiliser les données nettoyées
            elif "df_clean" in st.session_state:
                df_visu = st.session_state["df_clean"]
                st.dataframe(df_visu)

            #  Sinon, utiliser les données originales
            else:
                df_visu = df
                st.dataframe(df_visu)

            #  Récupérer uniquement les colonnes numériques
            cols = get_colonnes_numeriques(df_visu)

            # ======================
            # BOXPLOT
            # ======================
            st.title("BoxPlot")
            #  Vérifier s’il y a des colonnes numériques
            if len(cols) > 0:
                selected = st.multiselect("Colonnes", cols)  # choix multiple des colonnes

                #  Si l'utilisateur a sélectionné des colonnes
                if selected:
                    col1, col2, col3 = st.columns([1,2,1])  # centrer le graphique

                    with col2:
                        #  création de la figure
                        fig, ax = plt.subplots(figsize=(4, 3))  # petite taille

                        #  boxplot des colonnes sélectionnées
                        ax.boxplot([df_visu[c].dropna() for c in selected])

                        #  noms des colonnes sur l’axe X
                        ax.set_xticklabels(selected, rotation=45)

                        # affichage
                        st.pyplot(fig)

            # ======================
            # SCATTER
            # ======================
            st.title("ScatterPlot")
            # Vérifier qu’il y a au moins 2 colonnes numériques
            if len(cols) >= 2:
                x = st.selectbox("X", cols)  # choix axe X
                y = st.selectbox("Y", cols)  # choix axe Y

                col1, col2, col3 = st.columns([1,2,1])  # centrage

                with col2:
                    # création figure
                    fig, ax = plt.subplots(figsize=(4, 3))

                    #  scatter plot
                    ax.scatter(df_visu[x], df_visu[y])

                    # affichage
                    st.pyplot(fig)

# ==============================
# VOLET 2 : CLUSTERING
# ==============================
elif st.session_state["page"] == "volet2":

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Prétraitement"):
            st.session_state["page"] = "volet1"
            st.rerun()

    with col2:
        if st.button("🏠 Accueil"):
            st.session_state["page"] = "home"
            st.rerun()
    st.title("Volet 2 : Clustering")

    # utiliser les données préparées
    if "df_norm" in st.session_state:
        df = st.session_state["df_norm"]
        
    elif "df_clean" in st.session_state:
        df = st.session_state["df_clean"]
        st.info("Données nettoyées utilisées")
    else:
        st.warning("⚠️ Faites d'abord le prétraitement")
        st.stop()

    X = preparer_clustering(df)

    st.subheader("Données utilisées pour le clustering")
    st.dataframe(X)



    # choix de k
    # k = st.slider("Choisir le nombre de clusters (k)", 2, 10, 3)
    k = st.number_input("Choisir le nombre de clusters (k)", min_value=2, max_value=10, value=3)
    st.subheader("Paramètres DBSCAN")

    eps = st.slider("eps (distance)", 0.1, 5.0, 0.5)
    min_samples = st.slider("min_samples", 2, 10, 5)   
    # =========================
    # ELBOW
    # =========================
    st.subheader("Méthode Elbow")

    if st.button("Afficher la courbe Elbow"):
            inertias = []
            K_range = range(1, 11)

            for k_val in K_range:
                model = KMeans(n_clusters=k_val, random_state=42)
                model.fit(X)
                inertias.append(model.inertia_)

            import numpy as np

            x = np.array(list(K_range))
            y = np.array(inertias)

            line = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
            line_y = np.polyval(line, x)

            distances = np.abs(y - line_y)

            k_optimal = int(x[np.argmax(distances)])

                # 🔥 AFFICHER MESSAGE
            st.success(f"✅ k optimal ≈ {k_optimal}")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(K_range, inertias, marker='o')
                ax.set_title("Méthode Elbow")
                ax.set_xlabel("k")
                ax.set_ylabel("Inertia")
                st.pyplot(fig)




    # =========================
        # K-MEANS & K-MEDOIDS
    # =========================
    st.subheader("Clustering : K-Means & K-Medoids")

    col1, col2 = st.columns(2)

    with col1:
            if st.button("Appliquer K-Means"):
                X_kmeans, centers_kmeans = appliquer_kmeans(X, k)
                st.session_state["kmeans"] = (X_kmeans, centers_kmeans)

    with col2:
            if st.button("Appliquer K-Medoids"):
                X_kmedoids, centers_kmedoids = appliquer_kmedoids(X, k)
                st.session_state["kmedoids"] = (X_kmedoids, centers_kmedoids)


    # AFFICHAGE
    col1, col2 = st.columns(2)

    if "kmeans" in st.session_state:
            X_kmeans, centers_kmeans = st.session_state["kmeans"]

            with col1:
                st.subheader("Résultat K-Means")
                # st.dataframe(X_kmeans)
                st.write(X_kmeans["Cluster"])
                 # PCA
                st.write("### PCA - K-Means")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                fig, ax = plt.subplots()
                # points
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_kmeans["Cluster"])
               

               # 🔥 centres (croix)
                centers_pca = pca.transform(centers_kmeans)

                ax.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    marker='X',
                    s=200,
                    c='black',
                    label='Centres'
                )
                ax.set_title("K-Means (PCA)")
                st.pyplot(fig)


    if "kmedoids" in st.session_state:
            X_kmedoids, centers_kmedoids = st.session_state["kmedoids"]
            with col2:
                st.subheader("Résultat K-Medoids")
                # st.dataframe(X_kmedoids)
                st.write(X_kmedoids["Cluster"])
                st.write("### PCA - K-Medoids")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                fig, ax = plt.subplots()
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_kmedoids["Cluster"])
                
                # 🔥 centres (médoïdes)
                centers_pca = pca.transform(centers_kmedoids)

                ax.scatter(
                    centers_pca[:, 0],
                    centers_pca[:, 1],
                    marker='X',
                    s=200,
                    c='red',
                    label='Medoids'
                )

                ax.set_title("K-Medoids (PCA)")
                st.pyplot(fig)

    # =========================
    # AGNES & DIANA
    # =========================
    st.subheader("Clustering hiérarchique : AGNES & DIANA")

    col3, col4 = st.columns(2)

    with col3:
            if st.button("Appliquer AGNES"):
                X_agnes, _ = appliquer_agnes(X, k)
                st.session_state["agnes"] = (X_agnes, None)

    with col4:
            if st.button("Appliquer DIANA"):
                X_diana, _ = appliquer_diana(X, k)
                st.session_state["diana"] = (X_diana, None)


        # AFFICHAGE
    col3, col4 = st.columns(2)

    if "agnes" in st.session_state:
            X_agnes, _ = st.session_state["agnes"]
            with col3:
                st.subheader("Résultat AGNES")
                # st.dataframe(X_agnes.sort_values("Cluster"))
                st.write(X_agnes["Cluster"])
                st.write("### PCA - AGNES")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                fig, ax = plt.subplots()
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_agnes["Cluster"])
                ax.set_title("AGNES (PCA)")
                st.pyplot(fig)
    if "diana" in st.session_state:
            X_diana, _ = st.session_state["diana"]
            with col4:
                st.subheader("Résultat DIANA")
                # st.dataframe(X_diana)
                st.write(X_diana["Cluster"])
                st.write("### PCA - DIANA")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                fig, ax = plt.subplots()
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_diana["Cluster"])
                ax.set_title("DIANA (PCA)")
                st.pyplot(fig)


    col5, col6 = st.columns(2)
     
    with col5:
      if st.button("Appliquer DBSCAN"):
        X_dbscan, _ = appliquer_dbscan(X, eps, min_samples)
        st.session_state["dbscan"] = (X_dbscan, None)
       

    #    Affichage
    col5, col6 = st.columns(2)

    if "dbscan" in st.session_state:
      X_dbscan, _ = st.session_state["dbscan"]
      with col5:
        st.subheader("Résultat DBSCAN")
        # st.dataframe(X_dbscan)
        st.write(X_dbscan["Cluster"])
        st.write("### PCA - DBSCAN")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots()
        # points
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_dbscan["Cluster"])
       
       #  ajouter cercles (exemple pour 10 points)
        #  cercle pour chaque point
        for i in range(len(X_pca)):
            circle = patches.Circle(
                (X_pca[i, 0], X_pca[i, 1]),
                eps,              # rayon
                fill=False,
                edgecolor='gray',
                linewidth=0.5,
                alpha=0.3
            )
            ax.add_patch(circle)
       
        ax.set_title("DBSCAN (PCA)")
        st.pyplot(fig)


    # =========================
    # SILHOUETTE
    # =========================
    st.subheader("Évaluation : Score de Silhouette")

    if st.button("Calculer le score de Silhouette"):

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)

        # =========================
        # K-MEANS
        # =========================
        if "kmeans" in st.session_state:
            X_kmeans, _ = st.session_state["kmeans"]
            score_kmeans = silhouette_score(X, X_kmeans["Cluster"])
            with col1:
                st.success(f"Silhouette K-Means : {score_kmeans:.3f}")

        # =========================
        # K-MEDOIDS
        # =========================
        if "kmedoids" in st.session_state:
            X_kmedoids, _ = st.session_state["kmedoids"]
            score_kmedoids = silhouette_score(X, X_kmedoids["Cluster"])
            with col2:
                st.success(f"Silhouette K-Medoids : {score_kmedoids:.3f}")

        # =========================
        # AGNES
        # =========================
        if "agnes" in st.session_state:
            X_agnes, _ = st.session_state["agnes"]
            score_agnes = silhouette_score(X, X_agnes["Cluster"])
            with col3:
                st.success(f"Silhouette AGNES : {score_agnes:.3f}")

        # =========================
        # DIANA
        # =========================
        if "diana" in st.session_state:
            X_diana, _ = st.session_state["diana"]
            score_diana = silhouette_score(X, X_diana["Cluster"])
            with col4:
                st.success(f"Silhouette DIANA : {score_diana:.3f}")

        # =========================
        # DBSCAN (sans bruit)
        # =========================
        if "dbscan" in st.session_state:
            X_dbscan, _ = st.session_state["dbscan"]

            labels = X_dbscan["Cluster"]

            # enlever bruit (-1)
            mask = labels != -1

            if len(set(labels[mask])) > 1:
                score_dbscan = silhouette_score(X[mask], labels[mask])
                with col5:
                    st.success(f"Silhouette DBSCAN : {score_dbscan:.3f}")
            else:
                with col5:
                    st.warning("DBSCAN : silhouette non calculable (clusters insuffisants)")


    # =========================
    # INERTIE
    # =========================
    st.subheader("Évaluation : Inertie")

    col1, col2 = st.columns(2)

    # =========================
    # K-MEANS
    # =========================
    if "kmeans" in st.session_state:
        X_kmeans, centers_kmeans = st.session_state["kmeans"]

        # recalcul du modèle pour récupérer inertia
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=len(set(X_kmeans["Cluster"])), random_state=42)
        model.fit(X)

        inertia_kmeans = model.inertia_

        with col1:
            st.info(f"Inertie K-Means : {inertia_kmeans:.3f}")

    # =========================
    # K-MEDOIDS 
    # =========================
    if "kmedoids" in st.session_state:
        X_kmedoids, centers_kmedoids = st.session_state["kmedoids"]

        # approximation inertie = somme distances aux médoids
        import numpy as np

        inertia_kmedoids = 0
        for i in range(len(X)):
            cluster = X_kmedoids["Cluster"].iloc[i]
            medoid = centers_kmedoids[cluster]
            inertia_kmedoids += np.linalg.norm(X.iloc[i] - medoid) ** 2

        with col2:
            st.info(f"Inertie K-Medoids : {inertia_kmedoids:.3f}")


# ==============================
# VOLET 3 : CLASSIFICATION
# ==============================
elif st.session_state["page"] == "volet3":

    # =========================
    # NAVIGATION
    # =========================
    col1, col2 = st.columns(2)

    with col1:

        if st.button("⬅️ Prétraitement"):

            st.session_state["page"] = "volet1"
            st.rerun()

    with col2:

        if st.button("🏠 Accueil"):

            st.session_state["page"] = "home"
            st.rerun()

    st.title("Volet 3 : Classification")

    # =========================
    # RECUPERATION DES DONNEES
    # =========================
    if "df_norm" in st.session_state:

        df_class = st.session_state["df_norm"].copy()

        

    elif "df_clean" in st.session_state:

        df_class = st.session_state["df_clean"].copy()


    else:

        st.warning("Effectuez d'abord le prétraitement.")
        st.stop()

    if df_class.empty:

        st.warning("Le dataset est vide.")
        st.stop()

    # =========================
    # DONNEES A CLASSIFIER
    # =========================
    st.subheader("Données à classifier")

    st.dataframe(df_class)

    # =========================
    # VARIABLE CIBLE Y
    # =========================
    cible = st.selectbox(
        "Variable cible Y (Classe)",
        df_class.columns.tolist(),
        index=len(df_class.columns) - 1,
        key="classification_cible"
    )

    # =========================
    # ATTRIBUTS EXPLICATIFS X
    # =========================
    colonnes = [
        col for col in df_class.columns
        if col != cible
    ]

    attributs = st.multiselect(
        "Attributs explicatifs X",
        colonnes,
        default=colonnes,
        key=f"classification_x_{cible}"
    )

    # =========================
    # POURCENTAGE DE TEST
    # =========================
    pourcentage_test = st.slider(
        "Pourcentage des données de test (%)",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Entraînement",
            f"{100 - pourcentage_test}%"
        )

    with col2:

        st.metric(
            "Test",
            f"{pourcentage_test}%"
        )

    st.markdown("---")

    # =========================
    # CHOIX DE L'ALGORITHME
    # =========================
    st.subheader("Algorithmes de classification")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "Appliquer KNN",
            use_container_width=True
        ):

            st.session_state["algo_class"] = "KNN"

    with col2:

        if st.button(
            "Naïve Bayes",
            use_container_width=True
        ):

            st.session_state["algo_class"] = "NB"

    with col3:

        if st.button(
            "Arbres",
            use_container_width=True
        ):

            st.session_state["algo_class"] = "Arbres"

    
    # Récupération de l'algorithme sélectionné
    algo = st.session_state.get("algo_class", None)
    # =========================
    # KNN
    # =========================
    if algo == "KNN":

        st.markdown("---")

        st.header("K-Nearest Neighbors (KNN)")

        if not attributs:

            st.warning("Sélectionnez les attributs X.")

        else:

            # =========================
            # DETECTION DES TYPES
            # =========================
            num_auto, cat_auto = detecter_types_attributs(
                df_class,
                attributs
            )

            # Possibilité de corriger la détection
            categoriels = st.multiselect(
                "Attributs qualitatifs ou binaires",
                attributs,
                default=cat_auto,
                key=f"knn_types_{cible}_{tuple(attributs)}"
            )

            numeriques = [
                col for col in attributs
                if col not in categoriels
            ]

            # =========================
            # DISTANCE AUTOMATIQUE
            # =========================
            distance = choisir_distance(
                numeriques,
                categoriels
            )

            st.info(
                f"Distance sélectionnée automatiquement : "
                f"{distance}"
            )

            # =========================
            # PREPARATION TRAIN / TEST
            # =========================
            try:

                X, y = preparer_classification(
                    df_class,
                    attributs,
                    cible
                )

                X_train, X_test, y_train, y_test = (
                    diviser_donnees(
                        X,
                        y,
                        test_size=pourcentage_test / 100
                    )
                )

            except ValueError as e:

                st.error(str(e))
                st.stop()

            # =========================
            # NOMBRE DE VOISINS
            # =========================
            k = st.number_input(
                "Nombre de voisins K",
                min_value=1,
                max_value=len(X_train),
                value=min(3, len(X_train)),
                step=1
            )

            # =========================
            # EXECUTION DE KNN
            # =========================
            if st.button(
                "Lancer la classification KNN",
                type="primary"
            ):

                try:

                    modele, y_pred, distance = appliquer_knn(
                        X_train,
                        X_test,
                        y_train,
                        int(k),
                        numeriques,
                        categoriels
                    )

                    # =========================
                    # EVALUATION
                    # =========================
                    scores = evaluer_classification(
                        y_test,
                        y_pred
                    )

                    matrice = get_matrice_confusion(
                        y_test,
                        y_pred,
                        modele.classes_
                    )

                    rapport = get_rapport_classification(
                        y_test,
                        y_pred
                    )

                    predictions = get_predictions(
                        X_test,
                        y_test,
                        y_pred
                    )

                   

                    # =========================
                    # RESULTATS
                    # =========================
                    st.header("Résultats de KNN")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Données d'entraînement",
                            len(X_train)
                        )

                    with col2:

                        st.metric(
                            "Données de test",
                            len(X_test)
                        )

                    # =========================
                    # METRIQUES
                    # =========================
                    st.subheader("Performances")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Accuracy",
                            f"{scores['Accuracy'] * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Precision",
                            f"{scores['Precision'] * 100:.2f}%"
                        )

                    with col3:

                        st.metric(
                            "Recall",
                            f"{scores['Recall'] * 100:.2f}%"
                        )

                    with col4:

                        st.metric(
                            "F1-score",
                            f"{scores['F1-score'] * 100:.2f}%"
                        )

                    # =========================
                    # MATRICE DE CONFUSION
                    # =========================
                    st.subheader("Matrice de confusion")

                    st.dataframe(matrice)

                    fig, ax = plt.subplots(figsize=(5, 4))

                    image = ax.imshow(
                        matrice.values,
                        cmap="Purples"
                    )

                    ax.set_xticks(
                        range(len(matrice.columns))
                    )

                    ax.set_yticks(
                        range(len(matrice.index))
                    )

                    ax.set_xticklabels(matrice.columns)
                    ax.set_yticklabels(matrice.index)

                    ax.set_xlabel("Classes prédites")
                    ax.set_ylabel("Classes réelles")

                    for i in range(len(matrice.index)):

                        for j in range(len(matrice.columns)):

                            ax.text(
                                j,
                                i,
                                matrice.iloc[i, j],
                                ha="center",
                                va="center"
                            )

                    fig.colorbar(image)

                    st.pyplot(fig)
                    plt.close(fig)

                    # =========================
                    # RAPPORT
                    # =========================
                    st.subheader(
                        "Rapport de classification"
                    )

                    st.dataframe(rapport.round(4))

                    # =========================
                    # PREDICTIONS
                    # =========================
                    st.subheader("Prédictions")

                    st.dataframe(predictions)

                    # =========================
                    # TELECHARGEMENT
                    # =========================
                    st.download_button(
                        "Télécharger les prédictions",
                        data=predictions.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="predictions_knn.csv",
                        mime="text/csv"
                    )

                except ValueError as e:

                    st.error(str(e))

    # =========================
    # NAIVE BAYES
    # =========================
    elif algo == "NB":

        st.header("Naïve Bayes")

        st.info(
            "Cet algorithme sera ajouté prochainement."
        )

    # =========================
    # ARBRES DE DECISION
    # =========================
    elif algo == "Arbres":
        if not attributs:
            st.warning("Sélectionnez les attributs X.")
        else:
            interface_arbres(df_class, cible, attributs, pourcentage_test / 100)
