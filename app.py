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
# ==============================
# LOAD CSS
# ==============================
def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


st.set_page_config(layout="wide")



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
            <p><b>Réalisé par :</b> Silya Sadi</p>
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

            choix = st.selectbox("Méthode", [
                "Aucune",
                "Supprimer les lignes avec valeurs manquantes",
                "Remplir par la moyenne",
                "Remplir par la médiane",
                "Remplir par le mode"
            ])

            if st.button("Appliquer nettoyage"):
               df_clean = nettoyer_donnees(df, choix)
               st.session_state["df_clean"] = df_clean

               st.subheader("Données après nettoyage")
               st.dataframe(df_clean)

        # ======================
        # NORMALISATION
        # ======================
        with tab3:
            if "df_clean" in st.session_state:
                df_base = st.session_state["df_clean"]
                st.dataframe(df_base)
            else:
                # df_base = df_current
                st.warning("⚠️ Données non nettoyées")
                st.stop()
            methode = st.selectbox("Normalisation", [
                "Aucune",
                "Min-Max Scaling",
                "Standardisation"
            ])

            if st.button("Appliquer normalisation"):
                df_norm = normaliser_donnees(df_base, methode)
                st.session_state["df_norm"] = df_norm
                
                st.dataframe(df_norm)

            st.markdown("---")

            col1, col2 = st.columns(2)

            #  Clustering
            with col1:
                if "df_clean" in st.session_state or "df_norm" in st.session_state:
                    if st.button("➡️ Clustering"):
                        st.session_state["page"] = "volet2"
                        st.rerun()

            #  Classification
            with col2:
                if "df_clean" in st.session_state or "df_norm" in st.session_state:
                    st.button("➡️ Classification")
                        

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

    if st.button("⬅️ Retour au Prétraitement"):
        st.session_state["page"] = "volet1"
        st.rerun()

    st.title("Classification")

    st.info("🚧 Cette partie est en cours de développement...")