import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
Z = 0.69


def entropie(y):
    """Info(S) = - somme(p_i * log2(p_i))"""
    _, effectifs = np.unique(y, return_counts=True)
    p = effectifs / len(y)
    return -np.sum(p * np.log2(p))

def gain(y, y_gauche, y_droite):
    """Gain(S, A) = Info(S) - somme(|Sv|/|S| * Info(Sv))"""
    n = len(y)
    info_apres = (len(y_gauche) / n) * entropie(y_gauche) + (len(y_droite) / n) * entropie(y_droite)
    return entropie(y) - info_apres

def split_info(y_gauche, y_droite):
    """SplitInfo(S, A) = - somme(|Sv|/|S| * log2(|Sv|/|S|))"""
    n = len(y_gauche) + len(y_droite)
    p = np.array([len(y_gauche), len(y_droite)]) / n
    return -np.sum(p * np.log2(p))

def gain_ratio(y, y_gauche, y_droite):
    """GainRatio(S, A) = Gain(S, A) / SplitInfo(S, A)"""
    si = split_info(y_gauche, y_droite)
    return gain(y, y_gauche, y_droite) / si if si > 0 else 0.0

def meilleur_seuil(x, y, min_obj=2):
    """Teste tous les seuils possibles de l'attribut x.
    Chaque branche doit contenir au moins min_obj exemples (comme C4.5, option -m 2).
    Retourne (seuil, gain, gain_ratio) du seuil qui maximise le gain."""
    valeurs = np.unique(x)
    meilleur = (None, -1.0, 0.0)
    for i in range(len(valeurs) - 1):
        seuil = (valeurs[i] + valeurs[i + 1]) / 2
        y_g, y_d = y[x <= seuil], y[x > seuil]
        if len(y_g) < min_obj or len(y_d) < min_obj:
            continue
        g = gain(y, y_g, y_d)
        if g > meilleur[1]:
            meilleur = (seuil, g, gain_ratio(y, y_g, y_d))
    return meilleur

def classe_majoritaire(y):
    valeurs, effectifs = np.unique(y, return_counts=True)
    return valeurs[np.argmax(effectifs)]

def creer_feuille(y):
    c = classe_majoritaire(y)
    return {'feuille': True, 'classe': c, 'n': len(y), 'erreurs': int(np.sum(y != c))}

def construire_arbre(X, y, noms, min_samples=2):
    # Cas d'arrêt : classe pure ou trop peu d'exemples
    if len(np.unique(y)) == 1 or len(y) < min_samples:
        return creer_feuille(y)

    # Meilleur seuil de chaque attribut
    candidats = []
    for j in range(X.shape[1]):
        seuil, g, gr = meilleur_seuil(X[:, j], y)
        if seuil is not None:
            candidats.append((j, seuil, g, gr))
    if not candidats:
        return creer_feuille(y)

    # Heuristique C4.5 : attributs dont le gain >= gain moyen, puis meilleur gain ratio
    gain_moyen = np.mean([c[2] for c in candidats])
    retenus = [c for c in candidats if c[2] >= gain_moyen - 1e-12]
    j, seuil, g, gr = max(retenus, key=lambda c: c[3])

    if g <= 0:
        return creer_feuille(y)

    masque = X[:, j] <= seuil
    noeud = creer_feuille(y)          # garde classe, n, erreurs (utile pour l'élagage)
    noeud.update({
        'feuille': False,
        'attribut': noms[j], 'indice': j, 'seuil': seuil,
        'gauche': construire_arbre(X[masque], y[masque], noms, min_samples),
        'droite': construire_arbre(X[~masque], y[~masque], noms, min_samples),
    })
    return noeud

def afficher_arbre(noeud, indent=''):
    if noeud['feuille']:
        print(f"{indent}-> classe = {noeud['classe']}  (n={noeud['n']}, erreurs={noeud['erreurs']})")
        return
    print(f"{indent}si {noeud['attribut']} <= {noeud['seuil']:.2f} :")
    afficher_arbre(noeud['gauche'], indent + '    ')
    print(f"{indent}sinon ({noeud['attribut']} > {noeud['seuil']:.2f}) :")
    afficher_arbre(noeud['droite'], indent + '    ')

def compter_feuilles(noeud):
    if noeud['feuille']:
        return 1
    return compter_feuilles(noeud['gauche']) + compter_feuilles(noeud['droite'])

def profondeur(noeud):
    if noeud['feuille']:
        return 0
    return 1 + max(profondeur(noeud['gauche']), profondeur(noeud['droite']))

def erreur_pessimiste(n, e, z=Z):
    f = e / n
    u = (f + z**2 / (2 * n) + z * np.sqrt(f / n - f**2 / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return n * u

def elaguer(noeud):
    """Élague l'arbre en place. Retourne l'erreur prédite du nœud (après élagage)."""
    err_feuille = erreur_pessimiste(noeud['n'], noeud['erreurs'])
    if noeud['feuille']:
        return err_feuille

    err_sous_arbre = elaguer(noeud['gauche']) + elaguer(noeud['droite'])

    if err_feuille <= err_sous_arbre:
        # On remplace le sous-arbre par une feuille
        noeud['feuille'] = True
        for cle in ('gauche', 'droite', 'attribut', 'indice', 'seuil'):
            noeud.pop(cle, None)
        return err_feuille
    return err_sous_arbre

def calculer_positions(noeud, niveau, compteur, pos):
    """Place les feuilles de gauche à droite, chaque parent au milieu de ses enfants."""
    if noeud['feuille']:
        pos[id(noeud)] = (compteur[0], -niveau)
        compteur[0] += 1
    else:
        calculer_positions(noeud['gauche'], niveau + 1, compteur, pos)
        calculer_positions(noeud['droite'], niveau + 1, compteur, pos)
        xg = pos[id(noeud['gauche'])][0]
        xd = pos[id(noeud['droite'])][0]
        pos[id(noeud)] = ((xg + xd) / 2, -niveau)

def dessiner_arbre(arbre, titre, classes):
    pos = {}
    calculer_positions(arbre, 0, [0], pos)
    palette = ['#a8e6a1', '#f9d98c', '#f4a6a6', '#b8c7f0', '#e6c2f0']
    couleurs = {cl: palette[i % len(palette)] for i, cl in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(12, 6))

    def tracer(noeud):
        x, y = pos[id(noeud)]
        if noeud['feuille']:
            texte = f"{noeud['classe']}\nn={noeud['n']}, err={noeud['erreurs']}"
            ax.text(x, y, texte, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', fc=couleurs.get(noeud['classe'], 'white'), ec='black'))
        else:
            texte = f"{noeud['attribut']}\n<= {noeud['seuil']:.2f} ?"
            ax.text(x, y, texte, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', fc='#d0e4f7', ec='black'))
            for enfant, etiquette in ((noeud['gauche'], 'oui'), (noeud['droite'], 'non')):
                xe, ye = pos[id(enfant)]
                ax.plot([x, xe], [y - 0.12, ye + 0.12], color='gray', zorder=0)
                ax.text((x + xe) / 2, (y + ye) / 2, etiquette, fontsize=8, color='darkred')
                tracer(enfant)

    tracer(arbre)
    ax.set_title(titre)
    ax.axis('off')
    return fig

def predire_un(noeud, x):
    while not noeud['feuille']:
        noeud = noeud['gauche'] if x[noeud['indice']] <= noeud['seuil'] else noeud['droite']
    return noeud['classe']

def predire(arbre, X):
    return np.array([predire_un(arbre, x) for x in X])

def gini(y):
    # Gini(S) = 1 - somme(p_i^2)
    _, effectifs = np.unique(y, return_counts=True)
    p = effectifs / len(y)
    return 1 - np.sum(p ** 2)

def gini_apres_division(y_gauche, y_droite):
    # Gini_A(S) = somme(|Sv|/|S| * Gini(Sv))
    n = len(y_gauche) + len(y_droite)
    return (len(y_gauche) / n) * gini(y_gauche) + (len(y_droite) / n) * gini(y_droite)

def meilleure_division_attribut(x, y):
    # Teste tous les seuils de l'attribut x et retourne (seuil, gini_apres_division)
    valeurs = np.unique(x)
    meilleur = (None, np.inf)
    for i in range(len(valeurs) - 1):
        seuil = (valeurs[i] + valeurs[i + 1]) / 2
        g = gini_apres_division(y[x <= seuil], y[x > seuil])
        if g < meilleur[1]:
            meilleur = (seuil, g)
    return meilleur

def construire_cart(X, y, noms, min_samples=2):
    # Cas d'arrêt : classe pure ou trop peu d'exemples
    if len(np.unique(y)) == 1 or len(y) < min_samples:
        return creer_feuille(y)

    # Meilleure division parmi tous les attributs
    meilleur = None   # (indice, seuil, gini_apres)
    for j in range(X.shape[1]):
        seuil, g = meilleure_division_attribut(X[:, j], y)
        if seuil is not None and (meilleur is None or g < meilleur[2]):
            meilleur = (j, seuil, g)

    # Pas de division possible, ou aucune réduction de Gini
    if meilleur is None or gini(y) - meilleur[2] <= 1e-12:
        return creer_feuille(y)

    j, seuil, _ = meilleur
    masque = X[:, j] <= seuil
    noeud = creer_feuille(y)
    noeud.update({
        'feuille': False,
        'attribut': noms[j], 'indice': j, 'seuil': seuil,
        'gauche': construire_cart(X[masque], y[masque], noms, min_samples),
        'droite': construire_cart(X[~masque], y[~masque], noms, min_samples),
    })
    return noeud

def infos_sous_arbre(noeud):
    # Retourne (nombre d'erreurs sur les feuilles du sous-arbre, nombre de feuilles)
    if noeud['feuille']:
        return noeud['erreurs'], 1
    err_g, nb_g = infos_sous_arbre(noeud['gauche'])
    err_d, nb_d = infos_sous_arbre(noeud['droite'])
    return err_g + err_d, nb_g + nb_d

def lien_faible(noeud, N):
    # Cherche le nœud interne de plus petit g(t) = (R(t) - R(Tt)) / (|Tt| - 1)
    if noeud['feuille']:
        return np.inf, None
    err_sous_arbre, nb_feuilles = infos_sous_arbre(noeud)
    g = (noeud['erreurs'] - err_sous_arbre) / N / (nb_feuilles - 1)
    meilleur = (g, noeud)
    for enfant in (noeud['gauche'], noeud['droite']):
        candidat = lien_faible(enfant, N)
        if candidat[0] < meilleur[0]:
            meilleur = candidat
    return meilleur

def transformer_en_feuille(noeud):
    noeud['feuille'] = True
    for cle in ('gauche', 'droite', 'attribut', 'indice', 'seuil'):
        noeud.pop(cle, None)

def elaguer_alpha(arbre, alpha):
    # Supprime les liens faibles tant que g(t) <= alpha (modifie l'arbre en place)
    N = arbre['n']
    while True:
        g, noeud = lien_faible(arbre, N)
        if noeud is None or g > alpha + 1e-12:
            break
        transformer_en_feuille(noeud)

def suite_alphas(arbre):
    # Suite croissante des valeurs de alpha obtenues en élaguant jusqu'à la racine
    arbre = copy.deepcopy(arbre)
    N = arbre['n']
    alphas = [0.0]
    while True:
        g, noeud = lien_faible(arbre, N)
        if noeud is None:
            break
        alphas.append(max(g, alphas[-1]))
        transformer_en_feuille(noeud)
    return np.unique(alphas)

def validation_croisee(X, y, noms, alphas, k=5):
    rng = np.random.default_rng()
    indices = rng.permutation(len(y))
    plis = np.array_split(indices, k)
    scores = np.zeros((len(alphas), k))

    for i, idx_val in enumerate(plis):
        idx_app = np.concatenate([plis[j] for j in range(k) if j != i])
        arbre = construire_cart(X[idx_app], y[idx_app], noms)
        for a_i, a in enumerate(alphas):
            t = copy.deepcopy(arbre)
            elaguer_alpha(t, a)
            scores[a_i, i] = np.mean(predire(t, X[idx_val]) == y[idx_val])
    return scores.mean(axis=1)