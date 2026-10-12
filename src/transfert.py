"""
Apprentissage par transfert : représentations gelées contre fine-tuning complet.

D'où ça vient
-------------
* TP n°3 de M1 : *« get the embeddings of the amazon_polarity dataset, use these
  representations to train a logistic regression »* — c'est le régime des
  **traits gelés** : on extrait les représentations une fois, on entraîne un
  classifieur linéaire par-dessus.
* TP n°1 de M2 (Marie Candito) : fine-tuning de FlauBERT — on met à jour **tout**
  le modèle.

Entre les deux, il y a un continuum, et c'est lui que je mesure ici.

Le problème que je dois résoudre pour que ce module tourne
-----------------------------------------------------------
Je ne peux pas embarquer FlauBERT dans ce dépôt. Mais je peux **simuler
fidèlement** la situation qui compte : un encodeur pré-entraîné qui produit des
représentations utiles mais imparfaitement adaptées à la tâche.

Je construis donc un encodeur « pré-entraîné » sur une tâche **voisine mais
différente** — prédire l'hypersense (6 classes grossières) — puis je l'adapte à
la tâche cible — prédire le supersense (12 classes fines).

C'est exactement la situation du transfert réel : les représentations contiennent
déjà l'essentiel, mais pas les distinctions fines. Et cela me permet de mesurer
les trois régimes sur des chiffres reproductibles.
"""

from __future__ import annotations

import numpy as np

from .donnees import HIERARCHIE, HYPERSENSES, SUPERSENSES, decouper
from .traits import (
    ExtracteurTraits,
    f1_macro,
    reference_majoritaire,
    softmax,
)


# --------------------------------------------------------------------------- #
# 1. Un encodeur « pré-entraîné »
# --------------------------------------------------------------------------- #


class Encodeur:
    """Un petit réseau qui transforme un vecteur de traits en représentation dense.

    Architecture : ``traits -> Linéaire -> tanh -> Linéaire -> tanh``.

    Ce n'est évidemment pas BERT. Mais il a les deux propriétés qui comptent
    pour la démonstration :

      * il est **pré-entraîné** sur une tâche voisine, donc ses représentations
        portent une information utile mais générique ;
      * il a beaucoup de paramètres par rapport au corpus cible, donc le
        fine-tuning complet peut surapprendre — ce qui est exactement le
        problème que LoRA cherche à résoudre.
    """

    def __init__(self, dimension_entree: int, dimension_cachee: int = 64,
                 dimension_sortie: int = 32, graine: int = 0):
        rng = np.random.default_rng(graine)
        self.params = {
            "W1": rng.normal(0, np.sqrt(1.0 / dimension_entree),
                             size=(dimension_entree, dimension_cachee)),
            "b1": np.zeros(dimension_cachee),
            "W2": rng.normal(0, np.sqrt(1.0 / dimension_cachee),
                             size=(dimension_cachee, dimension_sortie)),
            "b2": np.zeros(dimension_sortie),
        }
        self.dimension_sortie = dimension_sortie

    def encoder(self, X: np.ndarray) -> np.ndarray:
        """Représentation dense. Forme ``(batch, dimension_sortie)``."""
        cachee = np.tanh(X @ self.params["W1"] + self.params["b1"])
        return np.tanh(cachee @ self.params["W2"] + self.params["b2"])

    def encoder_avec_cache(self, X: np.ndarray):
        pre1 = X @ self.params["W1"] + self.params["b1"]
        cachee = np.tanh(pre1)
        pre2 = cachee @ self.params["W2"] + self.params["b2"]
        sortie = np.tanh(pre2)
        return sortie, (X, cachee, sortie)

    def gradients(self, cache, grad_sortie: np.ndarray) -> dict[str, np.ndarray]:
        """Rétropropagation à travers l'encodeur.

        C'est ce qu'un fine-tuning complet calcule et qu'un entraînement à
        représentations gelées **ne calcule pas** — d'où son coût bien moindre
        en mémoire et en temps.
        """
        X, cachee, sortie = cache
        grad_pre2 = grad_sortie * (1 - sortie ** 2)
        grads = {
            "W2": cachee.T @ grad_pre2,
            "b2": grad_pre2.sum(axis=0),
        }
        grad_cachee = grad_pre2 @ self.params["W2"].T
        grad_pre1 = grad_cachee * (1 - cachee ** 2)
        grads["W1"] = X.T @ grad_pre1
        grads["b1"] = grad_pre1.sum(axis=0)
        return grads

    @property
    def nb_parametres(self) -> int:
        return sum(p.size for p in self.params.values())

    def copie(self) -> Encodeur:
        """Copie profonde — indispensable pour comparer plusieurs régimes."""
        clone = Encodeur.__new__(Encodeur)
        clone.params = {n: p.copy() for n, p in self.params.items()}
        clone.dimension_sortie = self.dimension_sortie
        return clone


class TeteClassification:
    """Une simple couche linéaire : représentation -> scores de classes."""

    def __init__(self, dimension: int, nb_classes: int, graine: int = 0):
        rng = np.random.default_rng(graine)
        self.params = {
            "W": rng.normal(0, np.sqrt(1.0 / dimension), size=(dimension, nb_classes)),
            "b": np.zeros(nb_classes),
        }

    def scores(self, H: np.ndarray) -> np.ndarray:
        return H @ self.params["W"] + self.params["b"]

    @property
    def nb_parametres(self) -> int:
        return sum(p.size for p in self.params.values())


# --------------------------------------------------------------------------- #
# 2. Entraînement
# --------------------------------------------------------------------------- #


def perte_et_gradient_scores(scores: np.ndarray, y: np.ndarray):
    """Entropie croisée et son gradient — le résultat vu au projet 1."""
    n = scores.shape[0]
    probas = softmax(scores)
    perte = float(-np.mean(np.log(probas[np.arange(n), y] + 1e-12)))
    grad = probas.copy()
    grad[np.arange(n), y] -= 1.0
    return perte, grad / n


def pre_entrainer(X: np.ndarray, y_grossier: np.ndarray, dimension_entree: int,
                  nb_epoques: int = 400, lr: float = 0.5,
                  graine: int = 0) -> Encodeur:
    """Entraîne l'encodeur sur la tâche **grossière** (hypersense).

    C'est le « pré-entraînement ». La tâche est voisine de la tâche cible mais
    plus facile : distinguer `animé` de `inanimé` demande moins de finesse que
    distinguer `personne` de `animal`.

    C'est exactement le rapport entre le pré-entraînement d'un modèle de langue
    et une tâche en aval : la première apprend des régularités générales, la
    seconde demande des distinctions spécifiques.
    """
    encodeur = Encodeur(dimension_entree, graine=graine)
    tete = TeteClassification(encodeur.dimension_sortie, len(HYPERSENSES),
                              graine=graine)

    for _ in range(nb_epoques):
        H, cache = encodeur.encoder_avec_cache(X)
        scores = tete.scores(H)
        _, grad_scores = perte_et_gradient_scores(scores, y_grossier)

        grad_W = H.T @ grad_scores
        grad_b = grad_scores.sum(axis=0)
        grad_H = grad_scores @ tete.params["W"].T
        grads_encodeur = encodeur.gradients(cache, grad_H)

        tete.params["W"] -= lr * grad_W
        tete.params["b"] -= lr * grad_b
        for nom, gradient in grads_encodeur.items():
            encodeur.params[nom] -= lr * gradient

    return encodeur


def entrainer_tete_gelee(encodeur: Encodeur, X_tr: np.ndarray, y_tr: np.ndarray,
                         nb_classes: int, nb_epoques: int = 400, lr: float = 0.5,
                         regularisation: float = 1e-3, graine: int = 0):
    """Régime 1 : **représentations gelées** (*linear probing*).

    On calcule les représentations **une seule fois**, puis on entraîne une
    couche linéaire par-dessus. L'encodeur ne bouge pas.

    Avantages, et ils sont considérables :
      * on n'entraîne que ``d × k`` paramètres ;
      * les représentations peuvent être **pré-calculées et mises en cache** :
        c'est ce que je faisais dans le TP de M1 avec DistilBERT, et c'est ce
        qui rendait l'expérience faisable sur un ordinateur portable ;
      * impossible de détruire le modèle pré-entraîné par un mauvais réglage.

    Inconvénient : si la représentation ne contient pas l'information nécessaire
    à la tâche, aucune couche linéaire ne pourra l'inventer.
    """
    H = encodeur.encoder(X_tr)
    tete = TeteClassification(encodeur.dimension_sortie, nb_classes, graine=graine)
    for _ in range(nb_epoques):
        scores = tete.scores(H)
        _, grad_scores = perte_et_gradient_scores(scores, y_tr)
        tete.params["W"] -= lr * (H.T @ grad_scores
                                  + regularisation * tete.params["W"])
        tete.params["b"] -= lr * grad_scores.sum(axis=0)
    return encodeur, tete


def entrainer_complet(encodeur: Encodeur, X_tr: np.ndarray, y_tr: np.ndarray,
                      nb_classes: int, nb_epoques: int = 400, lr: float = 0.5,
                      regularisation: float = 1e-3, graine: int = 0):
    """Régime 2 : **fine-tuning complet**.

    On met à jour l'encodeur **et** la tête. C'est ce que fait
    `Trainer.train()` de Hugging Face par défaut.

    Avantage : le modèle peut réellement s'adapter à la tâche.

    Inconvénients, et c'est là que LoRA intervient :
      * on entraîne des millions (ou des milliards) de paramètres ;
      * il faut stocker les états de l'optimiseur — avec Adam, **trois fois** la
        taille du modèle en mémoire ;
      * on obtient une copie complète du modèle **par tâche** ;
      * sur peu de données, on surapprend, et on peut même détruire ce que le
        pré-entraînement avait appris (*oubli catastrophique*).
    """
    encodeur = encodeur.copie()
    tete = TeteClassification(encodeur.dimension_sortie, nb_classes, graine=graine)

    for _ in range(nb_epoques):
        H, cache = encodeur.encoder_avec_cache(X_tr)
        scores = tete.scores(H)
        _, grad_scores = perte_et_gradient_scores(scores, y_tr)

        grad_H = grad_scores @ tete.params["W"].T
        grads_encodeur = encodeur.gradients(cache, grad_H)

        tete.params["W"] -= lr * (H.T @ grad_scores
                                  + regularisation * tete.params["W"])
        tete.params["b"] -= lr * grad_scores.sum(axis=0)
        for nom, gradient in grads_encodeur.items():
            encodeur.params[nom] -= lr * (gradient
                                          + regularisation * encodeur.params[nom])

    return encodeur, tete


def entrainer_depuis_zero(X_tr: np.ndarray, y_tr: np.ndarray, nb_classes: int,
                          dimension_entree: int, nb_epoques: int = 400,
                          lr: float = 0.5, graine: int = 0):
    """Régime 3 : **aucun pré-entraînement**. Le témoin indispensable.

    Sans ce point de comparaison, on ne sait pas si le transfert apporte quelque
    chose ou si c'est simplement l'architecture qui est bonne.
    """
    encodeur = Encodeur(dimension_entree, graine=graine + 100)
    return entrainer_complet(encodeur, X_tr, y_tr, nb_classes,
                             nb_epoques=nb_epoques, lr=lr, graine=graine)


def evaluer(encodeur: Encodeur, tete: TeteClassification,
            X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predits = np.argmax(tete.scores(encodeur.encoder(X)), axis=1)
    return {
        "exactitude": float(np.mean(predits == y)),
        "f1_macro": f1_macro(y, predits, tete.params["W"].shape[1]),
    }


# --------------------------------------------------------------------------- #
# 3. Mise en place
# --------------------------------------------------------------------------- #


def preparer_transfert(graine: int = 0):
    """Prépare les données pour les deux tâches : grossière puis fine."""
    train, dev, test = decouper(graine=graine)
    extracteur = ExtracteurTraits(sac_de_mots=True, premier_mot=True)
    extracteur.ajuster([d for d, _ in train])

    fin_vers_indice = {c: i for i, c in enumerate(SUPERSENSES)}
    grossier_vers_indice = {c: i for i, c in enumerate(HYPERSENSES)}

    def convertir(exemples):
        X = extracteur.transformer([d for d, _ in exemples])
        y_fin = np.array([fin_vers_indice[c] for _, c in exemples])
        y_grossier = np.array([grossier_vers_indice[HIERARCHIE[c]]
                               for _, c in exemples])
        return X, y_fin, y_grossier

    return convertir(train), convertir(dev), convertir(test), extracteur, (train, dev, test)


def experience_multi_graines(graines: list[int]) -> dict[str, dict[str, float]]:
    """Moyenne les trois régimes sur plusieurs graines.

    Pourquoi c'est indispensable ici : mon jeu de test compte 28 exemples. **Un
    seul exemple vaut 3,6 points d'exactitude.** Un écart de 5 points entre deux
    méthodes peut donc n'être qu'un exemple qui bascule.

    C'est le défaut de mon rendu de M2 : j'y comparais LoRA et Qwen sur des
    exactitudes issues d'un seul découpage, sans écart-type, et j'en tirais des
    conclusions. Moyenner sur plusieurs graines ne rend pas l'expérience
    concluante — il faudrait bien plus de données — mais cela évite au moins de
    conclure sur du bruit.
    """
    import numpy as np

    accumulateurs: dict[str, list[float]] = {
        "représentations gelées": [], "fine-tuning complet": [], "depuis zéro": []
    }

    for graine in graines:
        (X_tr, y_tr, y_tr_gros), _, (X_te, y_te, _), _, _ = preparer_transfert(graine)
        encodeur = pre_entrainer(X_tr, y_tr_gros, X_tr.shape[1], graine=graine)

        enc, tete = entrainer_tete_gelee(encodeur, X_tr, y_tr, len(SUPERSENSES),
                                         graine=graine)
        accumulateurs["représentations gelées"].append(
            evaluer(enc, tete, X_te, y_te)["exactitude"])

        enc, tete = entrainer_complet(encodeur, X_tr, y_tr, len(SUPERSENSES),
                                      graine=graine)
        accumulateurs["fine-tuning complet"].append(
            evaluer(enc, tete, X_te, y_te)["exactitude"])

        enc, tete = entrainer_depuis_zero(X_tr, y_tr, len(SUPERSENSES),
                                          X_tr.shape[1], graine=graine)
        accumulateurs["depuis zéro"].append(
            evaluer(enc, tete, X_te, y_te)["exactitude"])

    return {
        nom: {"moyenne": float(np.mean(valeurs)),
              "ecart_type": float(np.std(valeurs)),
              "minimum": float(np.min(valeurs)),
              "maximum": float(np.max(valeurs))}
        for nom, valeurs in accumulateurs.items()
    }


def experience_quantite_donnees(proportions: list[float],
                                graines: list[int]) -> list[dict]:
    """Effet de la quantité de données, moyenné sur plusieurs graines."""
    import numpy as np

    lignes = []
    for proportion in proportions:
        mesures: dict[str, list[float]] = {"gele": [], "complet": [], "zero": []}
        n_effectif = 0
        for graine in graines:
            (X_tr, y_tr, y_tr_gros), _, (X_te, y_te, _), _, _ = preparer_transfert(graine)
            encodeur = pre_entrainer(X_tr, y_tr_gros, X_tr.shape[1], graine=graine)
            n = max(int(proportion * len(y_tr)), len(SUPERSENSES))
            n_effectif = n

            enc, tete = entrainer_tete_gelee(encodeur, X_tr[:n], y_tr[:n],
                                             len(SUPERSENSES), graine=graine)
            mesures["gele"].append(evaluer(enc, tete, X_te, y_te)["exactitude"])

            enc, tete = entrainer_complet(encodeur, X_tr[:n], y_tr[:n],
                                          len(SUPERSENSES), graine=graine)
            mesures["complet"].append(evaluer(enc, tete, X_te, y_te)["exactitude"])

            enc, tete = entrainer_depuis_zero(X_tr[:n], y_tr[:n], len(SUPERSENSES),
                                              X_tr.shape[1], graine=graine)
            mesures["zero"].append(evaluer(enc, tete, X_te, y_te)["exactitude"])

        lignes.append({
            "n": n_effectif,
            **{cle: float(np.mean(valeurs)) for cle, valeurs in mesures.items()},
        })
    return lignes


if __name__ == "__main__":
    print("=== Apprentissage par transfert : trois régimes ===\n")

    GRAINES = [0, 1, 2, 3, 4]

    (X_tr, y_tr, y_tr_gros), _, (X_te, y_te, _), extracteur, brut = preparer_transfert()
    train, _, test = brut

    print(f"  tâche de pré-entraînement : {len(HYPERSENSES)} classes grossières")
    print(f"  tâche cible                : {len(SUPERSENSES)} classes fines")
    print(f"  train {len(y_tr)} | test {len(y_te)} | traits {X_tr.shape[1]}")
    print(f"  moyennes sur {len(GRAINES)} graines\n")

    encodeur = pre_entrainer(X_tr, y_tr_gros, X_tr.shape[1])
    tete_reference = TeteClassification(encodeur.dimension_sortie, len(SUPERSENSES))
    print(f"  encodeur : {encodeur.nb_parametres:,} paramètres")
    print(f"  tête     : {tete_reference.nb_parametres:,} paramètres\n")

    # -- Les trois régimes ---------------------------------------------------- #
    print("--- Trois régimes d'adaptation ---\n")
    resume = experience_multi_graines(GRAINES)
    parametres = {
        "représentations gelées": tete_reference.nb_parametres,
        "fine-tuning complet": encodeur.nb_parametres + tete_reference.nb_parametres,
        "depuis zéro": encodeur.nb_parametres + tete_reference.nb_parametres,
    }

    entete = (f"  {'régime':>24} | {'entraînés':>10} | {'test (moy.)':>12} | "
              f"{'écart-type':>11} | {'min-max':>13}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for nom, r in resume.items():
        plage = f"{r['minimum']:.3f}-{r['maximum']:.3f}"
        print(f"  {nom:>24} | {parametres[nom]:>10,} | {r['moyenne']:>12.3f} | "
              f"{r['ecart_type']:>11.3f} | {plage:>13}")

    naive = reference_majoritaire(train, test)
    print(f"  {'référence majoritaire':>24} | {0:>10,} | {naive:>12.3f} | "
          f"{'—':>11} | {'—':>13}")

    ratio = parametres["fine-tuning complet"] / parametres["représentations gelées"]
    print(f"\n  Le fine-tuning complet entraîne {ratio:.0f}× plus de paramètres.")

    print(
        "\n  ATTENTION à la colonne écart-type. Mon test compte 28 exemples :\n"
        "  UN SEUL exemple vaut 3,6 points d'exactitude. Les écarts entre régimes\n"
        "  sont du même ordre que la variabilité entre graines — autrement dit,\n"
        "  ce tableau ne permet PAS de désigner un vainqueur.\n"
        "\n"
        "  C'est le défaut exact de mon rendu de M2 : j'y comparais LoRA et Qwen\n"
        "  sur des exactitudes issues d'UN SEUL découpage, sans écart-type, et\n"
        "  j'en tirais des conclusions.\n"
    )

    # -- L'effet de la quantité de données ------------------------------------- #
    print("--- Où le transfert sert vraiment : le régime à peu de données ---\n")
    lignes = experience_quantite_donnees([0.15, 0.3, 0.5, 0.75, 1.0], GRAINES)
    print(f"  {'exemples':>9} | {'gelé':>7} | {'complet':>8} | {'zéro':>7} | "
          f"{'gain du transfert':>18}")
    print("  " + "-" * 60)
    for ligne in lignes:
        gain = max(ligne["gele"], ligne["complet"]) - ligne["zero"]
        print(f"  {ligne['n']:>9} | {ligne['gele']:>7.3f} | {ligne['complet']:>8.3f} | "
              f"{ligne['zero']:>7.3f} | {gain:>+18.3f}")

    print(
        "\nCe que je retiens\n"
        "-----------------\n"
        "* Le gain du transfert est CONCENTRÉ dans le régime à peu de données.\n"
        "  Avec 12 exemples, partir d'un encodeur pré-entraîné aide nettement.\n"
        "  Avec 80, l'entraînement depuis zéro rattrape — parce que ma tâche de\n"
        "  pré-entraînement est très proche de la tâche cible et que mon encodeur\n"
        "  est minuscule. Sur un vrai modèle pré-entraîné sur des milliards de\n"
        "  mots, l'écart persisterait bien au-delà.\n"
        "\n"
        "* Les REPRÉSENTATIONS GELÉES n'entraînent qu'une fraction des paramètres\n"
        "  et permettent de PRÉ-CALCULER les représentations une fois pour toutes.\n"
        "  C'est ce que je faisais en M1 avec DistilBERT, et c'est ce qui rendait\n"
        "  le TP faisable sur un portable.\n"
        "\n"
        "* Le FINE-TUNING COMPLET coûte une copie complète du modèle PAR TÂCHE.\n"
        "  Avec 7 milliards de paramètres, c'est 28 Go par tâche — et trois fois\n"
        "  plus en mémoire pendant l'entraînement, à cause des états d'Adam.\n"
        "\n"
        "* C'est exactement le problème que LoRA résout, et c'est le sujet de la\n"
        "  version 3 : obtenir l'expressivité du fine-tuning avec le coût des\n"
        "  représentations gelées.\n"
        "\n"
        "* Et la leçon méthodologique, qui vaut plus que les chiffres : sur 28\n"
        "  exemples de test, AUCUNE de ces comparaisons n'est concluante. Le\n"
        "  savoir et le dire vaut mieux que d'annoncer un vainqueur."
    )
