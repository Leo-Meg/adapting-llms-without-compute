"""
Classification par traits : régression logistique et ingénierie de traits.

D'où ça vient
-------------
* TP n°1 de *Machine Learning 2* (M1) : *« Feature-Based Classification »* —
  construire des traits à la main, entraîner un classifieur linéaire.
* TP n°1 de *Machine Learning for NLP 3* (Marie Candito, M2) : la même tâche de
  classification de définitions, mais résolue avec FlauBERT + LoRA.

Pourquoi commencer par là
--------------------------
Parce que **c'est la référence à battre**, et que je ne l'avais pas calculée en
M2. Mon rendu comparait LoRA à Qwen en apprentissage en contexte, sans jamais
mesurer ce que donnait une régression logistique sur des sacs de mots — c'est-à-dire
la méthode de 1990, qui prend deux secondes à entraîner.

C'est une lacune méthodologique classique : on compare deux méthodes modernes
entre elles et on oublie de vérifier qu'elles battent la méthode simple. Les
travaux qui prennent la peine de le faire trouvent régulièrement que l'écart est
bien plus faible qu'annoncé.

Ce que ce module contient
--------------------------
Une régression logistique multinomiale écrite en NumPy — gradient calculé à la
main, descente de gradient, régularisation L2 — et surtout un jeu de traits que
je fais varier pour mesurer ce que chacun apporte.
"""

from __future__ import annotations

import numpy as np

from .donnees import SUPERSENSES, decouper, segmenter


# --------------------------------------------------------------------------- #
# 1. Extraction de traits
# --------------------------------------------------------------------------- #


class ExtracteurTraits:
    """Construit des vecteurs de traits à partir de définitions.

    Chaque famille de traits est activable séparément, ce qui permet de mesurer
    sa contribution par **ablation** — la seule façon honnête de savoir si un
    trait sert à quelque chose.

    Args:
        sac_de_mots: présence/absence de chaque mot du vocabulaire.
        premier_mot: le premier mot de la définition, comme trait à part entière.
        longueur: nombre de mots, discrétisé.
        suffixes: les trois derniers caractères de chaque mot.
    """

    def __init__(self, sac_de_mots: bool = True, premier_mot: bool = False,
                 longueur: bool = False, suffixes: bool = False,
                 frequence_minimale: int = 1):
        self.options = dict(sac_de_mots=sac_de_mots, premier_mot=premier_mot,
                            longueur=longueur, suffixes=suffixes)
        self.frequence_minimale = frequence_minimale
        self.index: dict[str, int] = {}

    def _traits_bruts(self, definition: str) -> list[str]:
        """Liste des traits actifs pour une définition, sous forme de chaînes.

        Représenter les traits par des chaînes préfixées (`mot=chien`,
        `premier=personne`) plutôt que par des indices est un choix
        pédagogique : on peut inspecter le modèle et lire quels traits pèsent.
        C'est ce que permettent les *feature templates* de CRFsuite ou de
        scikit-learn avec `DictVectorizer`.
        """
        mots = segmenter(definition)
        traits: list[str] = []

        if self.options["sac_de_mots"]:
            traits += [f"mot={m}" for m in mots]

        if self.options["premier_mot"] and mots:
            # L'hyperonyme d'une définition de dictionnaire est presque toujours
            # son premier mot : « Personne qui… », « Mammifère… », « Action
            # de… ». C'est le trait le plus informatif de toute la tâche, et il
            # est invisible pour un sac de mots qui ignore les positions.
            traits.append(f"premier={mots[0]}")

        if self.options["longueur"]:
            # Discrétisation : un trait continu dans un modèle linéaire suppose
            # une relation monotone avec la classe, ce qui est rarement vrai.
            tranche = min(len(mots) // 3, 5)
            traits.append(f"longueur={tranche}")

        if self.options["suffixes"]:
            traits += [f"suf={m[-3:]}" for m in mots if len(m) > 3]

        return traits

    def ajuster(self, definitions: list[str]) -> ExtracteurTraits:
        """Construit le vocabulaire de traits **sur le train uniquement**.

        Le construire sur tout le corpus serait une fuite d'information : le
        modèle « connaîtrait » l'existence de traits qu'il ne devrait découvrir
        qu'au test.
        """
        from collections import Counter

        compteur: Counter[str] = Counter()
        for definition in definitions:
            compteur.update(set(self._traits_bruts(definition)))
        retenus = sorted(t for t, n in compteur.items()
                         if n >= self.frequence_minimale)
        self.index = {t: i for i, t in enumerate(retenus)}
        return self

    def transformer(self, definitions: list[str]) -> np.ndarray:
        """Matrice ``(nb_exemples, nb_traits + 1)``, dernière colonne = biais.

        J'ajoute le biais comme une colonne constante plutôt que comme un
        paramètre séparé : le code de la descente de gradient devient identique
        pour tous les paramètres, et c'est une source de bugs en moins.
        """
        X = np.zeros((len(definitions), len(self.index) + 1))
        for i, definition in enumerate(definitions):
            for trait in self._traits_bruts(definition):
                j = self.index.get(trait)
                if j is not None:
                    X[i, j] = 1.0
            X[i, -1] = 1.0
        return X

    @property
    def nb_traits(self) -> int:
        return len(self.index) + 1


# --------------------------------------------------------------------------- #
# 2. Régression logistique multinomiale
# --------------------------------------------------------------------------- #


def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax stable — on retranche le maximum avant l'exponentielle."""
    stables = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(stables)
    return exp / exp.sum(axis=1, keepdims=True)


class RegressionLogistique:
    """Classifieur linéaire multiclasse, entraîné par descente de gradient.

    Le modèle est ``P(classe | x) = softmax(x W)``. Le gradient de l'entropie
    croisée par rapport à ``W`` vaut :

        dL/dW = Xᵀ (softmax(XW) − Y) / n  +  λ W

    C'est exactement le même résultat que dans mon projet « du perceptron au
    Transformer » : la composition softmax + entropie croisée a un gradient sans
    exponentielle ni logarithme. C'est ce qui la rend numériquement irréprochable.

    Args:
        regularisation: coefficient L2. Sur un corpus de 80 exemples et
            plusieurs centaines de traits, il y a bien plus de paramètres que de
            données — la régularisation n'est pas optionnelle.
    """

    def __init__(self, nb_classes: int, regularisation: float = 1e-2,
                 graine: int = 0):
        self.nb_classes = nb_classes
        self.regularisation = regularisation
        self.rng = np.random.default_rng(graine)
        self.W: np.ndarray | None = None

    def entrainer(self, X: np.ndarray, y: np.ndarray, nb_epoques: int = 300,
                  lr: float = 0.5, X_dev: np.ndarray | None = None,
                  y_dev: np.ndarray | None = None,
                  verbeux: bool = False) -> dict[str, list[float]]:
        n, d = X.shape
        self.W = np.zeros((d, self.nb_classes))
        Y = np.zeros((n, self.nb_classes))
        Y[np.arange(n), y] = 1.0

        historique: dict[str, list[float]] = {"perte": [], "exactitude_dev": []}

        for epoque in range(nb_epoques):
            probas = softmax(X @ self.W)
            perte = float(-np.mean(np.log(probas[np.arange(n), y] + 1e-12)))
            perte += 0.5 * self.regularisation * float(np.sum(self.W ** 2))

            gradient = X.T @ (probas - Y) / n + self.regularisation * self.W
            self.W -= lr * gradient

            historique["perte"].append(perte)
            if X_dev is not None:
                historique["exactitude_dev"].append(self.evaluer(X_dev, y_dev))
            if verbeux and (epoque + 1) % 100 == 0:
                print(f"    époque {epoque + 1:>4} | perte {perte:.4f}")

        return historique

    def probabilites(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W)

    def predire(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(X @ self.W, axis=1)

    def evaluer(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predire(X) == y))

    @property
    def nb_parametres(self) -> int:
        return 0 if self.W is None else self.W.size


# --------------------------------------------------------------------------- #
# 3. Métriques
# --------------------------------------------------------------------------- #


def exactitude(vrais: np.ndarray, predits: np.ndarray) -> float:
    return float(np.mean(vrais == predits))


def f1_macro(vrais: np.ndarray, predits: np.ndarray, nb_classes: int) -> float:
    """F1 moyennée **sur les classes** (macro), pas sur les exemples.

    La distinction est décisive sur un corpus déséquilibré. La F1 *micro* (ou
    l'exactitude) est dominée par les classes fréquentes ; la F1 *macro* donne
    le même poids à chaque classe, donc elle punit un modèle qui ignore les
    classes rares.

    Mon rendu de M2 utilisait `average='weighted'`, qui est encore une autre
    chose : une moyenne pondérée par les effectifs, donc très proche de la
    micro. Je ne savais pas ce que je choisissais.
    """
    scores = []
    for classe in range(nb_classes):
        vrais_positifs = int(np.sum((predits == classe) & (vrais == classe)))
        faux_positifs = int(np.sum((predits == classe) & (vrais != classe)))
        faux_negatifs = int(np.sum((predits != classe) & (vrais == classe)))
        if vrais_positifs == 0:
            scores.append(0.0)
            continue
        precision = vrais_positifs / (vrais_positifs + faux_positifs)
        rappel = vrais_positifs / (vrais_positifs + faux_negatifs)
        scores.append(2 * precision * rappel / (precision + rappel))
    return float(np.mean(scores))


def exactitude_hierarchique(vrais: list[str], predits: list[str]) -> float:
    """Crédite les erreurs « proches » dans la hiérarchie sémantique.

    Confondre `personne` et `animal` (tous deux `animé`) est une erreur bien
    moins grave que confondre `personne` et `temps`. L'exactitude classique met
    les deux à zéro ; celle-ci accorde un demi-point aux erreurs de distance 1.

    C'est une métrique que j'aurais dû utiliser en M2 : mon TP construisait
    justement une hiérarchie supersense -> hypersense, et l'évaluation
    l'ignorait complètement.
    """
    from .donnees import distance_hierarchique

    total = 0.0
    for vrai, predit in zip(vrais, predits):
        distance = distance_hierarchique(vrai, predit)
        total += {0: 1.0, 1: 0.5}.get(distance, 0.0)
    return total / max(len(vrais), 1)


def matrice_confusion(vrais: np.ndarray, predits: np.ndarray,
                      nb_classes: int) -> np.ndarray:
    matrice = np.zeros((nb_classes, nb_classes), dtype=int)
    for vrai, predit in zip(vrais, predits):
        matrice[vrai, predit] += 1
    return matrice


# --------------------------------------------------------------------------- #
# 4. Mise en place
# --------------------------------------------------------------------------- #


def preparer(extracteur: ExtracteurTraits, graine: int = 0):
    """Construit ``(X, y)`` pour train, dev et test."""
    train, dev, test = decouper(graine=graine)
    classe_vers_indice = {c: i for i, c in enumerate(SUPERSENSES)}

    extracteur.ajuster([d for d, _ in train])

    def convertir(exemples):
        X = extracteur.transformer([d for d, _ in exemples])
        y = np.array([classe_vers_indice[c] for _, c in exemples])
        return X, y

    return convertir(train), convertir(dev), convertir(test), (train, dev, test)


def reference_majoritaire(train, test) -> float:
    """La référence qu'il faut toujours calculer — et qu'on oublie toujours."""
    from collections import Counter

    majoritaire = Counter(c for _, c in train).most_common(1)[0][0]
    return sum(c == majoritaire for _, c in test) / max(len(test), 1)


if __name__ == "__main__":
    print("=== Classification par traits : la référence à battre ===\n")

    # -- 1. La référence naïve ------------------------------------------------ #
    train, dev, test = decouper()
    naive = reference_majoritaire(train, test)
    print(f"--- La référence naïve ---\n")
    print(f"  classe majoritaire sur le test : {100 * naive:.1f} % d'exactitude")
    print(f"  hasard uniforme                : {100 / len(SUPERSENSES):.1f} %\n")
    print("  C'est LE chiffre que mon rendu de M2 ne calculait pas. Annoncer")
    print("  « 85 % d'exactitude » ne veut rien dire tant qu'on ne sait pas ce")
    print("  que donne le modèle le plus bête possible.\n")

    # -- 2. Ablation des traits ------------------------------------------------ #
    print("--- Que vaut chaque famille de traits ? (ablation) ---\n")
    configurations = [
        ("sac de mots seul", dict(sac_de_mots=True)),
        ("+ premier mot", dict(sac_de_mots=True, premier_mot=True)),
        ("+ longueur", dict(sac_de_mots=True, premier_mot=True, longueur=True)),
        ("+ suffixes", dict(sac_de_mots=True, premier_mot=True, longueur=True,
                            suffixes=True)),
        ("premier mot SEUL", dict(sac_de_mots=False, premier_mot=True)),
        ("suffixes seuls", dict(sac_de_mots=False, suffixes=True)),
    ]

    entete = (f"  {'traits':>22} | {'dim.':>6} | {'exact.':>7} | {'F1 macro':>9} | "
              f"{'hiérarch.':>10}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))

    resultats = {}
    for nom, options in configurations:
        extracteur = ExtracteurTraits(**options)
        (X_tr, y_tr), (X_dev, y_dev), (X_te, y_te), (_, _, brut_test) = preparer(
            extracteur)
        modele = RegressionLogistique(len(SUPERSENSES))
        modele.entrainer(X_tr, y_tr, nb_epoques=400, lr=0.5)

        predits = modele.predire(X_te)
        vrais_noms = [c for _, c in brut_test]
        predits_noms = [SUPERSENSES[i] for i in predits]

        resultats[nom] = {
            "exactitude": exactitude(y_te, predits),
            "f1": f1_macro(y_te, predits, len(SUPERSENSES)),
            "hierarchique": exactitude_hierarchique(vrais_noms, predits_noms),
            "dimension": extracteur.nb_traits,
        }
        r = resultats[nom]
        print(f"  {nom:>22} | {r['dimension']:>6} | {r['exactitude']:>7.3f} | "
              f"{r['f1']:>9.3f} | {r['hierarchique']:>10.3f}")

    print(
        "\n  Le TRAIT LE PLUS UTILE de toute la tâche est le premier mot de la\n"
        "  définition. C'est logique : une définition de dictionnaire commence\n"
        "  presque toujours par son hyperonyme — « Personne qui… », « Mammifère… »,\n"
        "  « Action de… ». Un sac de mots, qui ignore les positions, le noie parmi\n"
        "  les autres.\n"
        "\n"
        "  C'est exactement ce que l'ingénierie de traits savait faire et que les\n"
        "  modèles neuronaux ont rendu inutile : encoder explicitement une\n"
        "  connaissance du domaine. Le prix à payer est qu'il faut CONNAÎTRE le\n"
        "  domaine.\n"
    )

    # -- 3. Trois métriques, trois lectures ------------------------------------ #
    print("--- Pourquoi trois métriques ---\n")
    print("  * EXACTITUDE : dominée par les classes fréquentes.")
    print("  * F1 MACRO   : même poids pour chaque classe — punit un modèle qui")
    print("    ignore les classes rares. Mon rendu de M2 utilisait `weighted`,")
    print("    qui est encore autre chose et que je n'avais pas choisi sciemment.")
    print("  * HIÉRARCHIQUE : crédite les erreurs proches. Confondre « personne »")
    print("    et « animal » (tous deux ANIMÉ) est moins grave que confondre")
    print("    « personne » et « temps ». Mon TP construisait cette hiérarchie…")
    print("    et l'évaluation l'ignorait complètement.\n")

    # -- 4. La courbe d'apprentissage ------------------------------------------ #
    print("--- Combien de données faut-il ? ---\n")
    extracteur = ExtracteurTraits(sac_de_mots=True, premier_mot=True)
    (X_tr, y_tr), _, (X_te, y_te), _ = preparer(extracteur)

    print(f"  {'exemples':>9} | {'exactitude test':>16}")
    print("  " + "-" * 30)
    for proportion in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(int(proportion * len(y_tr)), len(SUPERSENSES))
        modele = RegressionLogistique(len(SUPERSENSES))
        modele.entrainer(X_tr[:n], y_tr[:n], nb_epoques=400, lr=0.5)
        print(f"  {n:>9} | {modele.evaluer(X_te, y_te):>16.3f}")

    print(
        "\n  La courbe est encore ascendante à 80 exemples : ce modèle est limité\n"
        "  par les DONNÉES, pas par sa capacité. C'est précisément la situation\n"
        "  où l'apprentissage par transfert (versions 2 et 3) et l'apprentissage\n"
        "  en contexte (version 4) deviennent intéressants — ils apportent une\n"
        "  connaissance acquise ailleurs."
    )
