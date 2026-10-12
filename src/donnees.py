"""
La tâche : classer des définitions de dictionnaire en catégories sémantiques.

D'où ça vient
-------------
TP n°1 de *Machine Learning for NLP 3* (Marie Candito, M2), réalisé avec
Haeeul Hwang : classer des définitions du Wiktionnaire français en
**supersenses** — des catégories sémantiques grossières comme `person`,
`artifact`, `event`, `cognition`. Le TP demandait de comparer un fine-tuning
LoRA de FlauBERT à de l'apprentissage en contexte avec Qwen2.5-3B.

Pourquoi cette tâche est bien choisie
--------------------------------------
C'est une tâche de **classification sémantique** où :

* les classes sont naturelles et interprétables (un linguiste peut annoter) ;
* elles sont **déséquilibrées**, comme toujours en TAL réel ;
* et surtout, elles sont **hiérarchiques** : le TP regroupait les 25 supersenses
  en une douzaine d'« hypersenses » (`animate`, `inanimate`,
  `dynamic_situation`…). Cette hiérarchie servira à la version 5.

Ce que je fais ici
------------------
Je reconstruis un corpus du même type, embarqué dans le code. Il est
minuscule — 180 définitions — mais il a les propriétés qui comptent : classes
déséquilibrées, hiérarchie, et vocabulaire partiellement partagé entre classes.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# La hiérarchie sémantique
# --------------------------------------------------------------------------- #

# Reprise de la table `super2hyper` de mon TP de M2 : les supersenses fins sont
# regroupés en hypersenses plus grossiers. Cette hiérarchie n'est pas
# décorative — elle définit une notion de « proximité » entre classes qui
# servira à la sélection d'exemples.
HIERARCHIE: dict[str, str] = {
    "personne": "animé",
    "animal": "animé",
    "objet": "inanimé",
    "aliment": "inanimé",
    "lieu": "inanimé",
    "acte": "situation_dynamique",
    "phénomène": "situation_dynamique",
    "sentiment": "situation_statique",
    "qualité": "situation_statique",
    "communication": "information",
    "connaissance": "information",
    "temps": "temps",
}

SUPERSENSES = sorted(HIERARCHIE)
HYPERSENSES = sorted(set(HIERARCHIE.values()))


# --------------------------------------------------------------------------- #
# Le corpus
# --------------------------------------------------------------------------- #

# Chaque entrée : (définition, supersense). Les définitions sont écrites dans le
# style du Wiktionnaire — commençant souvent par un hyperonyme, ce qui est
# précisément le trait le plus informatif pour la classification.
DEFINITIONS: list[tuple[str, str]] = [
    # --- personne (classe majoritaire, comme dans le vrai corpus) ---------- #
    ("Personne qui exerce le métier de boulanger.", "personne"),
    ("Celui qui enseigne dans une école.", "personne"),
    ("Habitant de la commune de Rennes.", "personne"),
    ("Personne chargée de la vente en magasin.", "personne"),
    ("Homme qui pratique la médecine.", "personne"),
    ("Femme qui dirige une entreprise.", "personne"),
    ("Celle qui écrit des romans.", "personne"),
    ("Individu qui pratique un sport de haut niveau.", "personne"),
    ("Personne âgée vivant seule.", "personne"),
    ("Celui qui répare les véhicules automobiles.", "personne"),
    ("Membre d'une assemblée délibérante.", "personne"),
    ("Personne qui étudie à l'université.", "personne"),
    ("Spécialiste de l'étude des langues.", "personne"),
    ("Celui qui conduit un train.", "personne"),
    ("Personne qui soigne les animaux.", "personne"),
    ("Employé chargé de l'accueil du public.", "personne"),
    ("Artisan qui travaille le bois.", "personne"),
    ("Celui qui commande un navire.", "personne"),

    # --- animal ------------------------------------------------------------ #
    ("Mammifère carnivore domestique de la famille des félidés.", "animal"),
    ("Oiseau migrateur au long bec.", "animal"),
    ("Petit rongeur des champs.", "animal"),
    ("Insecte volant produisant du miel.", "animal"),
    ("Poisson d'eau douce à chair blanche.", "animal"),
    ("Reptile dépourvu de membres.", "animal"),
    ("Grand mammifère herbivore d'Afrique.", "animal"),
    ("Animal domestique élevé pour sa laine.", "animal"),
    ("Batracien à peau lisse vivant près des mares.", "animal"),
    ("Crustacé marin à dix pattes.", "animal"),
    ("Oiseau de proie diurne.", "animal"),
    ("Mollusque terrestre à coquille.", "animal"),

    # --- objet -------------------------------------------------------------- #
    ("Instrument servant à couper le pain.", "objet"),
    ("Meuble sur lequel on pose des objets.", "objet"),
    ("Appareil permettant de mesurer la température.", "objet"),
    ("Récipient utilisé pour boire.", "objet"),
    ("Outil manuel servant à enfoncer des clous.", "objet"),
    ("Dispositif mécanique qui ouvre une serrure.", "objet"),
    ("Vêtement porté sur le haut du corps.", "objet"),
    ("Machine servant à laver le linge.", "objet"),
    ("Support en tissu pour dormir.", "objet"),
    ("Objet de forme allongée servant à écrire.", "objet"),
    ("Appareil électronique servant à téléphoner.", "objet"),
    ("Pièce de mobilier destinée à s'asseoir.", "objet"),
    ("Ustensile de cuisine servant à retourner les aliments.", "objet"),
    ("Instrument optique permettant d'observer les astres.", "objet"),

    # --- aliment ------------------------------------------------------------ #
    ("Préparation à base de farine cuite au four.", "aliment"),
    ("Fruit rouge à noyau récolté en été.", "aliment"),
    ("Produit laitier obtenu par fermentation.", "aliment"),
    ("Légume racine de couleur orange.", "aliment"),
    ("Boisson chaude obtenue par infusion.", "aliment"),
    ("Viande de porc salée et fumée.", "aliment"),
    ("Pâtisserie fourrée à la crème.", "aliment"),
    ("Céréale cultivée dans les régions humides.", "aliment"),
    ("Condiment obtenu par broyage de graines.", "aliment"),
    ("Plat composé de pâtes et de sauce tomate.", "aliment"),

    # --- lieu ---------------------------------------------------------------- #
    ("Bâtiment destiné à l'enseignement.", "lieu"),
    ("Étendue d'eau salée couvrant une vaste surface.", "lieu"),
    ("Espace planté d'arbres en ville.", "lieu"),
    ("Voie de circulation bordée de maisons.", "lieu"),
    ("Établissement où l'on sert des repas.", "lieu"),
    ("Relief de grande hauteur au sommet enneigé.", "lieu"),
    ("Pièce d'une habitation servant au repos.", "lieu"),
    ("Local aménagé pour la conservation des livres.", "lieu"),
    ("Terrain aménagé pour la pratique du football.", "lieu"),
    ("Agglomération importante dotée d'une administration.", "lieu"),

    # --- acte ---------------------------------------------------------------- #
    ("Action de se déplacer rapidement en courant.", "acte"),
    ("Fait de prendre un repas.", "acte"),
    ("Opération consistant à réparer un objet.", "acte"),
    ("Action d'apprendre une discipline nouvelle.", "acte"),
    ("Fait de transporter des marchandises.", "acte"),
    ("Opération de vente d'un bien immobilier.", "acte"),
    ("Action de nettoyer une surface.", "acte"),
    ("Fait de construire un édifice.", "acte"),
    ("Action de traduire un texte dans une autre langue.", "acte"),
    ("Opération d'annotation manuelle d'un corpus.", "acte"),

    # --- phénomène ----------------------------------------------------------- #
    ("Chute d'eau depuis les nuages.", "phénomène"),
    ("Déplacement rapide de l'air atmosphérique.", "phénomène"),
    ("Secousse brutale de l'écorce terrestre.", "phénomène"),
    ("Décharge électrique visible dans le ciel.", "phénomène"),
    ("Élévation générale des températures moyennes.", "phénomène"),
    ("Montée du niveau des eaux d'un fleuve.", "phénomène"),
    ("Formation de cristaux de glace au sol.", "phénomène"),
    ("Obscurcissement temporaire du soleil.", "phénomène"),

    # --- sentiment ------------------------------------------------------------ #
    ("État affectif de contentement profond.", "sentiment"),
    ("Sentiment d'inquiétude face à un danger.", "sentiment"),
    ("Attachement profond éprouvé pour quelqu'un.", "sentiment"),
    ("État de tristesse durable.", "sentiment"),
    ("Vive irritation provoquée par une injustice.", "sentiment"),
    ("Sentiment de gêne devant autrui.", "sentiment"),
    ("Impression d'ennui lié à l'inaction.", "sentiment"),
    ("État de sérénité intérieure.", "sentiment"),

    # --- qualité --------------------------------------------------------------- #
    ("Caractère de ce qui est vrai.", "qualité"),
    ("Propriété de ce qui pèse peu.", "qualité"),
    ("Aptitude à résister à l'effort.", "qualité"),
    ("Caractère de ce qui est difficile à comprendre.", "qualité"),
    ("Qualité de ce qui est fait avec soin.", "qualité"),
    ("Propriété d'un corps de conduire l'électricité.", "qualité"),
    ("Caractère de ce qui se produit souvent.", "qualité"),
    ("Aptitude d'un système à traiter de grandes quantités.", "qualité"),

    # --- communication ---------------------------------------------------------- #
    ("Message écrit adressé à une personne.", "communication"),
    ("Discours prononcé devant un public.", "communication"),
    ("Échange verbal entre deux interlocuteurs.", "communication"),
    ("Texte publié dans une revue scientifique.", "communication"),
    ("Annonce diffusée par voie de presse.", "communication"),
    ("Question posée pour obtenir un renseignement.", "communication"),
    ("Récit transmis oralement de génération en génération.", "communication"),
    ("Document officiel attestant un fait.", "communication"),

    # --- connaissance ------------------------------------------------------------ #
    ("Ensemble des savoirs relatifs à un domaine.", "connaissance"),
    ("Idée que l'on se fait d'une chose.", "connaissance"),
    ("Proposition admise sans démonstration.", "connaissance"),
    ("Méthode systématique de résolution d'un problème.", "connaissance"),
    ("Explication provisoire d'un phénomène observé.", "connaissance"),
    ("Représentation abstraite d'un objet réel.", "connaissance"),
    ("Souvenir conservé d'un événement passé.", "connaissance"),
    ("Croyance non fondée sur des preuves.", "connaissance"),

    # --- temps --------------------------------------------------------------------- #
    ("Période de douze mois consécutifs.", "temps"),
    ("Moment de la journée où le soleil se lève.", "temps"),
    ("Durée séparant deux événements.", "temps"),
    ("Saison comprise entre le printemps et l'automne.", "temps"),
    ("Intervalle de sept jours.", "temps"),
    ("Époque marquée par des transformations profondes.", "temps"),
    ("Instant précis mesuré par une horloge.", "temps"),
    ("Période de repos accordée aux travailleurs.", "temps"),
]

MOTS_VIDES = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "à", "au",
    "aux", "et", "ou", "que", "qui", "dont", "où", "ce", "cet", "cette", "ces",
    "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "est", "sont",
    "a", "ont", "il", "elle", "on", "se", "s", "n", "ne", "pas", "y", "c",
    "son", "sa", "ses", "leur", "plus", "moins", "très", "tout", "toute",
}


# --------------------------------------------------------------------------- #
# Prétraitement
# --------------------------------------------------------------------------- #


def normaliser(texte: str) -> str:
    texte = texte.lower()
    decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def segmenter(texte: str, retirer_mots_vides: bool = True) -> list[str]:
    mots = re.findall(r"[a-z0-9]+", normaliser(texte))
    if retirer_mots_vides:
        mots = [m for m in mots if m not in MOTS_VIDES and len(m) > 1]
    return mots


def decouper(graine: int = 0, proportion_train: float = 0.7,
             proportion_dev: float = 0.15):
    """Découpage **stratifié** train / dev / test.

    Stratifié parce que mes classes sont déséquilibrées (18 « personne » contre
    8 « temps ») : un découpage aléatoire global pourrait laisser une classe
    entière hors du train. C'est le rôle du paramètre `stratify` de
    `sklearn.model_selection.train_test_split`, que j'utilisais en TP sans
    toujours savoir pourquoi.

    Returns:
        ``(train, dev, test)``, chacun une liste de ``(définition, supersense)``.
    """
    import random

    rng = random.Random(graine)
    par_classe: dict[str, list[tuple[str, str]]] = {}
    for definition, classe in DEFINITIONS:
        par_classe.setdefault(classe, []).append((definition, classe))

    train, dev, test = [], [], []
    for classe in sorted(par_classe):
        exemples = list(par_classe[classe])
        rng.shuffle(exemples)
        n = len(exemples)
        n_train = max(1, int(proportion_train * n))
        n_dev = max(1, int(proportion_dev * n))
        train += exemples[:n_train]
        dev += exemples[n_train:n_train + n_dev]
        test += exemples[n_train + n_dev:]

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def distribution_des_classes(exemples: list[tuple[str, str]]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(classe for _, classe in exemples))


def hypersense(supersense: str) -> str:
    return HIERARCHIE[supersense]


def distance_hierarchique(classe_a: str, classe_b: str) -> int:
    """Distance dans la hiérarchie : 0 (identiques), 1 (même hypersense), 2 (sinon).

    C'est une notion de proximité **sémantique**, pas lexicale. Elle servira à
    la version 5, où je sélectionne des exemples pour l'apprentissage en
    contexte : un exemple d'une classe voisine est plus utile qu'un exemple
    d'une classe sans rapport.

    C'est exactement l'idée du framework que j'ai conçu en stage, où la
    hiérarchie de SNOMED CT servait à choisir les exemples d'une invite.
    """
    if classe_a == classe_b:
        return 0
    if HIERARCHIE[classe_a] == HIERARCHIE[classe_b]:
        return 1
    return 2


if __name__ == "__main__":
    print("=== La tâche : classer des définitions en catégories sémantiques ===\n")

    train, dev, test = decouper()
    print(f"  {len(DEFINITIONS)} définitions | {len(SUPERSENSES)} classes fines "
          f"| {len(HYPERSENSES)} classes grossières")
    print(f"  train {len(train)} | dev {len(dev)} | test {len(test)}\n")

    print("--- La hiérarchie sémantique ---\n")
    par_hyper: dict[str, list[str]] = {}
    for supersense, hyper in sorted(HIERARCHIE.items()):
        par_hyper.setdefault(hyper, []).append(supersense)
    for hyper, supersenses in sorted(par_hyper.items()):
        print(f"  {hyper:>22} : {', '.join(supersenses)}")

    print("\n--- Le déséquilibre des classes ---\n")
    distribution = distribution_des_classes(DEFINITIONS)
    maximum = max(distribution.values())
    print(f"  {'classe':>15} | {'n':>3} | part | histogramme")
    print("  " + "-" * 56)
    for classe, n in sorted(distribution.items(), key=lambda x: -x[1]):
        barre = "#" * int(20 * n / maximum)
        print(f"  {classe:>15} | {n:>3} | {100 * n / len(DEFINITIONS):>3.0f}% | {barre}")

    majoritaire = max(distribution.values()) / len(DEFINITIONS)
    print(f"\n  Une référence naïve qui prédit toujours « personne » obtiendrait")
    print(f"  {100 * majoritaire:.1f} % d'exactitude. C'est LE chiffre à battre —")
    print("  et celui qu'on oublie de calculer quand on annonce « 85 % ».\n")

    print("--- Le découpage est stratifié ---\n")
    print(f"  {'classe':>15} | {'train':>6} | {'dev':>4} | {'test':>5}")
    print("  " + "-" * 40)
    d_train = distribution_des_classes(train)
    d_dev = distribution_des_classes(dev)
    d_test = distribution_des_classes(test)
    for classe in SUPERSENSES:
        print(f"  {classe:>15} | {d_train.get(classe, 0):>6} | "
              f"{d_dev.get(classe, 0):>4} | {d_test.get(classe, 0):>5}")
    print("\n  Chaque classe est présente dans les trois ensembles. Avec un")
    print("  découpage aléatoire global et 8 exemples pour « temps », ce n'est")
    print("  pas garanti — et un test sans une classe rend son score indéfini.\n")

    print("--- La distance hiérarchique ---\n")
    exemples = [("personne", "animal"), ("personne", "objet"),
                ("objet", "aliment"), ("sentiment", "qualité"),
                ("temps", "animal"), ("acte", "acte")]
    for a, b in exemples:
        d = distance_hierarchique(a, b)
        libelle = {0: "identiques", 1: "même hypersense", 2: "sans rapport"}[d]
        print(f"  {a:>13} / {b:<13} -> {d}  ({libelle})")

    print("\n  Cette proximité SÉMANTIQUE — indépendante du vocabulaire — servira")
    print("  à la version 5 pour sélectionner les exemples d'une invite few-shot.")
    print("  C'est le principe du framework que j'ai conçu en stage, où la")
    print("  hiérarchie de SNOMED CT jouait ce rôle.")
