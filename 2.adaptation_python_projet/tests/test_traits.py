"""
Tests de la version 1 — données, traits, régression logistique.

    python -m tests.test_traits
"""

from __future__ import annotations

import numpy as np

from src.donnees import (
    DEFINITIONS,
    HIERARCHIE,
    SUPERSENSES,
    decouper,
    distance_hierarchique,
    distribution_des_classes,
    segmenter,
)
from src.traits import (
    ExtracteurTraits,
    RegressionLogistique,
    exactitude,
    exactitude_hierarchique,
    f1_macro,
    matrice_confusion,
    preparer,
    reference_majoritaire,
    softmax,
)


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #


def test_toutes_les_classes_sont_dans_la_hierarchie() -> None:
    classes = {classe for _, classe in DEFINITIONS}
    assert classes == set(HIERARCHIE) == set(SUPERSENSES)


def test_chaque_classe_a_assez_d_exemples() -> None:
    """Une classe à 2 exemples ne peut pas être découpée en train/dev/test."""
    for classe, n in distribution_des_classes(DEFINITIONS).items():
        assert n >= 5, f"{classe} n'a que {n} exemples"


def test_decoupage_stratifie() -> None:
    """Chaque classe doit apparaître dans les trois ensembles."""
    train, dev, test = decouper()
    for ensemble, nom in ((train, "train"), (dev, "dev"), (test, "test")):
        classes = {c for _, c in ensemble}
        assert classes == set(SUPERSENSES), f"{nom} : classes manquantes"


def test_decoupage_sans_recouvrement() -> None:
    train, dev, test = decouper()
    textes_train = {d for d, _ in train}
    textes_dev = {d for d, _ in dev}
    textes_test = {d for d, _ in test}
    assert textes_train.isdisjoint(textes_dev)
    assert textes_train.isdisjoint(textes_test)
    assert textes_dev.isdisjoint(textes_test)
    assert len(train) + len(dev) + len(test) == len(DEFINITIONS)


def test_decoupage_reproductible() -> None:
    assert decouper(graine=7)[0] == decouper(graine=7)[0]


def test_distance_hierarchique() -> None:
    assert distance_hierarchique("personne", "personne") == 0
    assert distance_hierarchique("personne", "animal") == 1     # tous deux animés
    assert distance_hierarchique("personne", "temps") == 2
    # Symétrique.
    for a in SUPERSENSES:
        for b in SUPERSENSES:
            assert distance_hierarchique(a, b) == distance_hierarchique(b, a)


def test_segmentation_retire_les_mots_vides() -> None:
    mots = segmenter("Personne qui exerce le métier de boulanger.")
    assert "le" not in mots and "de" not in mots and "qui" not in mots
    assert "personne" in mots and "boulanger" in mots


# --------------------------------------------------------------------------- #
# Extraction de traits
# --------------------------------------------------------------------------- #


def test_vocabulaire_construit_sur_le_train_seulement() -> None:
    """Le construire sur tout le corpus serait une fuite d'information."""
    train, _, test = decouper()
    extracteur = ExtracteurTraits().ajuster([d for d, _ in train])
    dimension = extracteur.nb_traits
    extracteur.transformer([d for d, _ in test])
    assert extracteur.nb_traits == dimension, "le test a modifié le vocabulaire"


def test_traits_inconnus_ignores_silencieusement() -> None:
    extracteur = ExtracteurTraits().ajuster(["Personne qui court."])
    X = extracteur.transformer(["Zorglub hippopotame xyzzy."])
    assert X.shape == (1, extracteur.nb_traits)
    assert X[0, :-1].sum() == 0.0   # aucun trait connu
    assert X[0, -1] == 1.0          # sauf le biais


def test_colonne_de_biais_toujours_a_un() -> None:
    extracteur = ExtracteurTraits().ajuster([d for d, _ in DEFINITIONS])
    X = extracteur.transformer([d for d, _ in DEFINITIONS[:10]])
    assert np.all(X[:, -1] == 1.0)


def test_le_premier_mot_est_un_trait_a_part() -> None:
    extracteur = ExtracteurTraits(sac_de_mots=False, premier_mot=True)
    extracteur.ajuster(["Personne qui court.", "Mammifère carnivore."])
    assert any(t.startswith("premier=") for t in extracteur.index)
    assert not any(t.startswith("mot=") for t in extracteur.index)


def test_ablation_change_la_dimension() -> None:
    definitions = [d for d, _ in DEFINITIONS]
    petit = ExtracteurTraits(sac_de_mots=False, premier_mot=True).ajuster(definitions)
    grand = ExtracteurTraits(sac_de_mots=True, premier_mot=True,
                             suffixes=True).ajuster(definitions)
    assert petit.nb_traits < grand.nb_traits


def test_frequence_minimale_filtre_les_hapax() -> None:
    definitions = [d for d, _ in DEFINITIONS]
    sans_filtre = ExtracteurTraits(frequence_minimale=1).ajuster(definitions)
    avec_filtre = ExtracteurTraits(frequence_minimale=3).ajuster(definitions)
    assert avec_filtre.nb_traits < sans_filtre.nb_traits


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #


def test_softmax_est_une_distribution() -> None:
    scores = np.random.default_rng(0).normal(size=(5, 4)) * 20
    probas = softmax(scores)
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert np.all(probas > 0)
    # Stable même sur des scores énormes.
    assert np.all(np.isfinite(softmax(np.array([[1000.0, 1001.0]]))))


def test_le_modele_apprend() -> None:
    extracteur = ExtracteurTraits(sac_de_mots=True, premier_mot=True)
    (X_tr, y_tr), _, (X_te, y_te), (train, _, test) = preparer(extracteur)

    modele = RegressionLogistique(len(SUPERSENSES))
    avant = modele_aleatoire_exactitude = 1.0 / len(SUPERSENSES)
    historique = modele.entrainer(X_tr, y_tr, nb_epoques=400, lr=0.5)

    assert historique["perte"][-1] < historique["perte"][0]
    assert modele.evaluer(X_tr, y_tr) > 0.8, "le modèle doit apprendre son train"
    assert modele.evaluer(X_te, y_te) > avant


def test_le_modele_bat_la_reference_majoritaire() -> None:
    """Le seul test qui compte vraiment pour une référence."""
    extracteur = ExtracteurTraits(sac_de_mots=True, premier_mot=True)
    (X_tr, y_tr), _, (X_te, y_te), (train, _, test) = preparer(extracteur)
    modele = RegressionLogistique(len(SUPERSENSES))
    modele.entrainer(X_tr, y_tr, nb_epoques=400, lr=0.5)

    naive = reference_majoritaire(train, test)
    assert modele.evaluer(X_te, y_te) > naive * 1.5, (
        f"modèle {modele.evaluer(X_te, y_te):.3f} vs naïf {naive:.3f}"
    )


def test_la_perte_decroit_de_facon_monotone() -> None:
    extracteur = ExtracteurTraits()
    (X_tr, y_tr), _, _, _ = preparer(extracteur)
    modele = RegressionLogistique(len(SUPERSENSES), regularisation=1e-3)
    historique = modele.entrainer(X_tr, y_tr, nb_epoques=200, lr=0.3)
    pertes = historique["perte"]
    # Avec un pas raisonnable, la descente ne doit jamais remonter.
    assert all(b <= a + 1e-9 for a, b in zip(pertes, pertes[1:]))


def test_la_regularisation_reduit_la_norme_des_poids() -> None:
    extracteur = ExtracteurTraits()
    (X_tr, y_tr), _, _, _ = preparer(extracteur)
    normes = []
    for regularisation in (1e-4, 1e-1):
        modele = RegressionLogistique(len(SUPERSENSES), regularisation=regularisation)
        modele.entrainer(X_tr, y_tr, nb_epoques=300, lr=0.5)
        normes.append(float(np.linalg.norm(modele.W)))
    assert normes[1] < normes[0], normes


def test_le_premier_mot_est_le_trait_le_plus_utile() -> None:
    """Le résultat central de cette version.

    Une définition de dictionnaire commence presque toujours par son hyperonyme.
    Ce seul trait vaut le sac de mots complet, avec cinq fois moins de dimensions.
    """
    def mesurer(**options) -> tuple[float, int]:
        extracteur = ExtracteurTraits(**options)
        (X_tr, y_tr), _, (X_te, y_te), _ = preparer(extracteur)
        modele = RegressionLogistique(len(SUPERSENSES))
        modele.entrainer(X_tr, y_tr, nb_epoques=400, lr=0.5)
        return modele.evaluer(X_te, y_te), extracteur.nb_traits

    sac_seul, dim_sac = mesurer(sac_de_mots=True)
    premier_seul, dim_premier = mesurer(sac_de_mots=False, premier_mot=True)

    assert premier_seul >= sac_seul, (premier_seul, sac_seul)
    assert dim_premier < dim_sac / 2, (dim_premier, dim_sac)


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #


def test_exactitude() -> None:
    assert exactitude(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0
    assert exactitude(np.array([0, 1, 2]), np.array([0, 0, 0])) == 1 / 3


def test_f1_macro_punit_les_classes_ignorees() -> None:
    """La différence essentielle avec l'exactitude sur un corpus déséquilibré."""
    # 8 exemples de classe 0, 2 de classe 1. Le modèle prédit toujours 0.
    vrais = np.array([0] * 8 + [1] * 2)
    predits = np.zeros(10, dtype=int)
    assert exactitude(vrais, predits) == 0.8          # flatteur
    assert f1_macro(vrais, predits, 2) < 0.5          # sévère, et à raison


def test_f1_macro_parfaite() -> None:
    vrais = np.array([0, 1, 2, 0, 1, 2])
    assert f1_macro(vrais, vrais, 3) == 1.0


def test_exactitude_hierarchique_credite_les_erreurs_proches() -> None:
    """Confondre « personne » et « animal » vaut mieux que « personne » et « temps »."""
    parfait = exactitude_hierarchique(["personne"], ["personne"])
    proche = exactitude_hierarchique(["personne"], ["animal"])
    lointain = exactitude_hierarchique(["personne"], ["temps"])
    assert parfait == 1.0
    assert proche == 0.5
    assert lointain == 0.0


def test_exactitude_hierarchique_toujours_superieure_a_l_exactitude() -> None:
    vrais = ["personne", "animal", "temps", "objet"]
    predits = ["animal", "animal", "objet", "aliment"]
    stricte = sum(v == p for v, p in zip(vrais, predits)) / len(vrais)
    assert exactitude_hierarchique(vrais, predits) >= stricte


def test_matrice_de_confusion() -> None:
    vrais = np.array([0, 0, 1, 1])
    predits = np.array([0, 1, 1, 1])
    matrice = matrice_confusion(vrais, predits, 2)
    assert matrice.sum() == 4
    assert matrice[0, 0] == 1 and matrice[0, 1] == 1
    assert matrice[1, 1] == 2


def test_reference_majoritaire() -> None:
    train = [("a", "x")] * 8 + [("b", "y")] * 2
    test = [("c", "x")] * 3 + [("d", "y")] * 1
    assert reference_majoritaire(train, test) == 0.75


def executer_tous_les_tests() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    echecs = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]     {test.__name__}")
        except AssertionError as e:
            echecs += 1
            print(f"  [ÉCHEC]  {test.__name__} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests passés.")
    if echecs:
        raise SystemExit(1)


if __name__ == "__main__":
    print("=== Tests — version 1 : classification par traits ===\n")
    executer_tous_les_tests()
