"""
Tests de la version 2 — apprentissage par transfert.

    python -m tests.test_transfert
"""

from __future__ import annotations

import numpy as np

from src.donnees import HYPERSENSES, SUPERSENSES
from src.transfert import (
    Encodeur,
    TeteClassification,
    entrainer_complet,
    entrainer_depuis_zero,
    entrainer_tete_gelee,
    evaluer,
    experience_quantite_donnees,
    perte_et_gradient_scores,
    pre_entrainer,
    preparer_transfert,
)


def _donnees(graine: int = 0):
    return preparer_transfert(graine)


# --------------------------------------------------------------------------- #
# Encodeur
# --------------------------------------------------------------------------- #


def test_formes_de_l_encodeur() -> None:
    encodeur = Encodeur(50, dimension_cachee=16, dimension_sortie=8)
    X = np.random.default_rng(0).normal(size=(7, 50))
    assert encodeur.encoder(X).shape == (7, 8)


def test_sortie_bornee_par_tanh() -> None:
    encodeur = Encodeur(20)
    X = np.random.default_rng(0).normal(size=(5, 20)) * 100
    H = encodeur.encoder(X)
    assert np.all(np.abs(H) <= 1.0)


def test_copie_est_independante() -> None:
    """Sans copie profonde, comparer deux régimes est impossible."""
    encodeur = Encodeur(10)
    clone = encodeur.copie()
    clone.params["W1"] += 1.0
    assert not np.allclose(encodeur.params["W1"], clone.params["W1"])


def test_gradients_de_l_encodeur() -> None:
    """Différences finies sur la rétropropagation de l'encodeur."""
    rng = np.random.default_rng(0)
    encodeur = Encodeur(6, dimension_cachee=5, dimension_sortie=4, graine=0)
    X = rng.normal(size=(3, 6))

    def cout() -> float:
        return float(np.sum(encodeur.encoder(X) ** 2))

    sortie, cache = encodeur.encoder_avec_cache(X)
    grads = encodeur.gradients(cache, 2.0 * sortie)

    eps = 1e-6
    for nom, parametre in encodeur.params.items():
        for _ in range(4):
            idx = tuple(int(rng.integers(0, d)) for d in parametre.shape)
            initial = parametre[idx]
            parametre[idx] = initial + eps
            c_plus = cout()
            parametre[idx] = initial - eps
            c_moins = cout()
            parametre[idx] = initial

            numerique = (c_plus - c_moins) / (2 * eps)
            analytique = grads[nom][idx]
            denom = max(abs(numerique) + abs(analytique), 1e-9)
            assert abs(numerique - analytique) / denom < 1e-4, f"{nom}{idx}"


def test_perte_et_gradient() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(6, 4))
    y = rng.integers(0, 4, size=6)
    perte, grad = perte_et_gradient_scores(scores, y)
    assert perte > 0
    assert grad.shape == scores.shape
    # Le gradient somme à zéro par ligne (propriété du softmax).
    assert np.allclose(grad.sum(axis=1), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Pré-entraînement
# --------------------------------------------------------------------------- #


def test_le_pre_entrainement_change_l_encodeur() -> None:
    (X_tr, _, y_gros), _, _, _, _ = _donnees()
    avant = Encodeur(X_tr.shape[1], graine=0)
    apres = pre_entrainer(X_tr, y_gros, X_tr.shape[1], nb_epoques=100)
    assert not np.allclose(avant.params["W1"], apres.params["W1"])


def test_les_representations_pre_entrainees_separent_les_classes_grossieres() -> None:
    """Contrôle de santé : le pré-entraînement doit servir à quelque chose.

    Une tête linéaire posée sur les représentations gelées doit apprendre la
    tâche grossière presque parfaitement, puisque l'encodeur a été entraîné
    pour ça.
    """
    (X_tr, _, y_gros), _, _, _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])
    _, tete = entrainer_tete_gelee(encodeur, X_tr, y_gros, len(HYPERSENSES))
    assert evaluer(encodeur, tete, X_tr, y_gros)["exactitude"] > 0.8


# --------------------------------------------------------------------------- #
# Les trois régimes
# --------------------------------------------------------------------------- #


def test_le_regime_gele_ne_modifie_pas_l_encodeur() -> None:
    """C'est la définition même du *linear probing*."""
    (X_tr, y_tr, y_gros), _, _, _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])
    avant = {n: p.copy() for n, p in encodeur.params.items()}
    entrainer_tete_gelee(encodeur, X_tr, y_tr, len(SUPERSENSES))
    for nom, valeur in avant.items():
        assert np.allclose(encodeur.params[nom], valeur), nom


def test_le_fine_tuning_modifie_l_encodeur() -> None:
    (X_tr, y_tr, y_gros), _, _, _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])
    avant = {n: p.copy() for n, p in encodeur.params.items()}
    adapte, _ = entrainer_complet(encodeur, X_tr, y_tr, len(SUPERSENSES))
    # L'original est intact (grâce à la copie)…
    for nom, valeur in avant.items():
        assert np.allclose(encodeur.params[nom], valeur), nom
    # …et la copie a bougé.
    assert not np.allclose(adapte.params["W1"], avant["W1"])


def test_ecart_de_parametres_entraines() -> None:
    """L'argument central en faveur des méthodes économes."""
    (X_tr, y_tr, y_gros), _, _, _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])
    tete = TeteClassification(encodeur.dimension_sortie, len(SUPERSENSES))

    geles = tete.nb_parametres
    complets = encodeur.nb_parametres + tete.nb_parametres
    assert complets > geles * 20, (geles, complets)


def test_les_trois_regimes_battent_la_reference_aleatoire() -> None:
    (X_tr, y_tr, y_gros), _, (X_te, y_te, _), _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])
    hasard = 1.0 / len(SUPERSENSES)

    for entraineur in (entrainer_tete_gelee, entrainer_complet):
        enc, tete = entraineur(encodeur, X_tr, y_tr, len(SUPERSENSES))
        assert evaluer(enc, tete, X_te, y_te)["exactitude"] > hasard

    enc, tete = entrainer_depuis_zero(X_tr, y_tr, len(SUPERSENSES), X_tr.shape[1])
    assert evaluer(enc, tete, X_te, y_te)["exactitude"] > hasard


def test_le_fine_tuning_memorise_mieux_le_train() -> None:
    """Plus de paramètres = plus de latitude pour mémoriser."""
    (X_tr, y_tr, y_gros), _, _, _, _ = _donnees()
    encodeur = pre_entrainer(X_tr, y_gros, X_tr.shape[1])

    enc_gele, tete_gelee = entrainer_tete_gelee(encodeur, X_tr, y_tr,
                                                len(SUPERSENSES))
    enc_complet, tete_complete = entrainer_complet(encodeur, X_tr, y_tr,
                                                   len(SUPERSENSES))

    train_gele = evaluer(enc_gele, tete_gelee, X_tr, y_tr)["exactitude"]
    train_complet = evaluer(enc_complet, tete_complete, X_tr, y_tr)["exactitude"]
    assert train_complet >= train_gele, (train_gele, train_complet)


def test_le_transfert_aide_surtout_quand_les_donnees_sont_rares() -> None:
    """Le résultat central de cette version, moyenné sur plusieurs graines.

    Avec très peu d'exemples, partir d'un encodeur pré-entraîné aide nettement.
    L'écart se réduit quand les données abondent.
    """
    lignes = experience_quantite_donnees([0.15, 1.0], graines=[0, 1, 2, 3, 4])
    peu, beaucoup = lignes[0], lignes[1]

    gain_peu = max(peu["gele"], peu["complet"]) - peu["zero"]
    gain_beaucoup = max(beaucoup["gele"], beaucoup["complet"]) - beaucoup["zero"]

    assert gain_peu > 0.05, gain_peu
    assert gain_peu > gain_beaucoup, (gain_peu, gain_beaucoup)


def test_la_variabilite_entre_graines_est_du_meme_ordre_que_les_ecarts() -> None:
    """La leçon méthodologique : sur 28 exemples de test, rien n'est concluant.

    Un seul exemple vaut 3,6 points. Je vérifie que l'écart-type entre graines
    est bien du même ordre que les écarts entre méthodes — ce qui interdit de
    désigner un vainqueur.
    """
    from src.transfert import experience_multi_graines

    resume = experience_multi_graines([0, 1, 2, 3, 4])
    moyennes = [r["moyenne"] for r in resume.values()]
    ecarts_types = [r["ecart_type"] for r in resume.values()]

    etendue_entre_methodes = max(moyennes) - min(moyennes)
    variabilite = max(ecarts_types)
    assert variabilite > etendue_entre_methodes / 4, (
        f"étendue {etendue_entre_methodes:.3f} vs variabilité {variabilite:.3f}"
    )


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
    print("=== Tests — version 2 : transfert ===\n")
    executer_tous_les_tests()
