# Adapter un modèle sans données ni calcul

Traits, transfert, LoRA, apprentissage en contexte et sélection d'exemples, tous réimplémentés en NumPy.

**Léo Mégret** — implémentation, notes techniques en français.  
**Haeeul Hwang** — documentation anglaise, documentation coréenne, relecture.

> **État du dépôt** — étape 2 sur 5, publiée par Léo Mégret.

---

## Pourquoi ce dépôt

En M2, avec Haeeul, nous avons écrit six lignes de `peft` pour faire un
fine-tuning LoRA. Elles marchaient. Je n'aurais su expliquer aucun de leurs
arguments.

Nous avons ensuite comparé ce LoRA à de l'apprentissage en contexte, sur deux
exemples choisis à la main et jamais remis en question, et nous en avons tiré une
conclusion.

Ce dépôt est la reprise de ce travail, autour d'une seule question. Comment
adapter un modèle à une tâche quand on n'a ni beaucoup de données ni beaucoup de
calcul.

Je commence par ce que notre rendu ne contenait pas : la méthode simple.

---

## Les étapes

| | Étape | Modules | Tests | Publiée par | Note détaillée (français) |
|---|---|---|---:|---|---|
| **1** | La référence à battre : traits et régression logistique | `src/donnees.py`, `src/traits.py` | 26 | Léo | [`traits-et-regression-logistique`](notes/traits-et-regression-logistique.md) |
| **2** | Apprentissage par transfert : trois régimes, et une leçon de méthode | `src/transfert.py` | 14 | Léo | [`apprentissage-par-transfert`](notes/apprentissage-par-transfert.md) |

**40** tests au total.

---

## Ce que chaque étape a trouvé

**1. La référence à battre : traits et régression logistique**  
Le premier mot seul, 70 dimensions, bat un sac de mots complet de 274 dimensions (0,429 contre 0,357). Les définitions de dictionnaire sont structurées : le premier mot est l'hyperonyme.

**2. Apprentissage par transfert : trois régimes, et une leçon de méthode**  
Le témoin « depuis zéro » atteint 0,379 contre 0,407 pour le fine-tuning complet, avec σ = 0,086 sur un test de 28 exemples. Le tableau ne désigne aucun vainqueur, et nous l'écrivons.

---

## Lancer le code

```bash
python -m src.transfert

python -m tests.test_traits
python -m tests.test_transfert
```

---

## Comment nous travaillons

**Rien à télécharger.** Le corpus est écrit dans le code. Tous mes
notebooks de master commençaient par un `wget` vers un serveur universitaire ou
un montage de Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à
une commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu était faux ou incomplet, je le dis et je donne le bon.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français**, parce que c'est la langue dans laquelle je
l'ai pensé. La documentation est en anglais, et en coréen quand Haeeul est
passée dessus.

---

*English: [README.md](README.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
