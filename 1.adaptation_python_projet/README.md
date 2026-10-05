# Version 1, la référence à battre. Traits et régression logistique

> **Où j'en suis.** Avant de parler de LoRA ou d'apprentissage en contexte, je
> calcule ce que donne la méthode de 1990 sur la même tâche. C'est le chiffre que
> mon rendu de M2 ne calculait pas, et sans lui les autres ne veulent rien dire.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/donnees.py` | 122 définitions de dictionnaire réparties en 12 classes sémantiques, hiérarchie supersense vers hypersense, découpage stratifié, distance hiérarchique. |
| `src/traits.py` | Extracteur de traits activables par famille, sac de mots, premier mot, longueur, suffixes. Régression logistique multinomiale en NumPy, et trois métriques dont une hiérarchique. |
| `tests/test_traits.py` | 26 tests. |

## Origine universitaire

- TP n°1 de Machine Learning 2 (M1), *Feature-Based Classification*.
- TP n°1 de Machine Learning for NLP 3 (Marie Candito, M2), avec Haeeul Hwang.
  Classer des définitions du Wiktionnaire en supersenses, en comparant un
  fine-tuning LoRA de FlauBERT à de l'apprentissage en contexte avec
  Qwen2.5-3B.

## Lancer le code

```bash
python -m src.donnees
```

```bash
python -m src.traits
```

```bash
python -m tests.test_traits
```

---

## La lacune que je répare

Mon rendu de M2 comparait LoRA à Qwen en apprentissage en contexte. Il ne
calculait jamais ce que donnait une régression logistique sur des sacs de mots,
qui prend deux secondes à entraîner.

C'est une lacune méthodologique classique. On compare deux méthodes modernes entre
elles et on oublie de vérifier qu'elles battent la méthode simple. Les travaux qui
prennent la peine de le faire trouvent souvent que l'écart est plus faible
qu'annoncé.

```
classe majoritaire sur le test : 14,3 %
hasard uniforme                :  8,3 %
```

Voilà le plancher. Annoncer 85 % d'exactitude ne veut rien dire tant qu'on ne l'a
pas écrit.

---

## Résultat n°1, le trait le plus utile est le premier mot

| traits | dimension | exactitude | F1 macro | hiérarchique |
|---|---:|---:|---:|---:|
| sac de mots seul | 274 | 0,357 | 0,258 | 0,411 |
| avec le premier mot | 343 | 0,429 | 0,350 | 0,482 |
| avec la longueur | 346 | 0,429 | 0,351 | 0,482 |
| avec les suffixes | 484 | 0,393 | 0,318 | 0,429 |
| premier mot seul | 70 | 0,429 | 0,375 | 0,482 |
| suffixes seuls | 139 | 0,286 | 0,182 | 0,339 |

Le premier mot seul, soit 70 dimensions, fait aussi bien que le sac de mots
complet plus le premier mot, soit 343 dimensions, et obtient même une meilleure
F1 macro.

Cela s'explique. Une définition de dictionnaire commence presque toujours par son
hyperonyme.

> *« **Personne** qui exerce le métier de boulanger. »*
> *« **Mammifère** carnivore domestique. »*
> *« **Action** de se déplacer rapidement. »*

Un sac de mots ignore les positions, et noie donc ce mot décisif parmi les
autres.

C'est ce que l'ingénierie de traits savait faire et que les modèles neuronaux ont
rendu inutile, encoder explicitement une connaissance du domaine. Le prix à payer
est qu'il faut connaître le domaine.

À noter aussi que les suffixes dégradent le résultat. Ils ajoutent 141 dimensions
de bruit sur 80 exemples d'entraînement.

---

## Résultat n°2, trois métriques, trois lectures

| métrique | ce qu'elle mesure | son défaut |
|---|---|---|
| exactitude | proportion de bonnes réponses | dominée par les classes fréquentes |
| F1 macro | moyenne des F1 par classe | punit un modèle qui ignore les classes rares |
| hiérarchique | crédite les erreurs proches | demande une hiérarchie annotée |

`test_f1_macro_punit_les_classes_ignorees` montre l'écart sur un cas construit. Un
modèle qui prédit toujours la classe majoritaire obtient 0,80 d'exactitude et
moins de 0,50 de F1 macro.

Mon rendu de M2 utilisait `average='weighted'`, qui est encore une troisième
chose, une moyenne pondérée par les effectifs, donc proche de la micro. Je ne
savais pas ce que je choisissais.

### La métrique hiérarchique

Confondre `personne` et `animal`, qui sont tous deux animés, est une erreur moins
grave que confondre `personne` et `temps`. L'exactitude classique met les deux à
zéro, la mienne accorde un demi-point aux erreurs de distance 1.

C'est une métrique que j'aurais dû utiliser en M2. Mon TP construisait justement
cette hiérarchie, dans `super2hyper`, et l'évaluation l'ignorait.

---

## Résultat n°3, le modèle est limité par les données

| exemples d'entraînement | exactitude test |
|---:|---:|
| 12 | 0,143 |
| 20 | 0,250 |
| 40 | 0,357 |
| 60 | 0,393 |
| 80 | 0,429 |

La courbe est encore ascendante à 80 exemples. Ce n'est pas la capacité du modèle
qui plafonne, c'est la quantité de données.

C'est précisément la situation où une connaissance acquise ailleurs, sur des
milliards de mots, devrait aider.

---

## Deux réflexes méthodologiques que je verrouille par des tests

**Le vocabulaire de traits est construit sur le train uniquement.** Le construire
sur tout le corpus serait une fuite d'information, le modèle connaîtrait
l'existence de traits qu'il ne devrait découvrir qu'au test. Vérifié par
`test_vocabulaire_construit_sur_le_train_seulement`.

**Le découpage est stratifié.** Avec 8 exemples pour la classe `temps`, un
découpage aléatoire global peut produire un test sans aucun exemple de cette
classe, et son score devient indéfini. Vérifié par `test_decoupage_stratifie`.

---

## Ce qui reste ouvert

La courbe d'apprentissage est encore ascendante à 80 exemples. Ce n'est pas le
modèle qui plafonne, c'est la quantité de données.

J'ai maintenant le chiffre de référence qui manquait à mon rendu de M2. Ce que je
n'ai pas, c'est de quoi le comparer. Tant que je n'ai pas mesuré une méthode qui
réutilise une connaissance acquise ailleurs, je ne sais pas ce que ce 14,3 %
vaut.
