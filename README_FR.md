# Adapter un modèle sans données ni calcul

La référence à battre, calculée d'abord. Traits et régression logistique, écrits en NumPy.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 1.** C'est la première étape d'un travail que je mène
> par étapes, chacune dans son propre dossier. Seule la version 1 existe à ce
> jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

---

## Pourquoi ce dépôt

En M2, avec Haeeul Hwang, nous avons écrit six lignes de `peft` pour faire un
fine-tuning LoRA. Elles marchaient. Je n'aurais su expliquer aucun de leurs
arguments.

Nous avons ensuite comparé ce LoRA à de l'apprentissage en contexte, sur deux
exemples choisis à la main et jamais remis en question, et nous en avons tiré une
conclusion.

Ce dépôt est la reprise de ce travail, autour d'une seule question. Comment
adapter un modèle à une tâche quand on n'a ni beaucoup de données ni beaucoup de
calcul.

Je commence par ce que notre rendu ne contenait pas, la méthode simple.

---

## Ce qui existe aujourd'hui

### Version 1, la référence à battre. Traits et régression logistique

Avant de parler de LoRA ou d'apprentissage en contexte, je calcule ce que donne la
méthode simple sur la même tâche. C'est le chiffre que mon rendu de M2 ne
calculait pas, et sans lui les autres ne veulent rien dire.

| Fichier | Ce que j'y fais |
|---|---|
| `src/donnees.py` | 122 définitions de dictionnaire réparties en 12 classes sémantiques, hiérarchie supersense vers hypersense, découpage stratifié, distance hiérarchique. |
| `src/traits.py` | Extracteur de traits activables par famille, sac de mots, premier mot, longueur, suffixes. Régression logistique multinomiale en NumPy, et trois métriques dont une hiérarchique. |
| `tests/test_traits.py` | 26 tests. |

Origine universitaire. TP n°1 de *Machine Learning 2* (M1), et TP n°1 de *Machine
Learning for NLP 3* (Marie Candito, M2) avec Haeeul Hwang, où il s'agissait de
classer des définitions du Wiktionnaire en supersenses, en comparant un
fine-tuning LoRA de FlauBERT à de l'apprentissage en contexte avec Qwen2.5-3B.

---

## Lancer le code

```bash
cd 1.adaptation_python_projet
python -m src.donnees
python -m src.traits
python -m tests.test_traits
```

NumPy suffit.

---

## Ce que je retiens de cette étape

**Comparer deux méthodes modernes sans vérifier qu'elles battent la méthode simple
est une lacune méthodologique classique.** C'est ce que nous avions fait. Les
travaux qui prennent la peine de calculer la référence trouvent souvent que
l'écart annoncé est plus faible qu'il n'y paraît.

```
classe majoritaire sur le test : 14,3 %
hasard uniforme                :  8,3 %
```

**Trois métriques donnent trois lectures du même modèle.** L'exactitude est dominée
par les classes fréquentes. Le F1 macro punit un modèle qui ignore les classes
rares. La métrique hiérarchique distingue une erreur proche d'une erreur absurde.

**Le découpage doit être stratifié.** Avec 8 exemples pour la classe `temps`, un
découpage aléatoire global peut produire un test sans aucun exemple de cette
classe, et son score devient indéfini.

---

## Ce qui reste ouvert

La courbe d'apprentissage est encore ascendante à 80 exemples. Ce n'est pas le
modèle qui plafonne, c'est la quantité de données.

J'ai maintenant le chiffre de référence qui manquait à mon rendu de M2. Ce que je
n'ai pas, c'est de quoi le comparer. Tant que je n'ai pas mesuré une méthode qui
réutilise une connaissance acquise ailleurs, je ne sais pas ce que ce 14,3 %
vaut.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est écrit dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive. Deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français.**

---

*Version anglaise, [README.md](README.md).*
