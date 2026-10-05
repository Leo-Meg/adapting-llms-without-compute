# Version 2, apprentissage par transfert. Trois régimes, et une leçon de méthode

> **Où j'en suis.** La version 1 a montré que le modèle plafonne par manque de
> données. Le transfert est la réponse. Cette version en mesure trois régimes,
> et découvre au passage que sur 28 exemples de test, **aucune comparaison n'est
> concluante**.

---

## Nouveautés par rapport à la version 1

| Fichier | Ce que j'ajoute |
|---|---|
| `src/transfert.py` | `Encodeur` avec rétropropagation complète, `TeteClassification`, pré-entraînement sur une tâche voisine, les trois régimes d'adaptation, expériences moyennées sur plusieurs graines. |
| `tests/test_transfert.py` | 14 tests, dont la vérification par différences finies des gradients de l'encodeur. |

## Origine universitaire

- **TP n°3 de M1.** *« get the embeddings of the amazon_polarity dataset, use
  these representations to train a logistic regression »*, le régime des
  **représentations gelées**.
- **TP n°1 de M2** (Marie Candito), fine-tuning de FlauBERT, on met à jour
  **tout** le modèle.

Entre les deux, il y a un continuum. C'est lui que je mesure.

## Lancer le code

```bash
python -m src.transfert
```

```bash
python -m tests.test_transfert
```

---

## Le protocole, et pourquoi il est honnête

Je ne peux pas embarquer FlauBERT dans ce dépôt. Je **simule** donc la situation
qui compte, un encodeur pré-entraîné sur une tâche **voisine mais différente**.

```
pré-entraînement : prédire l'HYPERSENSE   (6 classes grossières : animé, inanimé…)
tâche cible      : prédire le SUPERSENSE  (12 classes fines : personne, animal…)
```

C'est exactement le rapport entre le pré-entraînement d'un modèle de langue et
une tâche en aval, le premier apprend des régularités générales, la seconde
demande des distinctions spécifiques.

---

## Les trois régimes

| régime | ce qu'on entraîne | avantage | coût |
|---|---|---|---|
| **représentations gelées** | la tête seule | représentations **pré-calculables**, impossible de casser le modèle | ne peut pas inventer une information absente |
| **fine-tuning complet** | tout | s'adapte réellement | une copie du modèle **par tâche**, ×3 en mémoire avec Adam |
| **depuis zéro** | tout, sans pré-entraînement | le témoin indispensable | pas de connaissance apportée |

---

## Résultat n°1, le tableau qui ne permet pas de conclure

Moyennes sur **5 graines**.

| régime | entraînés | test (moy.) | écart-type | min et max |
|---|---:|---:|---:|---|
| représentations gelées | 396 | 0,279 | 0,027 | 0,250 à 0,321 |
| fine-tuning complet | 24 492 | **0,407** | 0,048 | 0,357 à 0,500 |
| depuis zéro | 24 492 | 0,379 | **0,086** | 0,286 à 0,536 |
| référence majoritaire | 0 | 0,143 |, |, |

Le fine-tuning complet entraîne **62× plus de paramètres**.

**Mais regardez la colonne écart-type.** Mon test compte 28 exemples. **un seul
exemple vaut 3,6 points d'exactitude**. L'écart-type de « depuis zéro » (0,086)
est *deux fois* l'écart entre les deux meilleures méthodes.

> Ce tableau ne permet pas de désigner un vainqueur, et je préfère l'écrire.

C'est le défaut exact de mon rendu de M2, j'y comparais LoRA et Qwen sur des
exactitudes issues d'**un seul découpage**, sans écart-type, et j'en tirais des
conclusions.

---

## Résultat n°2, où le transfert sert réellement

| exemples | gelé | complet | depuis zéro | **gain du transfert** |
|---:|---:|---:|---:|---:|
| **12** | **0,257** | 0,221 | 0,136 | **+0,121** |
| 24 | 0,236 | 0,271 | 0,214 | +0,057 |
| 40 | 0,279 | 0,314 | 0,264 | +0,050 |
| 60 | 0,307 | 0,321 | 0,314 | +0,007 |
| 80 | 0,279 | **0,407** | 0,379 | +0,029 |

**Le gain du transfert est concentré dans le régime à peu de données.** Avec 12
exemples, il vaut 12 points, à 80, il s'évanouit.

Deux lectures.

- C'est **le résultat attendu**, c'est toute la raison d'être du transfert.
- Et c'est **atténué par mon dispositif.** Ma tâche de pré-entraînement est
  très proche de la cible et mon encodeur est minuscule (24 096 paramètres). Sur
  un vrai modèle pré-entraîné sur des milliards de mots, l'écart persisterait
  bien au-delà.

On voit aussi que le régime **gelé** est le meilleur à 12 exemples et le pire à
80, sa faible capacité est une protection quand les données manquent, et un
plafond quand elles abondent.

---

## Ce que coûte vraiment un fine-tuning complet

| | représentations gelées | fine-tuning complet |
|---|---:|---:|
| paramètres entraînés (ici) | 396 | 24 492 |
| représentations pré-calculables | **oui** | non |
| copies du modèle par tâche | 0 | **1** |
| mémoire d'optimiseur (Adam) | ×3 des 396 | **×3 du modèle entier** |

Pour un modèle de 7 milliards de paramètres, cela fait **28 Go par tâche**, et
trois fois plus en mémoire pendant l'entraînement, parce qu'Adam stocke deux
moments par paramètre.

C'est le problème que LoRA résout, et je n'ai pas encore mesuré ce que cela
donne.

---

## Deux points de code que je verrouille par des tests

**`Encodeur.copie()` est une copie profonde.** Sans elle, comparer deux régimes
est impossible, le second partirait d'un encodeur déjà modifié par le premier.
(`test_copie_est_independante`, `test_le_fine_tuning_modifie_l_encodeur`)

**Le régime gelé ne modifie *aucun* paramètre de l'encodeur.** C'est la
définition même du *linear probing*, et c'est le genre de propriété qu'on croit
évidente jusqu'au jour où un `-=` traîne au mauvais endroit.
(`test_le_regime_gele_ne_modifie_pas_l_encodeur`)

Et les gradients de l'encodeur sont vérifiés par différences finies, comme dans
mon projet « du perceptron au Transformer ».

---

## Ce qui reste ouvert

Les représentations gelées coûtent 396 paramètres et plafonnent à 0,279. Le
fine-tuning complet en coûte 24 492 pour atteindre 0,407.

Entre les deux, je ne connais aucun régime intermédiaire.
