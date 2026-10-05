# Adapter un modèle sans données ni calcul

La référence à battre, calculée d'abord. Traits et régression logistique, écrits en NumPy.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 2.** Je mène ce travail par étapes, chacune dans son
> propre dossier. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

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

## Les versions publiées

| | Dossier | Contenu | Tests |
|---|---|---|---:|
| **1** | `1.adaptation_python_projet` | La référence à battre, traits et régression logistique | 26 |
| **2** | `2.adaptation_python_projet` | Apprentissage par transfert, trois régimes | 14 |

Soit **40 tests** au total. Chaque dossier contient tout le contenu du
précédent, plus une étape.

---

## Lancer la dernière version

```bash
cd 2.adaptation_python_projet
python -m src.transfert
python -m tests.test_transfert
```

---

## Ce qui reste ouvert

Les représentations gelées coûtent 396 paramètres et plafonnent à 0,279. Le
fine-tuning complet en coûte 24 492 pour atteindre 0,407.

Entre les deux, je ne connais aucun régime intermédiaire.

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

---

*Version anglaise, [README.md](README.md).*
