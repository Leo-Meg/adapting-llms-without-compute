# Adapting a model without data or compute

Features, transfer, LoRA, in-context learning and example selection, all reimplemented in NumPy.

**Léo Mégret** — implementation, French technical notes.  
**Haeeul Hwang** — English documentation, Korean documentation, review.

> **Repository status** — step 2 of 5, landed by Léo Mégret.

---

## Why this repository

In his final Master's year, with Haeeul, Léo wrote six lines of `peft` to run a
LoRA fine-tune. They worked. He could not have explained a single one of their
arguments.

We then compared that LoRA with in-context learning, on two examples chosen by
hand and never questioned, and drew a conclusion from it.

This repository takes that work up again, around a single question. How do you
adapt a model to a task when you have neither much data nor much compute.

It starts with what our submission did not contain: the simple method.

---

## Steps

| | Step | Modules | Tests | Landed by | Detailed note (French) |
|---|---|---|---:|---|---|
| **1** | The baseline to beat: features and logistic regression | `src/donnees.py`, `src/traits.py` | 26 | Léo | [`traits-et-regression-logistique`](notes/traits-et-regression-logistique.md) |
| **2** | Transfer learning: three regimes, and a lesson in method | `src/transfert.py` | 14 | Léo | [`apprentissage-par-transfert`](notes/apprentissage-par-transfert.md) |

**40** tests in total.

---

## What each step found

**1. The baseline to beat: features and logistic regression**  
The first word alone, 70 dimensions, beats a full 274-dimension bag of words (0.429 against 0.357). Dictionary definitions are structured: the first word is the hypernym.

**2. Transfer learning: three regimes, and a lesson in method**  
The from-scratch control reaches 0.379 against 0.407 for full fine-tuning, with σ = 0.086 on a 28-example test set. The table designates no winner, and we say so.

---

## Running it

```bash
python -m src.transfert

python -m tests.test_traits
python -m tests.test_transfert
```

---

## How we work

**Nothing to download.** The corpus is written into the code. All of
Léo's Master's notebooks began with a `wget` to a university server or a Google
Drive mount. Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in the original coursework are quoted, not erased.** Where a result
that was handed in was wrong or incomplete, we say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what we
expected, we write down what we found.

**The code is commented in French**, because that is the language it was thought
out in. The documentation is in English, and in Korean where Haeeul has caught
up.

---

*Français : [README_FR.md](README_FR.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
