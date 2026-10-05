# Adapting a model without data or compute

The baseline to beat, computed first. Features and logistic regression, written in NumPy.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 1.** This is the first step of work I am doing in
> stages, each in its own folder. Only version 1 exists so far. I publish as I go
> rather than once everything is finished.

---

## Why this repository

In my final Master's year, with Haeeul Hwang, we wrote six lines of `peft` to run
a LoRA fine-tune. They worked. I could not have explained a single one of their
arguments.

We then compared that LoRA with in-context learning, on two examples chosen by
hand and never questioned, and drew a conclusion from it.

This repository takes that work up again, around a single question. How do you
adapt a model to a task when you have neither much data nor much compute.

I start with what our submission did not contain, the simple method.

---

## What exists today

### Version 1, the baseline to beat. Features and logistic regression

Before talking about LoRA or in-context learning, I compute what the simple method
gives on the same task. That is the figure my coursework did not compute, and
without it the others mean nothing.

| File | What I do in it |
|---|---|
| `src/donnees.py` | 122 dictionary definitions across 12 semantic classes, a supersense to hypersense hierarchy, a stratified split, a hierarchical distance. |
| `src/traits.py` | A feature extractor switchable by family, bag of words, first word, length, suffixes. Multinomial logistic regression in NumPy, and three metrics including a hierarchical one. |
| `tests/test_traits.py` | 26 tests. |

Academic origin. Lab 1 of *Machine Learning 2* (first Master's year), and lab 1 of
*Machine Learning for NLP 3* (Marie Candito, final year) with Haeeul Hwang, where
the task was to classify Wiktionary definitions into supersenses, comparing a LoRA
fine-tune of FlauBERT with in-context learning using Qwen2.5-3B.

---

## Running the code

```bash
cd 1.adaptation_python_projet
python -m src.donnees
python -m src.traits
python -m tests.test_traits
```

NumPy is enough.

---

## What I take from this step

**Comparing two modern methods without checking that they beat the simple one is a
classic methodological gap.** That is what we did. Work that does take the trouble
to compute the baseline often finds the claimed gap is smaller than it looks.

```
majority class on the test set : 14.3 %
uniform random                 :  8.3 %
```

**Three metrics give three readings of the same model.** Accuracy is dominated by
frequent classes. Macro F1 punishes a model that ignores rare ones. The
hierarchical metric tells a near miss from an absurd one.

**The split has to be stratified.** With 8 examples for the `temps` class, a global
random split can produce a test set with no example of that class at all, and its
score becomes undefined.

---

## What is still open

The learning curve is still rising at 80 examples. It is not the model that
plateaus, it is the amount of data.

I now have the baseline figure my coursework was missing. What I do not have is
anything to compare it with. Until I have measured a method that reuses knowledge
acquired elsewhere, I do not know what that 14.3 % is worth.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus is written into the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount. Two
years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my coursework are quoted, not erased.** Where a result I handed in
was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I write down what I found.

**The code is commented in French.**

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
