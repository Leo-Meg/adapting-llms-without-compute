# Adapting a model without data or compute

The baseline to beat, computed first. Features and logistic regression, written in NumPy.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 2.** I am doing this work in stages, each in its
> own folder. I publish as I go rather than once everything is finished.

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

## Published versions

| | Folder | Contents | Tests |
|---|---|---|---:|
| **1** | `1.adaptation_python_projet` | The baseline to beat, features and logistic regression | 26 |
| **2** | `2.adaptation_python_projet` | Transfer learning, three regimes | 14 |

That is **40 tests** in total. Each folder contains everything the previous
one had, plus one step.

---

## Running the latest version

```bash
cd 2.adaptation_python_projet
python -m src.transfert
python -m tests.test_transfert
```

---

## What is still open

Frozen representations cost 396 parameters and plateau at 0.279. Full
fine-tuning costs 24,492 of them to reach 0.407.

Between the two, I know of no intermediate regime.

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

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
