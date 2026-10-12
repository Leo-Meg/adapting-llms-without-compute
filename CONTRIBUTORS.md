# Contributors

Two people, one portfolio. We did part of the original coursework together, and
we rebuild all of it together.

## Léo Mégret — [@Leo-Meg](https://github.com/Leo-Meg)

MSc Computational Linguistics, Université Paris Cité.

Writes the implementations in `src/` and the detailed technical notes in
`notes/`, which are in French because that is the language they were thought out
in. Maintains `README_FR.md`.

## Haeeul Hwang — [@hhaeeull](https://github.com/hhaeeull)

MSc Computational Linguistics, Université Paris Cité.

Writes the test suites in `tests/`, the English `README.md` and the Korean
`README_KO.md`, and the GitHub Actions workflow that runs every test on every
push. Reviews each step before it goes out.

The M2 lab this whole repository grew out of is ours jointly: the
comparison between a LoRA fine-tune and in-context learning, down to the
two hand-picked examples we never questioned.

## How we work

Each step goes out in three pushes, so neither of us owns a step alone:

| | what it lands | who |
|---|---|---|
| **a** | the module and its French note | Léo |
| **b** | the tests for it | Haeeul |
| **c** | the English and Korean documentation | Haeeul |

Whoever pushes updates their own language and does not touch the other's file.
The scripts enforce it: a push refuses to stage the other author's README, even
if it changed on disk. A translation can therefore sit one step behind, and the
banner at the top of each README says so. That is the honest state of the work,
not a defect.

The turn order for all six repositories of the portfolio lives outside the
repositories themselves, in a shared notebook.
