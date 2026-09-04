# Contributing

Thanks for looking at **Local FEM Solver**. This is a research codebase; contributions,
issues, and questions are welcome.

## Setup

```powershell
pip install -r requirements.txt
```

Bulk meshes, checkpoints, and COMSOL `.mph` files are not in this repository — see
[`docs/PUBLISHING.md`](docs/PUBLISHING.md) for what's tracked vs. local-only, and the
[README](README.md#try-it-now-no-install) if you just want to run the customer Predict app
without a dev setup at all.

## Tests

```powershell
pytest src/tests/
```

## Before opening a PR

- Keep the change scoped — this repo favors small, reviewable diffs over sweeping refactors.
- If you're touching a supported launcher or adding one, update
  [`scripts/README.md`](scripts/README.md) to match.
- If you're adding a doc, link it from the README's Documentation table or
  [`docs/README.md`](docs/README.md) so it's discoverable, or note in the PR why it's
  intentionally internal-only.
- Run the relevant tests in `src/tests/` before submitting.

## Deeper context

[`AGENTS.md`](AGENTS.md) is the working cheat sheet for the canonical model stacks,
checkpoints, and gates this repo scores against — useful background before a non-trivial
change, even though it's written for quick reference rather than onboarding.

## Questions / issues

Open a GitHub issue.
