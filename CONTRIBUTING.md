# Contributing

Thanks for looking at **Local FEM Solver**. This is a research codebase; contributions,
issues, and questions are welcome.

## Setup

```powershell
pip install -r requirements.txt
pip install -e .
```

The editable install is what makes `import src...` and `import scripts...` work from
any directory. Without it, running a script by path fails on the first `src` import --
scripts no longer patch `sys.path` themselves.

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

[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) maps the source tree and the CLI entry
points, and [`docs/MODEL_NOMENCLATURE.md`](docs/MODEL_NOMENCLATURE.md) explains which model
name means what — both are worth a skim before a non-trivial change.

## Questions / issues

Open a GitHub issue.
