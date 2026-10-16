# Development guide

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Everyday commands

```bash
pytest                              # all tests
pytest --cov=tide                   # with coverage
ruff check src tests benchmarks     # lint
ruff format src tests benchmarks    # auto-format
mkdocs serve                        # docs at http://127.0.0.1:8000
python benchmarks/bench_fit.py      # performance
```

CI (`.github/workflows/ci.yml`) runs lint, tests on Windows / macOS / Linux × Python
3.10–3.13, a job with the oldest supported dependencies (`ci/requirements-min.txt`), and a
build check. A pull request should be green before merging.

## Tests

- `tests/test_golden.py` — v0.6 must give the same bins as v0.5.0. Fixtures come from
  `tests/golden/generate.py`, run once against the v0.5.0 code; do not regenerate them
  unless the algorithm is meant to change.
- `tests/test_properties.py` — hypothesis generates random data and parameters and checks
  every hard constraint.
- `tests/test_estimator.py` — public API and scikit-learn integration.

## Releasing

One-time setup:
1. On https://pypi.org (and https://test.pypi.org) → *Your projects* → *Publishing* → add a
   **trusted publisher**: owner `leonidgarin`, repository `TIDE`, workflow `release.yml`,
   environment `pypi` (`testpypi` on TestPyPI).
2. On GitHub → *Settings* → *Environments*, create `pypi` and `testpypi`.

Each release:
1. Update `__version__` in `src/tide/__init__.py` and the date in `CHANGELOG.md`.
2. Optional dry run: *Actions* → *Release* → *Run workflow* → `testpypi`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tide-binning`.
3. `git tag v0.6.0 && git push origin v0.6.0` — the workflow checks that the tag matches the
   version, builds, and publishes to PyPI. No tokens needed.
