# IMC Prosperity 2026

This repository contains my IMC Prosperity challenge work.

## Contents

- `ROUND1/`: Round 1 trader versions, data, and results.
- `ROUND2/`: Round 2 trader versions, data, and research scripts.
- `imc-prosperity-4-backtester-master/`: third-party backtester, not my code.
  See below.

The main submission files are the `trader.py` and `trader_v*.py` files in each round folder.

## Tests

[![CI](https://github.com/krrishapatel/IMC-Prosperity2026/actions/workflows/ci.yml/badge.svg)](https://github.com/krrishapatel/IMC-Prosperity2026/actions/workflows/ci.yml)

```bash
pip install -r requirements.txt
python -m pytest
```

Every `trader*.py` in every round is imported and checked against the
competition's contract: a class named `Trader` with `run(self, state)`. That is
all the upload validates, and a file with a syntax error or a renamed method
looked exactly like a working one until submission.

The traders do `from datamodel import ...`, which comes from the vendored
backtester rather than from PyPI, so `requirements.txt` declares the
`jsonpickle` that `datamodel` needs and the root `conftest.py` puts the vendored
package directory on the path.

## Third-party code

`imc-prosperity-4-backtester-master/` is a copy of the IMC Prosperity 4
backtester, vendored here so the traders above can be run against the round data
without a separate install. It is not my work. It is
[nabayansaha/imc-prosperity-4-backtester](https://github.com/nabayansaha/imc-prosperity-4-backtester),
itself based on
[jmerle/imc-prosperity-3-backtester](https://github.com/jmerle/imc-prosperity-3-backtester),
and it keeps its own MIT license and copyright in
`imc-prosperity-4-backtester-master/LICENSE`.

## License

MIT, for the files written for this repository. The vendored backtester above is
covered by its own license, not by this one.
