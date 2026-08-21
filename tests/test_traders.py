"""Every trader file must import and satisfy the competition's contract.

The submission contract is narrow: the file defines a class named `Trader` with a
`run(self, state)` method returning `(orders, conversions, traderData)`. Getting
that wrong means the upload is rejected or scores zero, and nothing in this repo
checked it. A trader with a syntax error or a renamed method looked exactly like
a working one until submission.

Run with:
    pip install -r requirements.txt
    python -m pytest

The root conftest.py is what puts the vendored `datamodel` on the path.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# `from datamodel import ...` at the top of every trader is a bare top-level
# import, satisfied by the vendored backtester's package directory being on the
# path. See the root conftest.py.
DATAMODEL_DIR = REPO_ROOT / "imc-prosperity-4-backtester-master" / "prosperity4bt"

TRADER_FILES = sorted(
    path
    for path in REPO_ROOT.glob("ROUND*/trader*.py")
    if path.is_file()
)


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.parent.name}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_traders_to_check():
    # Guards against the glob silently matching nothing, which would make every
    # other test in this file vacuously pass.
    assert len(TRADER_FILES) >= 20


def test_the_vendored_datamodel_is_present():
    assert (DATAMODEL_DIR / "datamodel.py").exists(), (
        "the traders import datamodel from the vendored backtester, so removing "
        "that directory breaks all of them"
    )


@pytest.mark.parametrize("path", TRADER_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
class TestEveryTrader:
    def test_it_imports(self, path):
        load(path)

    def test_it_defines_a_trader_class(self, path):
        assert hasattr(load(path), "Trader"), f"{path.name} has no Trader class"

    def test_run_takes_the_state(self, path):
        run = load(path).Trader.run
        parameters = list(inspect.signature(run).parameters)

        # self plus the TradingState. The competition calls run(state) once per
        # timestamp, so any other arity fails at upload.
        assert parameters[:1] == ["self"]
        assert len(parameters) == 2, f"{path.name}: run{inspect.signature(run)}"


def test_the_main_submission_of_each_round_exists():
    # trader.py is the file that gets uploaded; the trader_v*.py files are the
    # history behind it.
    for round_dir in sorted(REPO_ROOT.glob("ROUND*")):
        assert (round_dir / "trader.py").exists(), f"{round_dir.name} has no trader.py"
