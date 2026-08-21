"""Put the vendored backtester's package directory on sys.path for the tests.

Every trader starts with `from datamodel import ...`, which is a bare top-level
import. datamodel lives inside the vendored backtester package, so without this
a plain `python -m pytest` fails on the import of the first trader rather than
on anything about the trader itself.
"""

import sys
from pathlib import Path

DATAMODEL_DIR = Path(__file__).resolve().parent / "imc-prosperity-4-backtester-master" / "prosperity4bt"

if str(DATAMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(DATAMODEL_DIR))
