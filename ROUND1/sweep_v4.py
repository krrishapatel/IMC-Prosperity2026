import itertools
import re
import subprocess
from pathlib import Path

BASE = Path("/Users/krrishapatel/Downloads/IMC Prosperity/ROUND1/trader_v4.py")
TMP = Path("/Users/krrishapatel/Downloads/IMC Prosperity/ROUND1/trader_v4_tmp.py")
BT = "/Users/krrishapatel/Downloads/IMC Prosperity/.venv/bin/prosperity4btest"


def set_const(src: str, name: str, value: str) -> str:
    pattern = rf"^{name}\s*=\s*.*$"
    return re.sub(pattern, f"{name} = {value}", src, flags=re.MULTILINE)


def backtest(path: Path) -> int:
    cmd = [BT, str(path), "1", "--no-progress", "--no-out"]
    out = subprocess.check_output(cmd, text=True)
    m = re.search(r"Total profit:\s*([-\d,]+)\s*$", out, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not parse total profit")
    return int(m.group(1).replace(",", ""))


def main() -> None:
    base = BASE.read_text()
    best = None
    best_cfg = None

    grid = itertools.product(
        [9991, 9992, 9993],         # OSM_BUY_LEVEL
        [10007, 10008, 10009],      # OSM_SELL_LEVEL
        [8, 12, 16],                # OSM_ORDER_CHUNK
        [2000, 5000, 8000],         # PEPPER_BOOTSTRAP_TS
        [20.0, 30.0, 40.0],         # PEPPER_BOOTSTRAP_EDGE
        [4.0, 7.0, 10.0],           # PEPPER_NORMAL_EDGE
    )

    tested = 0
    for buy_lvl, sell_lvl, osm_chunk, boot_ts, boot_edge, norm_edge in grid:
        src = base
        src = set_const(src, "OSM_BUY_LEVEL", str(buy_lvl))
        src = set_const(src, "OSM_SELL_LEVEL", str(sell_lvl))
        src = set_const(src, "OSM_ORDER_CHUNK", str(osm_chunk))
        src = set_const(src, "PEPPER_BOOTSTRAP_TS", str(boot_ts))
        src = set_const(src, "PEPPER_BOOTSTRAP_EDGE", str(boot_edge))
        src = set_const(src, "PEPPER_NORMAL_EDGE", str(norm_edge))
        TMP.write_text(src)
        pnl = backtest(TMP)
        tested += 1
        if best is None or pnl > best:
            best = pnl
            best_cfg = (buy_lvl, sell_lvl, osm_chunk, boot_ts, boot_edge, norm_edge)
            print("NEW BEST", best, best_cfg)

    print("TESTED", tested)
    print("BEST", best, best_cfg)


if __name__ == "__main__":
    main()
