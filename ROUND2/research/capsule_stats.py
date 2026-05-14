#!/usr/bin/env python3
"""
Quick read-only stats on ROUND2 data capsule CSVs (prices_round_2_day_*.csv).

Findings that motivated trader_v7:
  - Pepper: best_ask < model_fair only ~1.5% of ticks -> "ask < fair" strategies
    barely trade; must lift the full ask stack to reach +80.
  - Pepper: mid tracks (anchor + 0.001*ts) within ~±1.5 in sample -> fair model ok.
  - Osmium: mid mean ~10000–10002, spread ~16–17 -> makers should hug microprice,
    not an overly sticky 10000 fair.
Run: python3 capsule_stats.py  (from ROUND2/research or pass ROUND2/data path)
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def load_product_rows(path: Path, product: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("product") != product:
                continue
            if (
                not row.get("ask_price_1", "").strip()
                or not row.get("bid_price_1", "").strip()
                or not row.get("mid_price", "").strip()
            ):
                continue
            out.append(row)
    return out


def pepper_ask_below_fair(rows: list[dict[str, str]]) -> tuple[float, float]:
    r0 = rows[0]
    t0 = int(r0["timestamp"])
    mid0 = float(r0["mid_price"])
    anchor = round((mid0 - 0.001 * t0) / 1000.0) * 1000.0
    below = 0
    for r in rows:
        t = int(r["timestamp"])
        fair = anchor + 0.001 * t
        ba = int(r["ask_price_1"])
        if ba < fair:
            below += 1
    return anchor, below / max(1, len(rows))


def osmium_mid_spread(rows: list[dict[str, str]]) -> tuple[float, float]:
    mids = [float(r["mid_price"]) for r in rows[:: max(1, len(rows) // 2000)]]
    spreads = []
    for r in rows[:: max(1, len(rows) // 2000)]:
        spreads.append(int(r["ask_price_1"]) - int(r["bid_price_1"]))
    return statistics.mean(mids), statistics.mean(spreads)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])

    for day in (-1, 0, 1):
        p = root / f"prices_round_2_day_{day}.csv"
        if not p.exists():
            print(f"missing {p}")
            continue
        pep = load_product_rows(p, "INTARIAN_PEPPER_ROOT")
        osm = load_product_rows(p, "ASH_COATED_OSMIUM")
        a, frac = pepper_ask_below_fair(pep)
        mm, sp = osmium_mid_spread(osm)
        print(f"day {day}: pepper rows={len(pep)} anchor~{a:.0f}  P(ask<fair)={frac*100:.2f}%")
        print(f"         osmium mid~{mm:.2f}  mean_spread~{sp:.1f}")


if __name__ == "__main__":
    main()
