import json
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str
    ) -> None:
        base_length = len(
            self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
            {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in state.own_trades.values() for t in arr],
            [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in state.market_trades.values() for t in arr],
            state.position,
            [state.observations.plainValueObservations, {
                product: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                for product, o in state.observations.conversionObservations.items()
            }],
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

POSITION_LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

# Osmium dynamic fair + MM engine
OSM_BASE = 10_000.0
OSM_EMA_ALPHA = 0.12
OSM_MICRO_W = 0.75
OSM_IMB_W = 2.2
OSM_MR_TAKE = 0.35
OSM_MAKE_SIZE = 24
OSM_MIN_SPREAD_TO_MAKE = 2
OSM_INV_SKEW = 1.4
OSM_SIGNAL_GATE = 4.0

# Pepper robust directional core
PEP_TREND_PER_TS = 0.001
PEP_ANCHORS = (10_000.0, 11_000.0, 12_000.0)
PEP_TARGET = 80
PEP_BOOT_TS = 2000
PEP_BOOT_EDGE = 40.0
PEP_NORM_EDGE = 8.0
PEP_CHUNK = 80
PEP_HARVEST_EDGE = 30.0
PEP_HARVEST_FLOOR = 60
PEP_HARVEST_CHUNK = 12


class Trader:
    def _to_native(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {self._to_native(k): self._to_native(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_native(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_native(v) for v in value]
        if hasattr(value, "item"):
            try:
                return self._to_native(value.item())
            except Exception:
                pass
        return value

    def _allow_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _allow_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _best(self, d: OrderDepth) -> tuple[int | None, int | None]:
        bid = max(d.buy_orders.keys()) if d.buy_orders else None
        ask = min(d.sell_orders.keys()) if d.sell_orders else None
        return bid, ask

    def _mid(self, d: OrderDepth) -> float | None:
        bid, ask = self._best(d)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _micro(self, d: OrderDepth) -> float | None:
        bid, ask = self._best(d)
        if bid is None or ask is None:
            return None
        bv = max(1, d.buy_orders[bid])
        av = max(1, -d.sell_orders[ask])
        return (bid * av + ask * bv) / (bv + av)

    def _imbalance(self, d: OrderDepth) -> float:
        b = sorted(d.buy_orders.items(), reverse=True)[:4]
        a = sorted(d.sell_orders.items())[:4]
        bv = sum(max(0, v) / (i + 1) for i, (_, v) in enumerate(b))
        av = sum(abs(v) / (i + 1) for i, (_, v) in enumerate(a))
        tot = bv + av
        if tot <= 0:
            return 0.0
        return (bv - av) / tot

    def _pepper_fair(self, d: OrderDepth, ts: int, td: dict[str, Any]) -> float:
        if "pep_anchor" not in td:
            m = self._mid(d)
            if m is None:
                td["pep_anchor"] = 11_000.0
            else:
                base = m - PEP_TREND_PER_TS * ts
                td["pep_anchor"] = min(PEP_ANCHORS, key=lambda x: abs(x - base))
        return float(td["pep_anchor"]) + PEP_TREND_PER_TS * ts

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        res: dict[Symbol, list[Order]] = {"ASH_COATED_OSMIUM": [], "INTARIAN_PEPPER_ROOT": []}

        # -------- Osmium --------
        od = state.order_depths.get("ASH_COATED_OSMIUM")
        if od is not None and od.buy_orders and od.sell_orders:
            pos = state.position.get("ASH_COATED_OSMIUM", 0)
            bid, ask = self._best(od)
            assert bid is not None and ask is not None
            spread = ask - bid
            mid = self._mid(od)
            micro = self._micro(od)
            imb = self._imbalance(od)

            prev_ema = float(td.get("osm_ema", OSM_BASE))
            ema = prev_ema if mid is None else OSM_EMA_ALPHA * mid + (1.0 - OSM_EMA_ALPHA) * prev_ema
            td["osm_ema"] = ema

            fair = 0.55 * OSM_BASE + 0.45 * ema
            if micro is not None and mid is not None:
                fair += OSM_MICRO_W * (micro - mid)
            fair += OSM_IMB_W * imb

            edge = max(1.0, OSM_MR_TAKE * spread)

            # Take
            buy_cap = self._allow_buy("ASH_COATED_OSMIUM", pos)
            for p in sorted(od.sell_orders.keys()):
                if buy_cap <= 0 or p >= fair - edge:
                    break
                q = min(-od.sell_orders[p], buy_cap)
                if q > 0:
                    res["ASH_COATED_OSMIUM"].append(Order("ASH_COATED_OSMIUM", p, q))
                    pos += q
                    buy_cap -= q

            sell_cap = self._allow_sell("ASH_COATED_OSMIUM", pos)
            for p in sorted(od.buy_orders.keys(), reverse=True):
                if sell_cap <= 0 or p <= fair + edge:
                    break
                q = min(od.buy_orders[p], sell_cap)
                if q > 0:
                    res["ASH_COATED_OSMIUM"].append(Order("ASH_COATED_OSMIUM", p, -q))
                    pos -= q
                    sell_cap -= q

            # Make
            signal = abs((mid - fair) if mid is not None else 0.0)
            if spread >= OSM_MIN_SPREAD_TO_MAKE and signal <= OSM_SIGNAL_GATE:
                skew = (pos / POSITION_LIMITS["ASH_COATED_OSMIUM"]) * OSM_INV_SKEW
                bq = min(bid + 1, int(fair - skew) - 1)
                aq = max(ask - 1, int(fair - skew) + 1)
                bs = max(0, min(self._allow_buy("ASH_COATED_OSMIUM", pos), int(OSM_MAKE_SIZE * (1.0 - max(0.0, skew)))))
                ss = max(0, min(self._allow_sell("ASH_COATED_OSMIUM", pos), int(OSM_MAKE_SIZE * (1.0 - max(0.0, -skew)))))
                if bs > 0 and bq < fair:
                    res["ASH_COATED_OSMIUM"].append(Order("ASH_COATED_OSMIUM", bq, bs))
                if ss > 0 and aq > fair:
                    res["ASH_COATED_OSMIUM"].append(Order("ASH_COATED_OSMIUM", aq, -ss))

            logger.print(f"OSM pos={state.position.get('ASH_COATED_OSMIUM',0)} fair={fair:.2f} n={len(res['ASH_COATED_OSMIUM'])}")

        # -------- Pepper --------
        pd = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if pd is not None and pd.sell_orders:
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            fair = self._pepper_fair(pd, state.timestamp, td)
            buy_cap = self._allow_buy("INTARIAN_PEPPER_ROOT", pos)
            orders: list[Order] = []

            if buy_cap > 0 and pos < PEP_TARGET:
                edge = PEP_BOOT_EDGE if state.timestamp <= PEP_BOOT_TS else PEP_NORM_EDGE
                for ask in sorted(pd.sell_orders.keys()):
                    if buy_cap <= 0 or pos >= PEP_TARGET or ask > fair + edge:
                        break
                    q = min(-pd.sell_orders[ask], buy_cap, PEP_TARGET - pos, PEP_CHUNK)
                    if q > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask, q))
                        pos += q
                        buy_cap -= q

            if pos > PEP_HARVEST_FLOOR:
                sell_cap = self._allow_sell("INTARIAN_PEPPER_ROOT", pos)
                for bid in sorted(pd.buy_orders.keys(), reverse=True):
                    if pos <= PEP_HARVEST_FLOOR or sell_cap <= 0 or bid < fair + PEP_HARVEST_EDGE:
                        break
                    q = min(pd.buy_orders[bid], sell_cap, pos - PEP_HARVEST_FLOOR, PEP_HARVEST_CHUNK)
                    if q > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -q))
                        pos -= q
                        sell_cap -= q

            res["INTARIAN_PEPPER_ROOT"] = orders
            logger.print(f"PEP pos={state.position.get('INTARIAN_PEPPER_ROOT',0)} fair={fair:.2f} n={len(orders)}")

        td_out = json.dumps(self._to_native(td))
        conversions = 0
        logger.flush(state, res, conversions, td_out)
        return res, conversions, td_out
