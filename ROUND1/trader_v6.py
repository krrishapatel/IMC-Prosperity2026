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
PEPPER_TREND_PER_TS = 0.001
PEPPER_ANCHORS = (10_000.0, 11_000.0, 12_000.0)

# Strict Osmium take-only edges
OSM_BUY_LEVEL = 9998
OSM_SELL_LEVEL = 10002
OSM_ORDER_CHUNK = 12

# More aggressive Pepper accumulation than v3
PEPPER_TARGET_POS = 80
PEPPER_BOOTSTRAP_TS = 2000
PEPPER_BOOTSTRAP_EDGE = 40.0
PEPPER_NORMAL_EDGE = 8.0
PEPPER_ORDER_CHUNK = 80
PEPPER_HARVEST_EDGE = 30.0
PEPPER_HARVEST_FLOOR = 60
PEPPER_HARVEST_CHUNK = 12


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

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict[str, Any]) -> float:
        mid = self._book_mid(depth)
        if "pepper_anchor" not in td:
            if mid is not None:
                base = mid - PEPPER_TREND_PER_TS * timestamp
                td["pepper_anchor"] = min(PEPPER_ANCHORS, key=lambda a: abs(a - base))
            else:
                td["pepper_anchor"] = 11_000.0
        return float(td["pepper_anchor"]) + PEPPER_TREND_PER_TS * timestamp

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {"ASH_COATED_OSMIUM": [], "INTARIAN_PEPPER_ROOT": []}

        # Osmium strict take-only
        osm_depth = state.order_depths.get("ASH_COATED_OSMIUM")
        if osm_depth is not None:
            pos = state.position.get("ASH_COATED_OSMIUM", 0)
            buy_cap = self._allowable_buy("ASH_COATED_OSMIUM", pos)
            sell_cap = max(0, POSITION_LIMITS["ASH_COATED_OSMIUM"] + pos)
            osm_orders: list[Order] = []

            for ask in sorted(osm_depth.sell_orders.keys()):
                if ask > OSM_BUY_LEVEL or buy_cap <= 0:
                    break
                qty = min(-osm_depth.sell_orders[ask], buy_cap, OSM_ORDER_CHUNK)
                if qty > 0:
                    osm_orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                    buy_cap -= qty
                    pos += qty

            for bid in sorted(osm_depth.buy_orders.keys(), reverse=True):
                if bid < OSM_SELL_LEVEL or sell_cap <= 0:
                    break
                qty = min(osm_depth.buy_orders[bid], sell_cap, OSM_ORDER_CHUNK)
                if qty > 0:
                    osm_orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    sell_cap -= qty
                    pos -= qty

            result["ASH_COATED_OSMIUM"] = osm_orders

        # Pepper fast accumulation and hold
        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if pepper_depth is not None:
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            fair = self._pepper_fair(pepper_depth, state.timestamp, td)
            buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", pos)
            orders: list[Order] = []

            if buy_cap > 0 and pos < PEPPER_TARGET_POS:
                edge = PEPPER_BOOTSTRAP_EDGE if state.timestamp <= PEPPER_BOOTSTRAP_TS else PEPPER_NORMAL_EDGE
                for ask in sorted(pepper_depth.sell_orders.keys()):
                    if buy_cap <= 0 or pos >= PEPPER_TARGET_POS:
                        break
                    if ask > fair + edge:
                        break
                    qty = min(-pepper_depth.sell_orders[ask], buy_cap, PEPPER_TARGET_POS - pos, PEPPER_ORDER_CHUNK)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                        buy_cap -= qty
                        pos += qty

            # tactical harvest
            if pos > PEPPER_HARVEST_FLOOR:
                sell_cap = max(0, POSITION_LIMITS["INTARIAN_PEPPER_ROOT"] + pos)
                for bid in sorted(pepper_depth.buy_orders.keys(), reverse=True):
                    if pos <= PEPPER_HARVEST_FLOOR or sell_cap <= 0:
                        break
                    if bid < fair + PEPPER_HARVEST_EDGE:
                        break
                    qty = min(pepper_depth.buy_orders[bid], sell_cap, pos - PEPPER_HARVEST_FLOOR, PEPPER_HARVEST_CHUNK)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                        pos -= qty
                        sell_cap -= qty

            result["INTARIAN_PEPPER_ROOT"] = orders
            logger.print(
                f"PEPPER pos={state.position.get('INTARIAN_PEPPER_ROOT', 0)} fair={fair:.1f} orders={len(orders)}"
            )

        td_out = json.dumps(self._to_native(td))
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
