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
            [
                [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in state.own_trades.values()
                for t in arr
            ],
            [
                [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in state.market_trades.values()
                for t in arr
            ],
            state.position,
            [
                state.observations.plainValueObservations,
                {
                    product: [
                        o.bidPrice,
                        o.askPrice,
                        o.transportFees,
                        o.exportTariff,
                        o.importTariff,
                        o.sugarPrice,
                        o.sunlightIndex,
                    ]
                    for product, o in state.observations.conversionObservations.items()
                },
            ],
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

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

PEPPER_TREND_PER_TS = 0.001
PEPPER_ANCHORS = (10_000.0, 11_000.0, 12_000.0)
PEPPER_BUY_EDGE = 3.0
PEPPER_EARLY_AGG_TS = 10_000
PEPPER_EARLY_MARKUP = 8.0
PEPPER_SLOPE_ALPHA = 0.15
PEPPER_MIN_DT = 5
PEPPER_STOP_SLOPE = 0.0002
PEPPER_SAFE_CHUNK = 8
PEPPER_NORMAL_CHUNK = 20


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
        if "pepper_slope" not in td:
            td["pepper_slope"] = PEPPER_TREND_PER_TS

        if mid is not None and "pepper_last_mid" in td and "pepper_last_ts" in td:
            dt = int(timestamp - td["pepper_last_ts"])
            if dt >= PEPPER_MIN_DT:
                inst_slope = (mid - float(td["pepper_last_mid"])) / dt
                inst_slope = max(-0.005, min(0.005, inst_slope))
                prev = float(td["pepper_slope"])
                td["pepper_slope"] = (1.0 - PEPPER_SLOPE_ALPHA) * prev + PEPPER_SLOPE_ALPHA * inst_slope

        if mid is not None:
            td["pepper_last_mid"] = mid
            td["pepper_last_ts"] = timestamp

        slope = float(td["pepper_slope"])
        return float(td["pepper_anchor"]) + slope * timestamp

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {
            "ASH_COATED_OSMIUM": [],
            "INTARIAN_PEPPER_ROOT": [],
        }

        # Intentionally skip Osmium because current alpha is weak/negative.
        osm_pos = state.position.get("ASH_COATED_OSMIUM", 0)
        logger.print(f"ASH_COATED_OSMIUM pos={osm_pos} mode=OFF")

        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        if pepper_depth is not None:
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            fair = self._pepper_fair(pepper_depth, state.timestamp, td)
            buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", pos)
            orders: list[Order] = []

            # Build long quickly, especially early in the day.
            edge = PEPPER_EARLY_MARKUP if state.timestamp <= PEPPER_EARLY_AGG_TS else PEPPER_BUY_EDGE
            slope = float(td.get("pepper_slope", PEPPER_TREND_PER_TS))
            chunk_cap = PEPPER_SAFE_CHUNK if slope < PEPPER_STOP_SLOPE else PEPPER_NORMAL_CHUNK
            for ask in sorted(pepper_depth.sell_orders.keys()):
                if buy_cap <= 0:
                    break
                if ask > fair + edge:
                    break
                qty = min(-pepper_depth.sell_orders[ask], buy_cap, chunk_cap)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                    buy_cap -= qty

            result["INTARIAN_PEPPER_ROOT"] = orders
            logger.print(
                f"INTARIAN_PEPPER_ROOT pos={pos} fair={fair:.1f} edge={edge:.1f} "
                f"slope={slope:.6f} buys={len(orders)}"
            )

        td_out = json.dumps(self._to_native(td))
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
