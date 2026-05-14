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

# Product baselines
OSM_BASELINE = 10_000.0
PEPPER_TREND_PER_TS = 0.001
PEPPER_ANCHORS = (10_000.0, 11_000.0, 12_000.0)

# EMA settings
EMA_ALPHA = {
    "ASH_COATED_OSMIUM": 0.10,
    "INTARIAN_PEPPER_ROOT": 0.08,
}

# Dynamic fair blending
BASELINE_WEIGHT = {
    "ASH_COATED_OSMIUM": 0.55,
    "INTARIAN_PEPPER_ROOT": 0.20,
}
MICRO_WEIGHT = {
    "ASH_COATED_OSMIUM": 0.55,
    "INTARIAN_PEPPER_ROOT": 0.35,
}
IMBALANCE_WEIGHT = {
    "ASH_COATED_OSMIUM": 1.6,
    "INTARIAN_PEPPER_ROOT": 0.8,
}

# Taking + making
TAKE_FRAC = {
    "ASH_COATED_OSMIUM": 0.30,
    "INTARIAN_PEPPER_ROOT": 0.22,
}
MIN_TAKE_EDGE = 1.0
MAKE_SIZE = {
    "ASH_COATED_OSMIUM": 20,
    "INTARIAN_PEPPER_ROOT": 18,
}
MIN_SPREAD_TO_MAKE = 2
INV_SKEW_K = {
    "ASH_COATED_OSMIUM": 1.1,
    "INTARIAN_PEPPER_ROOT": 0.8,
}

# Pepper long bias (keep profitable directional core)
PEPPER_CORE_TARGET = 50
PEPPER_MAX_TARGET = 80
PEPPER_EARLY_TS = 2000
PEPPER_EARLY_EXTRA_BUY = 4.0


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

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _best_bid_ask(self, depth: OrderDepth) -> tuple[int | None, int | None]:
        bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return bid, ask

    def _mid(self, depth: OrderDepth) -> float | None:
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _microprice(self, depth: OrderDepth) -> float | None:
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        bid_vol = max(1, depth.buy_orders[bid])
        ask_vol = max(1, -depth.sell_orders[ask])
        return (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)

    def _spread(self, depth: OrderDepth) -> float:
        bid, ask = self._best_bid_ask(depth)
        if bid is None or ask is None:
            return 6.0
        return float(ask - bid)

    def _imbalance(self, depth: OrderDepth) -> float:
        # Top-of-book weighted imbalance
        bid_items = sorted(depth.buy_orders.items(), reverse=True)
        ask_items = sorted(depth.sell_orders.items())
        b = 0.0
        a = 0.0
        for i, (_, v) in enumerate(bid_items[:4]):
            b += max(0, v) / (1.0 + i)
        for i, (_, v) in enumerate(ask_items[:4]):
            a += abs(v) / (1.0 + i)
        tot = a + b
        if tot <= 0:
            return 0.0
        return (b - a) / tot

    def _pepper_anchor(self, depth: OrderDepth, timestamp: int, td: dict[str, Any]) -> float:
        if "pepper_anchor" in td:
            return float(td["pepper_anchor"])
        mid = self._mid(depth)
        if mid is None:
            td["pepper_anchor"] = 11_000.0
        else:
            base = mid - PEPPER_TREND_PER_TS * timestamp
            td["pepper_anchor"] = min(PEPPER_ANCHORS, key=lambda a: abs(a - base))
        return float(td["pepper_anchor"])

    def _dynamic_fair(self, product: str, depth: OrderDepth, timestamp: int, td: dict[str, Any]) -> float:
        mid = self._mid(depth)
        micro = self._microprice(depth)
        spread = self._spread(depth)
        imb = self._imbalance(depth)

        if product == "ASH_COATED_OSMIUM":
            baseline = OSM_BASELINE
        else:
            baseline = self._pepper_anchor(depth, timestamp, td) + PEPPER_TREND_PER_TS * timestamp

        # Update EMA on observed mid
        ema_key = f"{product}_ema"
        prev_ema = float(td.get(ema_key, baseline))
        if mid is None:
            ema = prev_ema
        else:
            alpha = EMA_ALPHA[product]
            ema = alpha * mid + (1.0 - alpha) * prev_ema
        td[ema_key] = ema

        fair = BASELINE_WEIGHT[product] * baseline + (1.0 - BASELINE_WEIGHT[product]) * ema
        if micro is not None and mid is not None:
            fair += MICRO_WEIGHT[product] * (micro - mid)
        fair += IMBALANCE_WEIGHT[product] * imb * max(1.0, spread / 2.0)
        return fair

    def _take_orders(self, product: str, depth: OrderDepth, fair: float, position: int) -> tuple[list[Order], int]:
        orders: list[Order] = []
        spread = self._spread(depth)
        edge = max(MIN_TAKE_EDGE, TAKE_FRAC[product] * spread)

        buy_cap = self._allowable_buy(product, position)
        for ask in sorted(depth.sell_orders.keys()):
            if buy_cap <= 0 or ask >= fair - edge:
                break
            qty = min(-depth.sell_orders[ask], buy_cap)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                position += qty
                buy_cap -= qty

        sell_cap = self._allowable_sell(product, position)
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if sell_cap <= 0 or bid <= fair + edge:
                break
            qty = min(depth.buy_orders[bid], sell_cap)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                position -= qty
                sell_cap -= qty

        return orders, position

    def _make_orders(self, product: str, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders
        spread = best_ask - best_bid
        if spread < MIN_SPREAD_TO_MAKE:
            return orders

        limit = POSITION_LIMITS[product]
        skew = position / limit
        fair_shifted = fair - INV_SKEW_K[product] * skew

        buy_cap = self._allowable_buy(product, position)
        sell_cap = self._allowable_sell(product, position)

        base = MAKE_SIZE[product]
        buy_size = max(0, min(buy_cap, int(base * (1.0 - max(0.0, skew)))))
        sell_size = max(0, min(sell_cap, int(base * (1.0 - max(0.0, -skew)))))

        bid_quote = min(best_bid + 1, int(fair_shifted) - 1)
        ask_quote = max(best_ask - 1, int(fair_shifted) + 1)

        if buy_size > 0 and bid_quote > 0 and bid_quote < fair_shifted:
            orders.append(Order(product, bid_quote, buy_size))
        if sell_size > 0 and ask_quote > 0 and ask_quote > fair_shifted:
            orders.append(Order(product, ask_quote, -sell_size))

        return orders

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {"ASH_COATED_OSMIUM": [], "INTARIAN_PEPPER_ROOT": []}

        for product in ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"):
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            position = state.position.get(product, 0)
            fair = self._dynamic_fair(product, depth, state.timestamp, td)

            take_orders, post_pos = self._take_orders(product, depth, fair, position)
            make_orders = self._make_orders(product, depth, fair, post_pos)

            # Pepper long-bias overlay for robustness in known uptrend
            if product == "INTARIAN_PEPPER_ROOT":
                buy_cap = self._allowable_buy(product, post_pos)
                target = PEPPER_CORE_TARGET
                if state.timestamp <= PEPPER_EARLY_TS:
                    target = min(PEPPER_MAX_TARGET, PEPPER_CORE_TARGET + 15)
                extra_orders: list[Order] = []
                if post_pos < target and buy_cap > 0 and depth.sell_orders:
                    best_ask = min(depth.sell_orders.keys())
                    if best_ask <= fair + PEPPER_EARLY_EXTRA_BUY:
                        qty = min(-depth.sell_orders[best_ask], buy_cap, target - post_pos)
                        if qty > 0:
                            extra_orders.append(Order(product, best_ask, qty))
                result[product] = take_orders + make_orders + extra_orders
                logger.print(f"{product} pos={position} fair={fair:.1f} t={len(take_orders)} m={len(make_orders)} x={len(extra_orders)}")
            else:
                result[product] = take_orders + make_orders
                logger.print(f"{product} pos={position} fair={fair:.1f} t={len(take_orders)} m={len(make_orders)}")

        td_out = json.dumps(self._to_native(td))
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
