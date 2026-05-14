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

OSMIUM_HARD_FAIR = 10_000.0
OSMIUM_EWM_ALPHA = 0.15
OSMIUM_EWM_BLEND = 0.90
OSMIUM_TAKE_FRAC = 0.15
OSMIUM_MAKE_SIZE = 20

PEPPER_TREND_PER_TS = 0.001
PEPPER_ANCHORS = (10_000.0, 11_000.0, 12_000.0)
PEPPER_TAKE_EDGE = 1.5
PEPPER_MAKE_SIZE = 16
PEPPER_SLOPE_ALPHA = 0.12
PEPPER_BASE_ALPHA = 0.10
PEPPER_MIN_SLOPE_DT = 5.0
PEPPER_ENTRY_BUFFER = 1.5
PEPPER_BUY_CHUNK = 20

MIN_TAKE_EDGE = 1.5
SKEW_POWER = 1.5
SKEW_CUTOFF = 0.75
CLEAR_TICKS = 1.0

IMBALANCE_K = {
    "ASH_COATED_OSMIUM": 0.35,
    "INTARIAN_PEPPER_ROOT": 0.0,
}
TOXICITY_DECAY = 0.85
TOXICITY_HIT = 1.0
TOXICITY_THRESHOLD = 1.5
TOXICITY_COOLDOWN_TICKS = 8
TOXICITY_SIZE_MULT = 0.45


class Trader:
    def _to_native(self, value: Any) -> Any:
        """
        Recursively convert numpy/scalar-like values to JSON-safe Python natives.
        """
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

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def _spread(self, depth: OrderDepth) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return 13.0
        return float(min(depth.sell_orders.keys()) - max(depth.buy_orders.keys()))

    def _imbalance(self, depth: OrderDepth) -> float:
        """
        Normalized book pressure in [-1, 1].
        Positive means bid side is heavier.
        """
        # Weight top levels more (liquidity walls near top carry more signal).
        bid_items = sorted(depth.buy_orders.items(), reverse=True)
        ask_items = sorted(depth.sell_orders.items())

        bid_vol = 0.0
        ask_vol = 0.0
        for i, (_, v) in enumerate(bid_items):
            w = 1.0 / (1.0 + i)
            bid_vol += w * max(0, v)
        for i, (_, v) in enumerate(ask_items):
            w = 1.0 / (1.0 + i)
            ask_vol += w * abs(v)

        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _osmium_fair(self, depth: OrderDepth, td: dict[str, Any]) -> float:
        mid = self._book_mid(depth)
        if mid is None or not (9_000 < mid < 11_000):
            return float(td.get("osm_ewm", OSMIUM_HARD_FAIR))
        prev = float(td.get("osm_ewm", OSMIUM_HARD_FAIR))
        ewm = OSMIUM_EWM_ALPHA * mid + (1.0 - OSMIUM_EWM_ALPHA) * prev
        td["osm_ewm"] = ewm
        core_fair = (1.0 - OSMIUM_EWM_BLEND) * OSMIUM_HARD_FAIR + OSMIUM_EWM_BLEND * ewm
        return core_fair + IMBALANCE_K["ASH_COATED_OSMIUM"] * self._imbalance(depth)

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict[str, Any]) -> float:
        mid = self._book_mid(depth)

        if "pep_slope" not in td:
            td["pep_slope"] = PEPPER_TREND_PER_TS
        if "pep_base" not in td:
            if mid is not None:
                est_base = mid - float(td["pep_slope"]) * timestamp
                td["pep_base"] = min(PEPPER_ANCHORS, key=lambda a: abs(a - est_base))
            else:
                td["pep_base"] = 11_000.0

        # Online slope update from observed price changes.
        if mid is not None and "pep_last_mid" in td and "pep_last_ts" in td:
            dt = timestamp - float(td["pep_last_ts"])
            if dt >= PEPPER_MIN_SLOPE_DT:
                inst_slope = (mid - float(td["pep_last_mid"])) / dt
                inst_slope = max(-0.01, min(0.01, inst_slope))
                prev_slope = float(td["pep_slope"])
                td["pep_slope"] = (1.0 - PEPPER_SLOPE_ALPHA) * prev_slope + PEPPER_SLOPE_ALPHA * inst_slope

        slope = float(td["pep_slope"])
        fair = float(td["pep_base"]) + slope * timestamp

        # Online base correction to absorb anchor/timestamp shifts.
        if mid is not None:
            observed_base = mid - slope * timestamp
            td["pep_base"] = (1.0 - PEPPER_BASE_ALPHA) * float(td["pep_base"]) + PEPPER_BASE_ALPHA * observed_base
            td["pep_last_mid"] = mid
            td["pep_last_ts"] = timestamp
            fair = float(td["pep_base"]) + slope * timestamp

        return fair + IMBALANCE_K["INTARIAN_PEPPER_ROOT"] * self._imbalance(depth)

    def _fair_value(self, product: str, timestamp: int, depth: OrderDepth, td: dict[str, Any]) -> float:
        if product == "ASH_COATED_OSMIUM":
            return self._osmium_fair(depth, td)
        if product == "INTARIAN_PEPPER_ROOT":
            return self._pepper_fair(depth, timestamp, td)
        return 0.0

    def _take_edge(self, product: str, depth: OrderDepth) -> float:
        if product == "INTARIAN_PEPPER_ROOT":
            return PEPPER_TAKE_EDGE
        return max(MIN_TAKE_EDGE, self._spread(depth) * OSMIUM_TAKE_FRAC)

    def _take_orders(
        self, product: str, depth: OrderDepth, fair: float, position: int
    ) -> tuple[list[Order], int]:
        orders: list[Order] = []
        edge = self._take_edge(product, depth)
        is_pepper = product == "INTARIAN_PEPPER_ROOT"

        buy_cap = self._allowable_buy(product, position)
        for ask in sorted(depth.sell_orders.keys()):
            if ask >= fair - edge or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order(product, ask, vol))
                position += vol
                buy_cap -= vol

        # Pepper is treated as structurally upward drifting in round 1.
        # Avoid opening/increasing short inventory via passive/active sells.
        if not is_pepper:
            sell_cap = self._allowable_sell(product, position)
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if bid <= fair + edge or sell_cap <= 0:
                    break
                vol = min(depth.buy_orders[bid], sell_cap)
                if vol > 0:
                    orders.append(Order(product, bid, -vol))
                    position -= vol
                    sell_cap -= vol

        return orders, position

    def _update_toxic_flow(
        self, product: str, state: TradingState, td: dict[str, Any], fair: float
    ) -> None:
        """
        Penalize maker size temporarily when recent fills are adverse.
        """
        key_score = f"{product}_tox_score"
        key_cd = f"{product}_tox_cd"

        score = float(td.get(key_score, 0.0)) * TOXICITY_DECAY
        cooldown = int(td.get(key_cd, 0))

        for tr in state.own_trades.get(product, []):
            if tr.timestamp != state.timestamp:
                continue
            if tr.buyer == "SUBMISSION":
                if fair < tr.price:
                    score += TOXICITY_HIT
            elif tr.seller == "SUBMISSION":
                if fair > tr.price:
                    score += TOXICITY_HIT

        if score >= TOXICITY_THRESHOLD:
            cooldown = TOXICITY_COOLDOWN_TICKS
            score = 0.0
        elif cooldown > 0:
            cooldown -= 1

        td[key_score] = score
        td[key_cd] = cooldown

    def _toxicity_size_mult(self, product: str, td: dict[str, Any]) -> float:
        return TOXICITY_SIZE_MULT if int(td.get(f"{product}_tox_cd", 0)) > 0 else 1.0

    def _make_orders(
        self, product: str, depth: OrderDepth, fair: float, position: int, td: dict[str, Any]
    ) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        limit = POSITION_LIMITS[product]
        skew = position / limit

        is_pepper = product == "INTARIAN_PEPPER_ROOT"
        base_size = PEPPER_MAKE_SIZE if is_pepper else OSMIUM_MAKE_SIZE
        base_size = max(1, int(base_size * self._toxicity_size_mult(product, td)))

        buy_scale = max(0.0, 1.0 - max(0.0, skew) ** SKEW_POWER) if skew <= SKEW_CUTOFF else 0.0
        buy_cap = self._allowable_buy(product, position)
        buy_size = min(buy_cap, max(1, int(base_size * buy_scale))) if buy_scale > 0 else 0

        bid_quote = min(best_bid + 1, int(fair) - 1)
        if skew < -0.5:
            bid_quote = min(best_bid + 1, int(fair))

        if buy_size > 0 and bid_quote > 0 and bid_quote < fair:
            orders.append(Order(product, bid_quote, buy_size))

        if not is_pepper:
            sell_scale = (
                max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER) if skew >= -SKEW_CUTOFF else 0.0
            )
            sell_cap = self._allowable_sell(product, position)
            sell_size = min(sell_cap, max(1, int(base_size * sell_scale))) if sell_scale > 0 else 0

            ask_quote = max(best_ask - 1, int(fair) + 1)
            if skew > 0.5:
                ask_quote = max(best_ask - 1, int(fair))

            if sell_size > 0 and ask_quote > 0 and ask_quote > fair:
                orders.append(Order(product, ask_quote, -sell_size))

        return orders

    def _clear_position(self, product: str, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        dynamic_ticks = max(CLEAR_TICKS, 0.5 * self._spread(depth))

        if position > 0:
            sell_cap = self._allowable_sell(product, position)
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if bid < fair - dynamic_ticks or sell_cap <= 0 or position <= 0:
                    break
                qty = min(depth.buy_orders[bid], position, sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    position -= qty
                    sell_cap -= qty

        if position < 0:
            buy_cap = self._allowable_buy(product, position)
            for ask in sorted(depth.sell_orders.keys()):
                if ask > fair + dynamic_ticks or buy_cap <= 0 or position >= 0:
                    break
                qty = min(-depth.sell_orders[ask], -position, buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    position += qty
                    buy_cap -= qty

        return orders

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {}

        for product in POSITION_LIMITS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            position = state.position.get(product, 0)
            fair = self._fair_value(product, state.timestamp, depth, td)
            edge = self._take_edge(product, depth)
            self._update_toxic_flow(product, state, td, fair)

            if product == "INTARIAN_PEPPER_ROOT":
                orders: list[Order] = []
                if depth.sell_orders:
                    best_ask = min(depth.sell_orders.keys())
                    buy_cap = self._allowable_buy(product, position)
                    if buy_cap > 0 and best_ask <= fair + PEPPER_ENTRY_BUFFER:
                        qty = min(-depth.sell_orders[best_ask], buy_cap, PEPPER_BUY_CHUNK)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))
                result[product] = orders
                logger.print(
                    f"{product} pos={position} fair={fair:.1f} edge={edge:.1f} "
                    f"t=0 c=0 m={len(orders)} tox_cd={int(td.get(f'{product}_tox_cd', 0))}"
                )
                continue

            take_orders, post_pos = self._take_orders(product, depth, fair, position)
            clear_orders = self._clear_position(product, depth, fair, post_pos)

            clear_delta = sum(o.quantity for o in clear_orders)
            make_position = post_pos + clear_delta
            make_orders = self._make_orders(product, depth, fair, make_position, td)

            result[product] = take_orders + clear_orders + make_orders

            logger.print(
                f"{product} pos={position} fair={fair:.1f} edge={edge:.1f} "
                f"t={len(take_orders)} c={len(clear_orders)} m={len(make_orders)} "
                f"tox_cd={int(td.get(f'{product}_tox_cd', 0))}"
            )

        td_out = json.dumps(self._to_native(td))
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
