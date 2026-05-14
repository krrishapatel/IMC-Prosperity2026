import json
import math
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState

# MAF: set in submission UI if available.
#
# Team total (e.g. 200k gate) = manual + algo + investments across rounds.
# This file only affects the *algorithmic* book; max position is still capped.

# ──────────────────────────────────────────────────────────────────────────────
# Logger  (competition-safe, do not modify)
# ──────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
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


POSITION_LIMITS = {
    "ASH_COATED_OSMIUM":    80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# Osmium — v4: slightly greedier takes (still bounded) when spread is wide.
OSMIUM_HARD_FAIR   = 10_000.0
OSMIUM_EWM_ALPHA   = 0.15
OSMIUM_EWM_BLEND   = 0.30
OSMIUM_TAKE_FRAC   = 0.22
OSMIUM_MAKE_SIZE   = 24

MIN_TAKE_EDGE = 1.25
SKEW_POWER    = 1.45
SKEW_CUTOFF   = 0.78

# Pepper v4 — different from v2/v3 ladder:
#   * Model fair = anchor + 0.001*ts (unchanged).
#   * TAKES: use a *raised* cutoff fair_m + slack(pos) so asks slightly above the
#     pure model line still count as "cheap" while building long (bounded).
#   * MAKER: one clip, price = join book — min(best_bid+TOB_IMPROVE, int(fair_m)-1)
#     so we step in front of the queue instead of only parking at fair-2/3.
PEPPER_TREND_PER_TS   = 0.001
PEPPER_TAKE_SLACK_MAX = 4.25      # extra ticks above model fair allowed as "cheap"
PEPPER_TAKE_SLACK_K   = 0.055      # slack shrinks as position approaches limit
PEPPER_TOB_IMPROVE    = 2         # bid at best_bid + this many ticks when possible
PEPPER_MAX_MAKER_OFF  = 1         # never post above int(fair_m) - this (passive vs model)


def _pepper_take_cutoff(fair_m: float, position: int) -> float:
    """Higher cutoff => classify more sell-side as liftable while still bounded."""
    slack = max(0.0, PEPPER_TAKE_SLACK_MAX - PEPPER_TAKE_SLACK_K * max(0, position))
    return fair_m + slack


# ──────────────────────────────────────────────────────────────────────────────
class Trader:

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
            return 16.0
        return float(min(depth.sell_orders.keys()) - max(depth.buy_orders.keys()))

    def _osmium_fair(self, depth: OrderDepth, td: dict) -> float:
        mid = self._book_mid(depth)
        if mid is None or not (9_000 < mid < 11_000):
            return float(td.get("osm_ewm", OSMIUM_HARD_FAIR))
        prev = float(td.get("osm_ewm", OSMIUM_HARD_FAIR))
        ewm  = OSMIUM_EWM_ALPHA * mid + (1.0 - OSMIUM_EWM_ALPHA) * prev
        td["osm_ewm"] = ewm
        return (1.0 - OSMIUM_EWM_BLEND) * OSMIUM_HARD_FAIR + OSMIUM_EWM_BLEND * ewm

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict) -> float:
        if "pep_anchor" not in td:
            mid = self._book_mid(depth)
            if mid is not None:
                base = mid - PEPPER_TREND_PER_TS * timestamp
                td["pep_anchor"] = round(base / 1_000.0) * 1_000.0
            else:
                td["pep_anchor"] = 11_000.0
        return float(td["pep_anchor"]) + PEPPER_TREND_PER_TS * timestamp

    def _fair_value(self, product: str, timestamp: int, depth: OrderDepth, td: dict) -> float:
        if product == "ASH_COATED_OSMIUM":
            return self._osmium_fair(depth, td)
        if product == "INTARIAN_PEPPER_ROOT":
            return self._pepper_fair(depth, timestamp, td)
        return 0.0

    def _osmium_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        sp   = self._spread(depth)
        edge = max(MIN_TAKE_EDGE, sp * OSMIUM_TAKE_FRAC)

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        for ask in sorted(depth.sell_orders.keys()):
            if ask >= fair - edge or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask, vol))
                position += vol; buy_cap -= vol

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid <= fair + edge or sell_cap <= 0:
                break
            vol = min(depth.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid, -vol))
                position -= vol; sell_cap -= vol

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        skew     = position / POSITION_LIMITS["ASH_COATED_OSMIUM"]

        buy_scale  = max(0.0, 1.0 - max(0.0,  skew) ** SKEW_POWER) if skew  <= SKEW_CUTOFF else 0.0
        sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER) if skew  >= -SKEW_CUTOFF else 0.0

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        mk = OSMIUM_MAKE_SIZE + (4 if sp >= 14 else 0)
        buy_size  = min(buy_cap,  max(1, int(mk * buy_scale)))  if buy_scale  > 0 else 0
        sell_size = min(sell_cap, max(1, int(mk * sell_scale))) if sell_scale > 0 else 0

        bid_q = min(best_bid + 1, int(fair) - 1)
        ask_q = max(best_ask - 1, int(fair) + 1)
        if skew > 0.5 and sell_size > 0:
            ask_q = max(best_ask - 1, int(fair))
        if skew < -0.5 and buy_size > 0:
            bid_q = min(best_bid + 1, int(fair))

        if buy_size  > 0 and bid_q > 0 and bid_q < fair:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q,  buy_size))
        if sell_size > 0 and ask_q > 0 and ask_q > fair:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    def _pepper_orders(self, depth: OrderDepth, fair_m: float, position: int) -> list[Order]:
        """
        v4: bounded-aggressive lift + join-the-book maker (not fixed fair-2/3 ladder).
        """
        orders: list[Order] = []
        if not depth.sell_orders:
            return orders

        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", position)
        if buy_cap <= 0:
            return orders

        cutoff = _pepper_take_cutoff(fair_m, position)

        for ask in sorted(depth.sell_orders.keys()):
            if ask >= cutoff or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                position += vol; buy_cap -= vol

        if buy_cap <= 0 or not depth.buy_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        cap_bid  = int(fair_m) - PEPPER_MAX_MAKER_OFF
        bid_q    = min(best_bid + PEPPER_TOB_IMPROVE, cap_bid)
        if bid_q <= best_bid:
            bid_q = best_bid + 1
        bid_q = max(1, bid_q)
        if bid_q >= fair_m:
            return orders

        orders.append(Order("INTARIAN_PEPPER_ROOT", bid_q, buy_cap))
        return orders

    def _clear_at_fair(self, product: str, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        fc = math.ceil(fair)
        ff = math.floor(fair)

        if position > 0 and fc in depth.buy_orders:
            qty = min(depth.buy_orders[fc], position, self._allowable_sell(product, position))
            if qty > 0:
                orders.append(Order(product, fc, -qty))
                position -= qty

        if position < 0 and ff in depth.sell_orders:
            qty = min(-depth.sell_orders[ff], -position, self._allowable_buy(product, position))
            if qty > 0:
                orders.append(Order(product, ff, qty))

        return orders

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {}
        ts = state.timestamp

        for product in POSITION_LIMITS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue

            position = state.position.get(product, 0)
            fair     = self._fair_value(product, ts, depth, td)

            if product == "ASH_COATED_OSMIUM":
                main_orders  = self._osmium_orders(depth, fair, position)
                clear_orders = self._clear_at_fair(product, depth, fair, position)
                result[product] = main_orders + clear_orders

            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper_orders(depth, fair, position)

            logger.print(f"{product} pos={position} fair={fair:.1f} orders={len(result[product])}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
