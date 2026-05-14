import json
import math
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState

# v8 — different architecture from v5–v7:
#
#   OSMIUM — "BBO spread harvest" (no slow blended fair for *quote placement*):
#     • Reference price = book mid (with fallback). Makers at best_bid+1 / best_ask-1
#       clipped to int(ref)±1 so quotes sit in the ~16-tick band where bot flow crosses.
#     • Taker phase REMOVED — on capsule data it almost never fires vs fair; saves bad
#       crosses and matches analysis that edge was too wide vs actual mispricing.
#     • Skew still limits size but cutoff relaxed so both sides quote more often.
#
#   PEPPER — "trend + microstructure":
#     • Phase 1 unchanged: lift entire ask stack to +80 (trend >> spread).
#     • Phase 2: passive uses BOTH trend-fair ceiling AND book mid so bids sit where
#       the oscillating book actually trades, not only int(fair)-2.
#     • Three passive levels when room is large (fills vs single thin quote).
#
# Round 2 MAF: implement bid() below. Top 50% of bids get +25% order-book quotes and
# pay their bid from final R2 profit; losers pay 0. Ignored during local R2 tests.
#
# Manual "Invest & Expand": PnL = (Research × Scale × Speed) − budget_used (see wiki).
# Research ∈ [0,200k] log in %; Scale ∈ [0,7] linear in %; Speed ∈ [0.1,0.9] by rank.

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


POSITION_LIMITS: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# Osmium — BBO + mid reference; no taker layer in v8
OSMIUM_MAKE_SIZE      = 40
OSMIUM_HARD_FALLBACK  = 10_000.0

SKEW_POWER   = 1.35
SKEW_CUTOFF  = 0.92

# Pepper
PEPPER_TREND_PER_TS   = 0.001
PEPPER_LADDER_LEVELS  = 3
PEPPER_LADDER_MIN_CAP = 28

# Blind auction — tune once you have a guess for the median bid across teams.
# Too low: risk missing top 50% (no extra quotes). Too high: pay more than needed.
MAF_BID = 18_000


class Trader:

    def bid(self) -> int:
        """Round 2 only. XIRECs you agree to pay if your bid is in the top 50%."""
        return MAF_BID

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

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
        if product == "INTARIAN_PEPPER_ROOT":
            return self._pepper_fair(depth, timestamp, td)
        if product == "ASH_COATED_OSMIUM":
            mid = self._book_mid(depth)
            if mid is not None and 9_700 < mid < 10_300:
                return float(mid)
            return OSMIUM_HARD_FALLBACK
        return 0.0

    def _osmium_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        """Makers only; ref = mid (passed as fair for Osmium)."""
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        mid = self._book_mid(depth)
        ref = float(mid) if mid is not None else fair
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        skew = position / POSITION_LIMITS["ASH_COATED_OSMIUM"]

        buy_scale = max(0.0, 1.0 - max(0.0, skew) ** SKEW_POWER) if skew <= SKEW_CUTOFF else 0.0
        sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER) if skew >= -SKEW_CUTOFF else 0.0

        buy_cap = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        raw_b = int(OSMIUM_MAKE_SIZE * buy_scale) if buy_scale > 0 else 0
        raw_s = int(OSMIUM_MAKE_SIZE * sell_scale) if sell_scale > 0 else 0
        buy_size = min(buy_cap, max(1, raw_b)) if buy_cap > 0 and raw_b > 0 else 0
        sell_size = min(sell_cap, max(1, raw_s)) if sell_cap > 0 and raw_s > 0 else 0

        ir = int(ref)
        bid_q = min(best_bid + 1, ir - 1)
        ask_q = max(best_ask - 1, ir + 1)
        if skew > 0.35 and sell_size > 0:
            ask_q = max(best_ask - 1, ir)
        if skew < -0.35 and buy_size > 0:
            bid_q = min(best_bid + 1, ir)

        if buy_size > 0 and 0 < bid_q < ref:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q, buy_size))
        if sell_size > 0 and ask_q > ref:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    def _pepper_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", position)
        if buy_cap <= 0:
            return orders

        if depth.sell_orders:
            for ask in sorted(depth.sell_orders.keys()):
                if buy_cap <= 0:
                    break
                vol = min(-depth.sell_orders[ask], buy_cap)
                if vol > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                    buy_cap -= vol

        if buy_cap <= 0 or not depth.buy_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        mid = self._book_mid(depth)
        ceiling = int(fair) - 1
        if ceiling >= fair:
            ceiling -= 1

        # Mid-aware: pull passive toward where the book actually prints
        mid_floor = int(mid) - 1 if mid is not None else int(fair) - 3
        trend_floor = int(fair) - 3
        raw = max(best_bid + 1, mid_floor, trend_floor)
        bid_primary = min(raw, ceiling)
        if bid_primary <= best_bid:
            bid_primary = best_bid + 1
        if bid_primary >= fair or bid_primary <= 0:
            return orders

        levels = [bid_primary, bid_primary - 1, bid_primary - 2]
        prices = []
        for p in levels:
            if p > best_bid and p < fair and p > 0 and p not in prices:
                prices.append(p)

        if not prices:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_primary, buy_cap))
            return orders

        if buy_cap < PEPPER_LADDER_MIN_CAP or len(prices) == 1:
            orders.append(Order("INTARIAN_PEPPER_ROOT", prices[0], buy_cap))
            return orders

        n = min(len(prices), PEPPER_LADDER_LEVELS)
        prices = prices[:n]
        remaining = buy_cap
        for i, pq in enumerate(prices):
            if remaining <= 0:
                break
            chunk = (remaining + n - i - 1) // (n - i)
            chunk = min(chunk, remaining)
            if chunk > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", pq, chunk))
                remaining -= chunk

        return orders

    def _clear_at_fair(self, product: str, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders
        fc = math.ceil(fair)
        ff = math.floor(fair)
        pos = position
        if pos > 0 and fc in depth.buy_orders:
            qty = min(depth.buy_orders[fc], pos, self._allowable_sell(product, pos))
            if qty > 0:
                orders.append(Order(product, fc, -qty))
                pos -= qty
        if pos < 0 and ff in depth.sell_orders:
            qty = min(-depth.sell_orders[ff], -pos, self._allowable_buy(product, pos))
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
            fair = self._fair_value(product, ts, depth, td)

            if product == "ASH_COATED_OSMIUM":
                main = self._osmium_orders(depth, fair, position)
                clear = self._clear_at_fair(product, depth, fair, position)
                result[product] = main + clear
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper_orders(depth, fair, position)

            logger.print(f"{product} pos={position} fair={fair:.1f} orders={len(result[product])}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
