import json
import math
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState

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


# ──────────────────────────────────────────────────────────────────────────────
# v10 — data-driven improvements over v9
#
# WHAT CHANGED AND WHY:
#
# OSMIUM TAKER EDGE: v9 used edge=0.5. Optimal is edge=4.0.
#
#   Grid search across all 3 historical days:
#     edge=0.5 → 52,015 total   (v9 behavior)
#     edge=1.0 → 55,056
#     edge=2.0 → 56,591
#     edge=3.0 → 56,837
#     edge=4.0 → 56,925  ← BEST
#     edge=5.0 → 55,570
#
#   Why edge=4 beats edge=0.5:
#   With edge=0.5, any bid above 10000.5 triggers a SELL taker.
#   bid=10001 happens constantly (bid=10001, mid=10009 is common).
#   We sell at 10001 when the market is pricing at 10009 → paper loss at start.
#   This causes the -2,000 dip in v9's backtester chart.
#   
#   With edge=4, we only sell when bid > 10004 — genuinely above fair.
#   This keeps us from aggressively shorting into a market that's 9 ticks above fair.
#   Cleaner position, less adverse mark-to-market during session.
#
# MAF: Kept at 11,500. v8 overpaid at 18,000 (net -6,324). v9/v10 at 11,500 = break-even.
#
# PEPPER: Unchanged. Oscillation trading tested repeatedly:
#   - Oscillation: -1,810 over 3 days (simulation #1)
#   - Reacting to market_trades: -2,161 over 3 days (simulation #2)
#   Market_trades shows last tick's activity; by the time we react, opportunity gone.
#   Best strategy: aggressive take to +80, hold all day.
#
# MAKER SIZE: 40 (unchanged). Average bot trade = 5.1 units, max = 10.
#   Size=40 fully captures every fill. No benefit to raising further.
#
# SKEW CUTOFF: 0.85 (unchanged from v9). Position misses are ~10 units/day = negligible.
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMITS: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# ── Osmium ─────────────────────────────────────────────────────────────────────
OSMIUM_FAIR       = 10_000.0
OSMIUM_TAKER_EDGE = 4.0       # KEY CHANGE: was 0.5 in v9. Grid search optimal = 4.0.
                               # Avoids selling into mid=10009 when bid=10001.
OSMIUM_MAKE_SIZE  = 40
SKEW_POWER        = 1.35
SKEW_CUTOFF       = 0.85

# ── Pepper ─────────────────────────────────────────────────────────────────────
PEPPER_TREND = 0.001

# ── MAF ────────────────────────────────────────────────────────────────────────
MAF_BID = 11_500   # break-even vs +25% volume benefit over 3 days


class Trader:

    def bid(self) -> int:
        return MAF_BID

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def _osmium_ref(self, depth: OrderDepth) -> float:
        mid = self._book_mid(depth)
        if mid is not None and 9_500 < mid < 10_500:
            return float(mid)
        return 10_000.0

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict) -> float:
        if "pep_anchor" not in td:
            mid = self._book_mid(depth)
            if mid is not None:
                base = mid - PEPPER_TREND * timestamp
                td["pep_anchor"] = round(base / 1_000.0) * 1_000.0
            else:
                td["pep_anchor"] = 11_000.0
        return float(td["pep_anchor"]) + PEPPER_TREND * timestamp

    def _osmium_orders(self, depth: OrderDepth, ref: float, position: int) -> list[Order]:
        """
        Two-layer osmium:

        TAKER (OSMIUM_FAIR ± TAKER_EDGE):
          Take only when price is genuinely mispriced vs the 10000 long-run mean.
          Edge=4.0 means: buy if ask < 9996, sell if bid > 10004.
          This avoids shorting when bid=10001 but mid=10009 (v9's initial dip problem).
          Grid-searched: edge=4 gives best 3-day total of 56,925 vs 52,015 at edge=0.5.

        MAKER (best_bid+1 / best_ask-1):
          Post inside the spread. Gets filled when bot market orders cross our price.
          ~465 bot trades/day, avg 5.1 units = 2,375 total volume.
          Skew scaling prevents inventory blowup.
        """
        orders: list[Order] = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        # ── Taker: only take genuinely mispriced levels ────────────────────────
        for ask in sorted(depth.sell_orders.keys()):
            if ask >= OSMIUM_FAIR - OSMIUM_TAKER_EDGE or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask, vol))
                position += vol; buy_cap -= vol

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid <= OSMIUM_FAIR + OSMIUM_TAKER_EDGE or sell_cap <= 0:
                break
            vol = min(depth.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid, -vol))
                position -= vol; sell_cap -= vol

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # ── Maker: post 1 inside best bid/ask ─────────────────────────────────
        skew = position / POSITION_LIMITS["ASH_COATED_OSMIUM"]

        buy_scale  = max(0.0, 1.0 - max(0.0,  skew) ** SKEW_POWER) if skew  <= SKEW_CUTOFF else 0.0
        sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER) if skew >= -SKEW_CUTOFF else 0.0

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        buy_size  = min(buy_cap,  max(1, int(OSMIUM_MAKE_SIZE * buy_scale)))  if buy_scale  > 0 and buy_cap  > 0 else 0
        sell_size = min(sell_cap, max(1, int(OSMIUM_MAKE_SIZE * sell_scale))) if sell_scale > 0 and sell_cap > 0 else 0

        ir = int(ref)
        bid_q = min(best_bid + 1, ir - 1)
        ask_q = max(best_ask - 1, ir + 1)

        if skew > 0.35 and sell_size > 0:
            ask_q = max(best_ask - 1, ir)
        if skew < -0.35 and buy_size > 0:
            bid_q = min(best_bid + 1, ir)

        if buy_size  > 0 and 0 < bid_q < ref:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q,  buy_size))
        if sell_size > 0 and ask_q > ref:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    def _pepper_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        """
        Optimal pepper: aggressive take to +80, then hold.

        Trend = +0.001/ts = +1000/day. Buying at ask (fair+7) and holding
        earns ~993/unit/day. 80 units * 993 = ~79,440/day.

        Tested oscillation selling: -1,810 and -2,161 over 3 days in two
        simulations. market_trades shows last-tick activity, so by the time
        we react, the opportunity is gone. Never sell pepper.

        Phase 2: passive bids below fair to mop up any inbound market sells
        while position < 80 (early session only, first 2-3 ticks).
        """
        orders: list[Order] = []
        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", position)
        if buy_cap <= 0:
            return orders

        # Phase 1: sweep all asks until +80
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

        # Phase 2: passive bids for any remaining room (only first few ticks)
        best_bid = max(depth.buy_orders.keys())
        mid = self._book_mid(depth)

        ceiling = int(fair) - 1
        if ceiling >= fair:
            ceiling -= 1

        mid_based = int(mid) - 1 if mid is not None else int(fair) - 3
        raw = max(best_bid + 1, mid_based, int(fair) - 3)
        bid_price = min(raw, ceiling)

        if bid_price <= best_bid:
            bid_price = best_bid + 1
        if bid_price >= fair or bid_price <= 0:
            return orders

        if buy_cap >= 20:
            bid2 = bid_price - 1
            if bid2 > best_bid and bid2 < fair:
                n1 = (buy_cap + 1) // 2
                n2 = buy_cap - n1
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, n1))
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid2,      n2))
            else:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_cap))
        else:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_cap))

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

            if product == "ASH_COATED_OSMIUM":
                ref = self._osmium_ref(depth)
                result[product] = self._osmium_orders(depth, ref, position)

            elif product == "INTARIAN_PEPPER_ROOT":
                fair = self._pepper_fair(depth, ts, td)
                result[product] = self._pepper_orders(depth, fair, position)

            logger.print(f"{product} pos={position} orders={len(result.get(product,[]))}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out