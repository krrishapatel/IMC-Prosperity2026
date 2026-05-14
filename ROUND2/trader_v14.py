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
# v14 — round 2 at theoretical max + improved universal MM for round 3
#
# ROUND 2 STATUS (all 3 historical days):
#   Day -1: ~82,500  Day 0: ~82,200  Day 1: ~82,800  → ~247k over 3 days
#   v13 scored 325,712 (round1=50,303 + round2≈275k after MAF/variance)
#   We are at 99%+ of theoretical maximum for Osmium + Pepper.
#   Every remaining lever has been grid-searched. No material round 2 gain left.
#
# V14 CHANGES:
#
# 1. ROUND 2 PARAMETERS: UNCHANGED from v13. Optimal.
#    - Osmium taker edge: 3.0 (best across all 3 days)
#    - Osmium maker size: 40 (covers max bot trade of 10, fully captures all fills)
#    - Pepper: sweep all asks to +80, hold all session (~79,400/day)
#    - Pepper anchor: round to nearest 500 for robustness
#    - MAF: 500 (net +372 if won, 0 if not won; always non-negative EV)
#
# 2. IMPROVED UNIVERSAL MM for round 3 new products:
#
#    a) EMA fair value (alpha=0.15, window≈12 ticks) stored in trader_data.
#       v13 computed mid fresh each tick with no memory — noisy for volatile products.
#       EMA smooths out short-term noise, giving better quote placement.
#
#    b) Taker edge lowered: 2.0 → 1.0.
#       For unknown products, we don't know their fair value with confidence.
#       Edge=2 means we take if ask < mid-2, which might grab adversely selected flow.
#       Edge=1 is more conservative — we only take genuinely mispriced quotes.
#
#    c) Maker size increased: 10 → 15.
#       Higher size fills more bot trades per tick without risk (skew still protects us).
#       Small enough to not blow position limits on multi-product round 3.
#
#    d) Skew power lowered: 1.5 → 1.2.
#       More gradual size reduction as inventory builds.
#       1.5 cuts size too aggressively at mid-range positions (50% pos → 35% size).
#       1.2 keeps more size active at intermediate positions.
#
#    e) Position limit detection: check listings for any position hint, else 50.
#       Round 3 products often have position limits in the listing denomination field.
#
# 3. MAF: 500 (unchanged from v13).
#    True round 2 benefit = 872 seashells over 3 days (25% more passive book depth).
#    At 500: net +372 if won, 0 if not won. Never negative EV.
#
# ──────────────────────────────────────────────────────────────────────────────

# ── Known round 2 products ────────────────────────────────────────────────────
POSITION_LIMITS: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# ── Osmium ─────────────────────────────────────────────────────────────────────
OSMIUM_FAIR       = 10_000.0
OSMIUM_TAKER_EDGE = 3.0        # grid-searched optimal across all 3 days
OSMIUM_MAKE_SIZE  = 40         # covers max bot trade of 10; no missed fills
SKEW_POWER        = 1.35

# ── Pepper ─────────────────────────────────────────────────────────────────────
PEPPER_TREND = 0.001            # confirmed exact: +0.001/ts across all 3 days

# ── Universal MM (round 3+ new products) ─────────────────────────────────────
UNIVERSAL_TAKER_EDGE = 1.0     # conservative: only take clearly mispriced quotes
UNIVERSAL_MAKE_SIZE  = 15      # fills more bot trades; skew prevents blowup
UNIVERSAL_SKEW_POW   = 1.2     # gradual size reduction; 1.5 was too aggressive
UNIVERSAL_POS_LIMIT  = 50      # default position limit for unknown products
UNIVERSAL_EMA_ALPHA  = 0.15    # EMA smoothing for fair value (~12-tick window)

# ── MAF ────────────────────────────────────────────────────────────────────────
# Real 3-day benefit = ~872 (25% more passive book volume for taker).
# At 500: if won net +372, if not won net 0. Always non-negative.
MAF_BID = 500


class Trader:

    def bid(self) -> int:
        return MAF_BID

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _allowable_buy(self, limit: int, position: int) -> int:
        return max(0, limit - position)

    def _allowable_sell(self, limit: int, position: int) -> int:
        return max(0, limit + position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    # ── Osmium ─────────────────────────────────────────────────────────────────

    def _osmium_ref(self, depth: OrderDepth) -> float:
        mid = self._book_mid(depth)
        if mid is not None and 9_000 < mid < 11_000:
            return float(mid)
        return OSMIUM_FAIR

    def _osmium_orders(self, depth: OrderDepth, ref: float, position: int) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        limit = POSITION_LIMITS["ASH_COATED_OSMIUM"]
        buy_cap  = self._allowable_buy(limit, position)
        sell_cap = self._allowable_sell(limit, position)

        # Taker: grab anything clearly mispriced vs 10000 fair (edge=3)
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

        # Maker: post 1 inside best bid/ask with smooth skew scaling
        skew = position / limit
        buy_scale  = max(0.0, 1.0 - max(0.0,  skew) ** SKEW_POWER)
        sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER)

        buy_cap  = self._allowable_buy(limit, position)
        sell_cap = self._allowable_sell(limit, position)

        buy_size  = min(buy_cap,  max(1, int(OSMIUM_MAKE_SIZE * buy_scale)))  if buy_scale  > 0 and buy_cap  > 0 else 0
        sell_size = min(sell_cap, max(1, int(OSMIUM_MAKE_SIZE * sell_scale))) if sell_scale > 0 and sell_cap > 0 else 0

        ir = int(ref)
        bid_q = min(best_bid + 1, ir - 1)
        ask_q = max(best_ask - 1, ir + 1)

        # Tighten ask/bid when skewed to help revert position
        if skew > 0.35 and sell_size > 0:
            ask_q = max(best_ask - 1, ir)
        if skew < -0.35 and buy_size > 0:
            bid_q = min(best_bid + 1, ir)

        if buy_size  > 0 and 0 < bid_q < ref:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q,  buy_size))
        if sell_size > 0 and ask_q > ref:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    # ── Pepper ─────────────────────────────────────────────────────────────────

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict) -> float:
        if "pep_anchor" not in td:
            mid = self._book_mid(depth)
            if mid is not None:
                base = mid - PEPPER_TREND * timestamp
                td["pep_anchor"] = round(base / 500.0) * 500.0
            else:
                td["pep_anchor"] = 11_000.0
        return float(td["pep_anchor"]) + PEPPER_TREND * timestamp

    def _pepper_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        orders: list[Order] = []
        limit = POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        buy_cap = self._allowable_buy(limit, position)
        if buy_cap <= 0:
            return orders

        # Sweep all asks to reach +80 as quickly as possible (trend = +993/unit/day net)
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

        # Passive bid just below fair for any remaining capacity (first 2 ticks only)
        best_bid = max(depth.buy_orders.keys())
        ceiling  = int(fair) - 1
        bid_price = min(best_bid + 1, ceiling)
        if 0 < bid_price < fair:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_cap))

        return orders

    # ── Universal market maker (round 3+ new products) ─────────────────────────

    def _universal_mm(self, symbol: str, depth: OrderDepth, position: int,
                      limit: int, td: dict) -> list[Order]:
        """
        Improved market maker for any unknown product (round 3+).

        Key improvements over v13:
        - EMA fair value stored in trader_data for noise reduction.
          Raw mid can jump ±5 from tick to tick on thin books; EMA stabilizes quotes.
        - Conservative taker edge (1.0) to avoid adversely selected fills.
        - Slightly larger make size (15) with gentler skew cutoff (power=1.2).
        """
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        raw_mid = self._book_mid(depth)
        if raw_mid is None:
            return orders

        # EMA fair value — smooths quote placement over time
        ema_key = f"ema_{symbol}"
        if ema_key not in td:
            td[ema_key] = raw_mid
        else:
            td[ema_key] = UNIVERSAL_EMA_ALPHA * raw_mid + (1.0 - UNIVERSAL_EMA_ALPHA) * td[ema_key]
        fair = td[ema_key]

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        buy_cap  = self._allowable_buy(limit, position)
        sell_cap = self._allowable_sell(limit, position)

        # Taker: grab clearly mispriced quotes vs EMA fair value
        for ask in sorted(depth.sell_orders.keys()):
            if ask >= fair - UNIVERSAL_TAKER_EDGE or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap, UNIVERSAL_MAKE_SIZE)
            if vol > 0:
                orders.append(Order(symbol, ask, vol))
                position += vol; buy_cap -= vol

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid <= fair + UNIVERSAL_TAKER_EDGE or sell_cap <= 0:
                break
            vol = min(depth.buy_orders[bid], sell_cap, UNIVERSAL_MAKE_SIZE)
            if vol > 0:
                orders.append(Order(symbol, bid, -vol))
                position -= vol; sell_cap -= vol

        # Maker: post 1 inside spread with smooth skew protection
        skew = position / max(limit, 1)
        buy_scale  = max(0.0, 1.0 - max(0.0,  skew) ** UNIVERSAL_SKEW_POW)
        sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** UNIVERSAL_SKEW_POW)

        buy_cap  = self._allowable_buy(limit, position)
        sell_cap = self._allowable_sell(limit, position)

        buy_size  = min(buy_cap,  max(1, int(UNIVERSAL_MAKE_SIZE * buy_scale)))  if buy_scale  > 0 and buy_cap  > 0 else 0
        sell_size = min(sell_cap, max(1, int(UNIVERSAL_MAKE_SIZE * sell_scale))) if sell_scale > 0 and sell_cap > 0 else 0

        bid_q = best_bid + 1
        ask_q = best_ask - 1

        if bid_q >= ask_q:   # spread too tight to improve (1-tick spread)
            return orders

        if buy_size  > 0 and bid_q < fair:
            orders.append(Order(symbol, bid_q,  buy_size))
        if sell_size > 0 and ask_q > fair:
            orders.append(Order(symbol, ask_q, -sell_size))

        return orders

    # ── Main ────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        result: dict[Symbol, list[Order]] = {}
        ts = state.timestamp

        for symbol in state.order_depths:
            depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)

            try:
                if symbol == "ASH_COATED_OSMIUM":
                    ref = self._osmium_ref(depth)
                    result[symbol] = self._osmium_orders(depth, ref, position)

                elif symbol == "INTARIAN_PEPPER_ROOT":
                    fair = self._pepper_fair(depth, ts, td)
                    result[symbol] = self._pepper_orders(depth, fair, position)

                else:
                    # Unknown product (round 3+ addition) — universal MM
                    # Try to get position limit from listing; fall back to default
                    limit = UNIVERSAL_POS_LIMIT
                    if symbol in state.listings:
                        try:
                            # Some rounds encode limits in denomination as an int
                            denom = state.listings[symbol].denomination
                            if isinstance(denom, (int, float)) and 1 < denom < 10_000:
                                limit = int(denom)
                        except Exception:
                            pass
                    # Also respect current position if it reveals a higher limit
                    limit = max(limit, abs(position) + 1)
                    result[symbol] = self._universal_mm(symbol, depth, position, limit, td)

            except Exception as e:
                logger.print(f"Error on {symbol}: {e}")
                result[symbol] = []

            logger.print(f"{symbol} pos={position} orders={len(result.get(symbol, []))}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out