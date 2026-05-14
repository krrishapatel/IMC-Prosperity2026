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
# v9 — data-driven parameter improvements over v8
#
# KEY FINDINGS from backtester + simulation analysis:
#
# v8 is earning ~90k/day (9k shown at ts=100k = 10% of day → extrapolates to 90k).
# Breakdown: ~15k osmium MM + ~75k pepper trend (mark-to-market).
#
# V9 improvements:
#
# 1. OSMIUM TAKER (new): v8 has NO taker. Analysis shows:
#    - 338 timestamps where ask1 < 10000 (avg ask = 9997.4, vol = 2972)
#    - 346 timestamps where bid1 > 10000 (avg bid = 10002.4, vol = 3300)
#    - Potential taker profit: ~13,276 per day
#    - Adding taker with edge=0.5 improves osmium from 15,569 → 17,862/day (+2,293)
#
# 2. MAF FIX: v8 bids 18,000. Analysis:
#    - Osmium MM + 25% volume: +3,892/day × 3 days = +11,676 benefit
#    - v8 pays 18,000: net LOSS of -6,324
#    - v9 bids 11,500: net GAIN of +176 (break-even but ensures top-50% entry)
#
# 3. OSMIUM MAKER SIZE: v8 uses 40. Max trade size is 10. Average is 5.
#    Keeping at 40 is fine — it's already above max trade size so all fills captured.
#    v9 uses 40 (unchanged, no benefit to raising).
#
# 4. PEPPER: v8 strategy is optimal. Hold +80 from ts=0, trend earns ~79k/day.
#    Oscillation selling tested and HURTS (-1,810 over 3 days) — don't do it.
#    v9 keeps pepper identical to v8.
#
# 5. SKEW CUTOFF: v8 uses 0.92. Analysis shows positions rarely hit limit
#    (only 10 units missed/day). Slight tightening to 0.85 keeps quotes active
#    on both sides longer without hurting fill rate.
#
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMITS: dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# ── Osmium ─────────────────────────────────────────────────────────────────────
# True long-run mean = 10000 (confirmed across all 3 days via regression).
# Spread is ~16 (bid ~9993, ask ~10009 typically).
# Taker: grab anything when ask < FAIR - TAKER_EDGE or bid > FAIR + TAKER_EDGE.
OSMIUM_FAIR       = 10_000.0
OSMIUM_TAKER_EDGE = 0.5          # NEW in v9: tight taker catches mispriced quotes
OSMIUM_MAKE_SIZE  = 40           # covers max single-trade volume (max=10, avg=5)
OSMIUM_HARD_FALLBACK = 10_000.0

# ── Pepper ─────────────────────────────────────────────────────────────────────
# Perfect linear trend: fair(ts) = anchor + 0.001 * ts.
# Strategy: sweep all asks to +80 ASAP, hold forever.
# Do NOT sell to oscillating bots — tested and confirmed to hurt PnL.
PEPPER_TREND = 0.001

# ── Shared ─────────────────────────────────────────────────────────────────────
SKEW_POWER  = 1.35
SKEW_CUTOFF = 0.85   # slightly tighter than v8's 0.92

# ── MAF ────────────────────────────────────────────────────────────────────────
# v8 bid 18,000 but analysis shows benefit is only ~11,676 over 3 days.
# v9 bids 11,500 to be near break-even while still entering top-50%.
MAF_BID = 11_500


class Trader:

    def bid(self) -> int:
        """Round 2 MAF. Top 50% pay this and get +25% quote access."""
        return MAF_BID

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    # ── Fair values ────────────────────────────────────────────────────────────

    def _osmium_fair(self, depth: OrderDepth) -> float:
        """
        Use raw book mid as reference for quote placement (same as v8).
        Hard fallback = 10000 if book is empty or out of range.
        """
        mid = self._book_mid(depth)
        if mid is not None and 9_500 < mid < 10_500:
            return float(mid)
        return OSMIUM_HARD_FALLBACK

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict) -> float:
        """
        fair(ts) = anchor + 0.001 * ts.
        Anchor inferred from first mid, rounded to nearest 1000.
        Day -1: 11000, Day 0: 12000, Day 1: 13000 (confirmed exact).
        """
        if "pep_anchor" not in td:
            mid = self._book_mid(depth)
            if mid is not None:
                base = mid - PEPPER_TREND * timestamp
                td["pep_anchor"] = round(base / 1_000.0) * 1_000.0
            else:
                td["pep_anchor"] = 11_000.0
        return float(td["pep_anchor"]) + PEPPER_TREND * timestamp

    # ── Osmium orders ───────────────────────────────────────────────────────────

    def _osmium_orders(self, depth: OrderDepth, ref: float, position: int) -> list[Order]:
        """
        Two layers:

        Layer 1 — TAKER (new in v9):
          Sweep anything priced away from OSMIUM_FAIR by more than TAKER_EDGE.
          Analysis: 338 cheap-ask timestamps (avg 9997.4) + 346 rich-bid timestamps
          (avg 10002.4) per day. Adding this taker improves day PnL 15,569 → 17,862.
          We use OSMIUM_FAIR (10000), not the rolling mid, as reference for taker
          because the mid fluctuates and we don't want to chase noise.

        Layer 2 — MAKER:
          Post at best_bid+1 / best_ask-1 with size=OSMIUM_MAKE_SIZE.
          Gets filled when bot market orders cross our price.
          ~465 such trades/day, avg size 5.1, total volume 2375.
          Skew scaling prevents inventory blowup.
        """
        orders: list[Order] = []
        if not depth.buy_orders and not depth.sell_orders:
            return orders

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        # ── Layer 1: Taker ────────────────────────────────────────────────────
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

        # ── Layer 2: Maker ────────────────────────────────────────────────────
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

        # Inventory nudge: when skewed, pull quotes toward closing the position
        if skew > 0.35 and sell_size > 0:
            ask_q = max(best_ask - 1, ir)
        if skew < -0.35 and buy_size > 0:
            bid_q = min(best_bid + 1, ir)

        if buy_size  > 0 and 0 < bid_q < ref:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q,  buy_size))
        if sell_size > 0 and ask_q > ref:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    # ── Pepper orders ───────────────────────────────────────────────────────────

    def _pepper_orders(self, depth: OrderDepth, fair: float, position: int) -> list[Order]:
        """
        Optimal pepper strategy (confirmed by simulation):

        The trend is +0.001/ts = +1000/day. Buying at ask (fair+7) and holding
        earns ~993/unit/day. Holding +80 from ts=0 earns ~79,000/day.

        Oscillation selling TESTED and CONFIRMED to HURT (-1,810 over 3 days):
        selling to bots above fair reduces position and the rebuy rarely happens
        before the trend has moved further up, netting a loss.

        Strategy:
          Phase 1: Sweep ALL ask levels to reach +80 as fast as possible.
          Phase 2: Once at +80, post passive bids below fair to mop up any
                   inbound market sells at a discount (pure alpha, no downside).
        """
        orders: list[Order] = []
        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", position)
        if buy_cap <= 0:
            return orders

        # Phase 1: Aggressive sweep — lift every ask until +80
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

        # Phase 2: Passive bids below fair for any remaining capacity
        best_bid = max(depth.buy_orders.keys())
        mid = self._book_mid(depth)

        # Bid at int(fair)-2, capped strictly below fair
        ceiling = int(fair) - 1
        if ceiling >= fair:
            ceiling -= 1

        # Pull toward actual mid to stay relevant in the book
        mid_based = int(mid) - 1 if mid is not None else int(fair) - 3
        raw = max(best_bid + 1, mid_based, int(fair) - 3)
        bid_price = min(raw, ceiling)

        if bid_price <= best_bid:
            bid_price = best_bid + 1
        if bid_price >= fair or bid_price <= 0:
            return orders

        # Split across two levels if we have substantial capacity
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

    # ── Main ────────────────────────────────────────────────────────────────────

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
                ref = self._osmium_fair(depth)
                result[product] = self._osmium_orders(depth, ref, position)

            elif product == "INTARIAN_PEPPER_ROOT":
                fair = self._pepper_fair(depth, ts, td)
                result[product] = self._pepper_orders(depth, fair, position)

            logger.print(f"{product} pos={position} orders={len(result.get(product,[]))}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out