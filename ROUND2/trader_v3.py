import json
import math
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, TradingState

# MAF: configure in the competition submission UI if your round supports it.

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
# Strategy parameters
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM":    80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# Osmium: unchanged core; slightly larger clips for inventory turnover.
OSMIUM_HARD_FAIR   = 10_000.0
OSMIUM_EWM_ALPHA   = 0.15
OSMIUM_EWM_BLEND   = 0.30
OSMIUM_TAKE_FRAC   = 0.15
OSMIUM_MAKE_SIZE   = 22

# Pepper v3: faster to +80, more maker surface than v2 (single fair-3 bid).
#  - Phase 1: same sweep, all asks < fair.
#  - Phase 2: ladder bids at int(fair)-2 and int(fair)-3 (deduped), split size.
#  - Early session: full buy_scale until PEPPER_FULL_CLIP_UNTIL_POS, larger base size.
PEPPER_TREND_PER_TS         = 0.001
PEPPER_MAKE_SIZE            = 22
PEPPER_MAKER_OFFSETS        = (2, 3)   # two passive levels inside the oscillation band
PEPPER_FULL_CLIP_UNTIL_POS  = 45       # ignore skew decay until this many units long
PEPPER_EARLY_TS             = 12_000   # before this ts, bump per-order clip
PEPPER_EARLY_SIZE_MULT      = 1.35

MIN_TAKE_EDGE = 1.5
SKEW_POWER    = 1.5
SKEW_CUTOFF   = 0.75


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

        buy_size  = min(buy_cap,  max(1, int(OSMIUM_MAKE_SIZE * buy_scale)))  if buy_scale  > 0 else 0
        sell_size = min(sell_cap, max(1, int(OSMIUM_MAKE_SIZE * sell_scale))) if sell_scale > 0 else 0

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

    def _pepper_orders(self, depth: OrderDepth, fair: float, position: int, timestamp: int) -> list[Order]:
        """
        v3: v2 sweep + dual maker ladder + early size / skew bypass to hit +80 sooner.
        """
        orders: list[Order] = []
        if not depth.sell_orders:
            return orders

        limit   = POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", position)
        if buy_cap <= 0:
            return orders

        for ask in sorted(depth.sell_orders.keys()):
            if ask >= fair or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                position += vol; buy_cap -= vol

        if buy_cap <= 0 or not depth.buy_orders:
            return orders

        skew = position / limit
        if position < PEPPER_FULL_CLIP_UNTIL_POS:
            buy_scale = 1.0
        else:
            buy_scale = max(0.0, 1.0 - max(0.0, skew) ** SKEW_POWER) if skew <= SKEW_CUTOFF else 0.0

        base = PEPPER_MAKE_SIZE
        if timestamp < PEPPER_EARLY_TS:
            base = int(PEPPER_MAKE_SIZE * PEPPER_EARLY_SIZE_MULT)
        buy_size = min(buy_cap, max(1, int(base * buy_scale))) if buy_scale > 0 else 0
        if buy_size <= 0:
            return orders

        center = int(fair)
        prices: list[int] = []
        for off in PEPPER_MAKER_OFFSETS:
            pq = center - off
            if pq > 0 and pq < fair and pq not in prices:
                prices.append(pq)

        if not prices:
            return orders

        n = len(prices)
        remaining = buy_size
        for i, pq in enumerate(prices):
            if remaining <= 0:
                break
            chunk = (remaining + n - i - 1) // (n - i)  # ceil split
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
                result[product] = self._pepper_orders(depth, fair, position, ts)

            logger.print(f"{product} pos={position} fair={fair:.1f} orders={len(result[product])}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
