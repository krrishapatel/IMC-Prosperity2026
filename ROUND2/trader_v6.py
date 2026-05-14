import json
import math
from typing import Any

from datamodel import Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# MAF: submission UI if applicable.

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
# v6 — Osmium: tight taker + BBO+1 / BBO-1 makers (bot flow). Pepper: +80 trend
#      carry + optional oscillation sell when market_trades print premium bids.
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM":    80,
    "INTARIAN_PEPPER_ROOT": 80,
}

# Osmium
OSMIUM_FAIR        = 10_000.0
OSMIUM_EWM_ALPHA   = 0.10
OSMIUM_EWM_BLEND   = 0.20
OSMIUM_TAKE_EDGE   = 1.0
OSMIUM_MAKE_SIZE   = 30
OSMIUM_MAX_CLIP    = 25

# Pepper
PEPPER_TREND          = 0.001
PEPPER_MIN_HOLD       = 60
PEPPER_SELL_ABOVE     = 3.0
PEPPER_MAX_OSC_SELL   = 20
PEPPER_PASSIVE_OFFSET = 2

SKEW_POWER   = 1.5
SKEW_CUTOFF  = 0.80


class Trader:

    def _allowable_buy(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: str, position: int) -> int:
        return max(0, POSITION_LIMITS[product] + position)

    def _book_mid(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def _osmium_fair(self, depth: OrderDepth, td: dict) -> float:
        mid = self._book_mid(depth)
        if mid is None or not (9_500 < mid < 10_500):
            return float(td.get("osm_ewm", OSMIUM_FAIR))
        prev = float(td.get("osm_ewm", OSMIUM_FAIR))
        ewm  = OSMIUM_EWM_ALPHA * mid + (1.0 - OSMIUM_EWM_ALPHA) * prev
        td["osm_ewm"] = ewm
        return (1.0 - OSMIUM_EWM_BLEND) * OSMIUM_FAIR + OSMIUM_EWM_BLEND * ewm

    def _pepper_fair(self, depth: OrderDepth, timestamp: int, td: dict) -> float:
        if "pep_anchor" not in td:
            mid = self._book_mid(depth)
            if mid is not None:
                base = mid - PEPPER_TREND * timestamp
                td["pep_anchor"] = round(base / 1_000.0) * 1_000.0
            else:
                td["pep_anchor"] = 11_000.0
        return float(td["pep_anchor"]) + PEPPER_TREND * timestamp

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

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        for ask in sorted(depth.sell_orders.keys()):
            if ask >= fair - OSMIUM_TAKE_EDGE or buy_cap <= 0:
                break
            vol = min(-depth.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask, vol))
                position += vol
                buy_cap -= vol

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid <= fair + OSMIUM_TAKE_EDGE or sell_cap <= 0:
                break
            vol = min(depth.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid, -vol))
                position -= vol
                sell_cap -= vol

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        skew = position / POSITION_LIMITS["ASH_COATED_OSMIUM"]

        if abs(skew) <= SKEW_CUTOFF:
            buy_scale  = max(0.0, 1.0 - max(0.0, skew) ** SKEW_POWER)
            sell_scale = max(0.0, 1.0 - max(0.0, -skew) ** SKEW_POWER)
        else:
            buy_scale  = 0.0 if skew > 0 else 1.0
            sell_scale = 0.0 if skew < 0 else 1.0

        buy_cap  = self._allowable_buy("ASH_COATED_OSMIUM", position)
        sell_cap = self._allowable_sell("ASH_COATED_OSMIUM", position)

        raw_buy  = int(OSMIUM_MAKE_SIZE * buy_scale) if buy_scale > 0 else 0
        raw_sell = int(OSMIUM_MAKE_SIZE * sell_scale) if sell_scale > 0 else 0
        buy_size  = min(buy_cap,  max(1, raw_buy),  OSMIUM_MAX_CLIP) if buy_cap > 0 and raw_buy > 0 else 0
        sell_size = min(sell_cap, max(1, raw_sell), OSMIUM_MAX_CLIP) if sell_cap > 0 and raw_sell > 0 else 0

        bid_q = min(best_bid + 1, int(fair) - 1)
        ask_q = max(best_ask - 1, int(fair) + 1)
        if skew > 0.4:
            ask_q = max(best_ask - 1, int(fair))
        if skew < -0.4:
            bid_q = min(best_bid + 1, int(fair))

        if buy_size > 0 and 0 < bid_q < fair:
            orders.append(Order("ASH_COATED_OSMIUM", bid_q, buy_size))
        if sell_size > 0 and ask_q > fair:
            orders.append(Order("ASH_COATED_OSMIUM", ask_q, -sell_size))

        return orders

    def _pepper_orders(
        self,
        depth: OrderDepth,
        fair: float,
        position: int,
        mkt: list[Trade],
    ) -> list[Order]:
        """
        Sequential within tick: optional sell into premium prints, then lift asks
        to use remaining buy room, then passive bid below fair.
        """
        orders: list[Order] = []
        pos = position

        max_trade_px: int | None = None
        for t in mkt:
            max_trade_px = t.price if max_trade_px is None else max(max_trade_px, t.price)

        sell_cap = self._allowable_sell("INTARIAN_PEPPER_ROOT", pos)
        if (
            max_trade_px is not None
            and max_trade_px >= fair + PEPPER_SELL_ABOVE
            and pos > PEPPER_MIN_HOLD
            and sell_cap > 0
            and depth.buy_orders
        ):
            best_bid = max(depth.buy_orders.keys())
            avail = depth.buy_orders.get(best_bid, 0)
            qty = min(PEPPER_MAX_OSC_SELL, pos - PEPPER_MIN_HOLD, sell_cap, avail)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))
                pos -= qty

        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", pos)
        if buy_cap > 0 and depth.sell_orders:
            for ask in sorted(depth.sell_orders.keys()):
                if buy_cap <= 0:
                    break
                vol = min(-depth.sell_orders[ask], buy_cap)
                if vol > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, vol))
                    pos += vol
                    buy_cap -= vol

        buy_cap = self._allowable_buy("INTARIAN_PEPPER_ROOT", pos)
        if buy_cap <= 0 or not depth.buy_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        ceiling = int(fair) - 1
        if ceiling >= fair:
            ceiling -= 1
        raw = max(best_bid + 1, int(fair) - PEPPER_PASSIVE_OFFSET)
        bid_p = min(raw, ceiling)
        if bid_p <= best_bid:
            bid_p = best_bid + 1
        if 0 < bid_p < fair:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_p, buy_cap))

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
            mkt = list(state.market_trades.get(product, []))

            if product == "ASH_COATED_OSMIUM":
                main = self._osmium_orders(depth, fair, position)
                clear = self._clear_at_fair(product, depth, fair, position)
                result[product] = main + clear
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper_orders(depth, fair, position, mkt)

            logger.print(f"{product} pos={position} fair={fair:.1f} n={len(result[product])}")

        td_out = json.dumps(td)
        conversions = 0
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
