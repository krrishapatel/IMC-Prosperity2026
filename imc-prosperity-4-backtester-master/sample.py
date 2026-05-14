import json
from statistics import mean
from typing import Any, Optional

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
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
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    POSITION_LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    def _best_bid_ask(self, order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

    def _allowable_buy(self, product: Symbol, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] - position)

    def _allowable_sell(self, product: Symbol, position: int) -> int:
        return max(0, self.POSITION_LIMITS[product] + position)

    def _place_taking_orders(
        self,
        product: Symbol,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        edge: int,
    ) -> tuple[list[Order], int]:
        orders: list[Order] = []

        buy_remaining = self._allowable_buy(product, position)
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price > fair_value - edge or buy_remaining <= 0:
                break
            available = -order_depth.sell_orders[ask_price]
            qty = min(available, buy_remaining)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                position += qty
                buy_remaining -= qty

        sell_remaining = self._allowable_sell(product, position)
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price < fair_value + edge or sell_remaining <= 0:
                break
            available = order_depth.buy_orders[bid_price]
            qty = min(available, sell_remaining)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                position -= qty
                sell_remaining -= qty

        return orders, position

    def _place_making_orders(
        self,
        product: Symbol,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        base_size: int,
    ) -> list[Order]:
        orders: list[Order] = []
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_capacity = self._allowable_buy(product, position)
        sell_capacity = self._allowable_sell(product, position)
        if buy_capacity <= 0 and sell_capacity <= 0:
            return orders

        buy_quote = min(best_bid + 1, int(fair_value - 1))
        sell_quote = max(best_ask - 1, int(fair_value + 1))
        if buy_quote >= sell_quote:
            buy_quote = best_bid
            sell_quote = best_ask

        skew = position / self.POSITION_LIMITS[product]
        buy_size = min(buy_capacity, max(0, int(base_size * (1.0 - max(0.0, skew)))))
        sell_size = min(sell_capacity, max(0, int(base_size * (1.0 + min(0.0, skew)))))

        if buy_size > 0:
            orders.append(Order(product, buy_quote, buy_size))
        if sell_size > 0:
            orders.append(Order(product, sell_quote, -sell_size))
        return orders

    def _fair_values(self, state: TradingState) -> dict[Symbol, float]:
        mids: dict[Symbol, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid_price(depth)
            if mid is not None:
                mids[product] = mid

        osmium_mid = mids.get("ASH_COATED_OSMIUM", 10_000.0)
        pepper_mid = mids.get("INTARIAN_PEPPER_ROOT", 11_000.0)
        pepper_anchor = 11_000.0

        return {
            "ASH_COATED_OSMIUM": mean([osmium_mid, 10_000.0]),
            "INTARIAN_PEPPER_ROOT": mean([pepper_mid, pepper_anchor]),
        }

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        fair_values = self._fair_values(state)

        for product in self.POSITION_LIMITS:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair_value = fair_values[product]

            take_edge = 1 if product == "INTARIAN_PEPPER_ROOT" else 2
            make_size = 12 if product == "INTARIAN_PEPPER_ROOT" else 15

            orders, post_take_position = self._place_taking_orders(
                product, order_depth, fair_value, position, take_edge
            )
            orders.extend(
                self._place_making_orders(product, order_depth, fair_value, post_take_position, make_size)
            )
            result[product] = orders

            logger.print(
                f"{product} pos={position} fair={fair_value:.1f} "
                f"bids={len(order_depth.buy_orders)} asks={len(order_depth.sell_orders)} "
                f"orders={len(orders)}"
            )

        trader_data = ""
        conversions = 0

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
